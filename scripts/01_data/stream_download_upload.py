#!/usr/bin/env python3
"""Download one symbol at a time, upload it, and optionally reclaim local space.

The queue is deliberately stateful.  A successful remote copy is recorded as
``completed`` before the next symbol starts, so an interrupted run can resume
without redownloading completed symbols.  SSH keys are preferred; password
automation is supported only through an explicitly named environment variable
and requires ``sshpass`` to be installed.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols:
        values = args.symbols.split(",")
    elif args.symbols_file:
        values = Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
    else:
        universe = json.loads(Path(args.universe).read_text(encoding="utf-8"))
        values = universe["symbols"]
    result = []
    for value in values:
        symbol = value.strip().upper()
        if symbol and symbol not in result:
            result.append(symbol)
    if not result:
        raise SystemExit("No symbols supplied")
    return result


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load_state(path: Path, symbols: list[str]) -> dict:
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {"created_at": _now(), "items": {}}
    items = state.setdefault("items", {})
    for symbol in symbols:
        items.setdefault(symbol, {"status": "pending", "attempts": 0})
    return state


def _remote_parts(remote: str) -> tuple[str, str]:
    if ":" not in remote:
        raise SystemExit("--remote must look like user@host:/absolute/path/data")
    host, path = remote.split(":", 1)
    if not host or not path.startswith("/"):
        raise SystemExit("--remote must contain an absolute remote path")
    return host, path.rstrip("/")


def _ssh_prefix(args: argparse.Namespace) -> list[str]:
    prefix = []
    if args.sshpass_env:
        if not shutil.which("sshpass"):
            raise SystemExit("--sshpass-env was set, but sshpass is not installed")
        if not os.environ.get(args.sshpass_env):
            raise SystemExit(f"Environment variable {args.sshpass_env} is empty")
        prefix = ["sshpass", "-e"]
    return prefix


def _ssh_options(args: argparse.Namespace) -> list[str]:
    # Keep password authentication interactive when sshpass is not installed.
    # BatchMode would reject the password before ssh can prompt for it.
    return [
        "-p", str(args.remote_port),
        "-o", "ControlMaster=auto",
        "-o", f"ControlPersist={args.control_persist}",
        "-o", "ControlPath=/tmp/predict-cryptos-ssh-%C",
    ]


def _ssh_command(args: argparse.Namespace, host: str, command: str) -> list[str]:
    return _ssh_prefix(args) + ["ssh", *_ssh_options(args), host, command]


def _rsync_command(args: argparse.Namespace, source: Path, remote: str) -> list[str]:
    return _ssh_prefix(args) + [
        "rsync", "-a", "--partial", "--checksum", "-e",
        "ssh " + " ".join(shlex.quote(x) for x in _ssh_options(args)),
        str(source), remote,
    ]


def _run(command: list[str], *, label: str) -> None:
    print("+", " ".join(shlex.quote(x) for x in command), flush=True)
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[2])
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")


def _quarantine_corrupt(path: Path, symbol: str) -> None:
    if not path.exists():
        return
    try:
        import pandas as pd
        pd.read_parquet(path, columns=["open_time"])
    except Exception:
        target_dir = path.parent.parent / "corrupt_klines"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{symbol}.parquet.corrupt.queue.{int(time.time())}"
        shutil.move(path, target)
        print(f"Quarantined corrupt file: {path} -> {target}", flush=True)


def _requested_range_ms(start_date: str, end_date: str) -> tuple[int, int]:
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    start_dt = datetime.combine(start, dt_time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end, dt_time.max, tzinfo=timezone.utc)
    return int(start_dt.timestamp() * 1000), int(end_dt.timestamp() * 1000) // 60_000 * 60_000


def _local_covers(path: Path, column: str, start_ms: int, end_ms: int, *, tail_ms: int = 0) -> bool:
    """Check coverage using only the timestamp column; corrupt files return false."""
    if not path.exists():
        return False
    try:
        import pandas as pd
        frame = pd.read_parquet(path, columns=[column])
        if frame.empty:
            return False
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        return not values.empty and int(values.min()) <= start_ms and int(values.max()) >= end_ms - tail_ms
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbols", default="")
    ap.add_argument("--symbols-file", default="")
    ap.add_argument("--universe", default="data/universe/top40.json")
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--remote", required=True, help="user@host:/absolute/path/data")
    ap.add_argument("--remote-port", type=int, default=22)
    ap.add_argument("--control-persist", default="10m",
                    help="Keep the SSH multiplexed connection open for this long")
    ap.add_argument("--state", default="data/download_queue.json")
    ap.add_argument("--delete-local", action="store_true",
                    help="Delete local kline/funding files only after rsync succeeds")
    ap.add_argument("--skip-positioning", action="store_true")
    ap.add_argument("--sshpass-env", default="",
                    help="Environment variable containing SSH password; SSH keys are preferred")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent symbol download/upload workers; 2 is a safe starting point")
    ap.add_argument("--download-attempts", type=int, default=3,
                    help="Attempts per symbol when Binance/proxy resets a connection")
    ap.add_argument("--download-backoff", type=float, default=15.0,
                    help="Seconds before retrying a failed symbol download")
    args = ap.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.download_attempts < 1:
        raise SystemExit("--download-attempts must be >= 1")
    if args.symbols and args.symbols_file:
        raise SystemExit("Use only one of --symbols and --symbols-file")

    symbols = _load_symbols(args)
    state_path = Path(args.state)
    state = _load_state(state_path, symbols)
    remote_host, remote_path = _remote_parts(args.remote)
    root = Path(__file__).resolve().parents[2]
    klines_dir = root / "data" / "klines"
    funding_dir = root / "data" / "funding"
    start_ms, end_ms = _requested_range_ms(args.start_date, args.end_date)
    flags = ["--start-date", args.start_date, "--end-date", args.end_date]
    if args.proxy:
        flags += ["--proxy", args.proxy]
    if args.skip_positioning:
        flags.append("--skip-positioning")

    _run(_ssh_command(
        args, remote_host,
        "mkdir -p " + shlex.quote(remote_path + "/klines") + " "
        + shlex.quote(remote_path + "/funding"),
    ), label="remote directory creation")

    state_lock = threading.Lock()

    def process_symbol(symbol: str) -> bool:
        item = state["items"][symbol]
        if item.get("status") == "completed":
            print(f"SKIP {symbol}: already completed", flush=True)
            if args.delete_local:
                for local_file in (
                    klines_dir / f"{symbol}.parquet",
                    funding_dir / f"{symbol}.parquet",
                ):
                    if local_file.exists():
                        local_file.unlink()
                        print(f"Deleted local completed copy: {local_file}", flush=True)
            return True
        if item.get("status") == "failed" and not args.retry_failed:
            print(f"SKIP {symbol}: failed previously (use --retry-failed)", flush=True)
            return True
        with state_lock:
            item.update(status="running", started_at=_now(), attempts=int(item.get("attempts", 0)) + 1)
            _save_state(state_path, state)
        try:
            kline = klines_dir / f"{symbol}.parquet"
            _quarantine_corrupt(kline, symbol)
            funding = funding_dir / f"{symbol}.parquet"
            kline_ready = _local_covers(kline, "open_time", start_ms, end_ms)
            # Funding is sampled every 8h, so the last event can precede the
            # end of the requested candle date by less than one day.
            funding_ready = _local_covers(
                funding, "funding_time", start_ms, end_ms, tail_ms=24 * 60 * 60 * 1000
            )
            if kline_ready and funding_ready:
                print(f"REUSE {symbol}: local kline/funding already cover requested range", flush=True)
            else:
                missing = []
                if not kline_ready:
                    missing.append("klines")
                if not funding_ready:
                    missing.append("funding")
                print(f"DOWNLOAD {symbol}: missing/incomplete {','.join(missing)}", flush=True)
                command = [sys.executable, "scripts/01_data/download_data.py", "--symbols", symbol, *flags]
                last_error = None
                for attempt in range(1, args.download_attempts + 1):
                    try:
                        _run(command, label=f"download {symbol} attempt {attempt}")
                        last_error = None
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt < args.download_attempts:
                            print(f"RETRY {symbol}: download attempt {attempt} failed; waiting {args.download_backoff:g}s", flush=True)
                            time.sleep(args.download_backoff * attempt)
                if last_error is not None:
                    raise last_error
            sources = [p for p in (klines_dir / f"{symbol}.parquet", funding_dir / f"{symbol}.parquet") if p.exists()]
            if not sources:
                raise RuntimeError(f"download produced no Parquet files for {symbol}")
            for source in sources:
                remote_file = remote_path + ("/klines/" if source.parent == klines_dir else "/funding/") + source.name
                _run(_rsync_command(args, source, f"{remote_host}:{remote_file}"), label=f"upload {source.name}")
                _run(_ssh_command(args, remote_host, "test -s " + shlex.quote(remote_file)), label=f"verify {remote_file}")
            if args.delete_local:
                for source in sources:
                    source.unlink()
                print(f"Deleted local copies after verified upload: {symbol}", flush=True)
            with state_lock:
                item.update(status="completed", completed_at=_now(), error=None)
                _save_state(state_path, state)
            print(f"COMPLETED {symbol}", flush=True)
            return True
        except Exception as exc:
            with state_lock:
                item.update(status="failed", failed_at=_now(), error=f"{type(exc).__name__}: {exc}")
                _save_state(state_path, state)
            print(f"FAILED {symbol}: {exc}", file=sys.stderr, flush=True)
            return False

    failed = False
    if args.workers == 1:
        for symbol in symbols:
            if not process_symbol(symbol):
                failed = True
    else:
        print(f"Running {len(symbols)} symbols with {args.workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="symbol") as pool:
            futures = {pool.submit(process_symbol, symbol): symbol for symbol in symbols}
            for future in as_completed(futures):
                if not future.result():
                    failed = True
    print(f"Queue state: {state_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
