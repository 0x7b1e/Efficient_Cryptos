#!/usr/bin/env python3
"""Download a date range locally, upload it, and merge it on a remote server.

Only the requested range is kept locally (under /tmp by default). The remote
Parquet is merged by timestamp and replaced atomically, so historical rows are
preserved and duplicate timestamps are removed.
"""
from __future__ import annotations
import argparse, base64, os, shlex, shutil, subprocess, tempfile
from pathlib import Path

MERGE_CODE = r'''import os, sys, tempfile
from pathlib import Path
import pandas as pd

def merge(old_path, delta_path, column):
    old = pd.read_parquet(old_path) if os.path.exists(old_path) else pd.DataFrame()
    delta = pd.read_parquet(delta_path) if os.path.exists(delta_path) else pd.DataFrame()
    if old.empty and delta.empty:
        raise RuntimeError("both old and delta files are empty")
    out = pd.concat([old, delta], ignore_index=True)
    out[column] = pd.to_numeric(out[column], errors="coerce")
    out = out.dropna(subset=[column]).drop_duplicates(subset=[column], keep="last")
    out = out.sort_values(column).reset_index(drop=True)
    Path(old_path).parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=Path(old_path).name + ".", suffix=".tmp", dir=Path(old_path).parent)
    os.close(fd)
    try:
        out.to_parquet(tmp, index=False)
        os.replace(tmp, old_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print(old_path, len(old), "+", len(delta), "=>", len(out), flush=True)

root, incoming = sys.argv[1], sys.argv[2]
for symbol in sys.argv[3].split(','):
    symbol = symbol.strip().upper()
    if not symbol: continue
    merge(f"{root}/klines/{symbol}.parquet", f"{incoming}/klines/{symbol}.parquet", "open_time")
    merge(f"{root}/funding/{symbol}.parquet", f"{incoming}/funding/{symbol}.parquet", "funding_time")
'''


def run(cmd, cwd=None):
    print("+", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", required=True, help="Comma-separated symbols")
    ap.add_argument("--start-date", required=True, help="UTC inclusive, YYYY-MM-DD")
    ap.add_argument("--end-date", required=True, help="UTC inclusive, YYYY-MM-DD")
    ap.add_argument("--remote", required=True, help="user@host:/absolute/data/root")
    ap.add_argument("--remote-port", type=int, default=22)
    ap.add_argument("--proxy", default=None)
    ap.add_argument("--python", default="python", help="Local Python interpreter")
    ap.add_argument("--remote-python", default="/root/autodl-tmp/Efficient_Cryptos/.venv/bin/python")
    ap.add_argument("--keep-local", action="store_true")
    args = ap.parse_args()
    if ":" not in args.remote or not args.remote.split(":", 1)[1].startswith("/"):
        ap.error("--remote must look like user@host:/absolute/path/data")
    host, remote_root = args.remote.split(":", 1)
    symbols = ",".join(dict.fromkeys(x.strip().upper() for x in args.symbols.split(",") if x.strip()))
    if not symbols: ap.error("no symbols")
    project = Path(__file__).resolve().parents[2]
    downloader = project / "scripts" / "01_data" / "download_data.py"
    temp_root = Path(tempfile.mkdtemp(prefix="predict_data_delta_"))
    incoming = f"{remote_root.rstrip('/')}/.incoming_update_{os.getpid()}"
    ssh_opts = ["-p", str(args.remote_port), "-o", "ControlMaster=auto", "-o", "ControlPersist=10m", "-o", "ControlPath=/tmp/predict-cryptos-ssh-%C"]
    ssh = ["ssh", *ssh_opts, host]
    rsync_ssh = "ssh " + " ".join(shlex.quote(x) for x in ssh_opts)
    try:
        # Build a temporary config so download_data writes only the requested delta.
        import yaml
        default_cfg = project / "config" / "default.yaml"
        cfg = yaml.safe_load(default_cfg.read_text(encoding="utf-8"))
        cfg.setdefault("general", {})["data_dir"] = str(temp_root / "data")
        cfg_path = temp_root / "config.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        cmd = [args.python, str(downloader), "--config", str(cfg_path), "--symbols", symbols,
               "--start-date", args.start_date, "--end-date", args.end_date, "--skip-positioning"]
        if args.proxy: cmd += ["--proxy", args.proxy]
        run(cmd, cwd=project)
        run(ssh + [f"mkdir -p {shlex.quote(incoming)}/klines {shlex.quote(incoming)}/funding"], cwd=project)
        for kind in ("klines", "funding"):
            for symbol in symbols.split(","):
                source = temp_root / "data" / kind / f"{symbol}.parquet"
                if not source.exists():
                    raise RuntimeError(f"download did not create {source}")
                target = f"{host}:{incoming}/{kind}/{symbol}.parquet"
                run(["rsync", "-a", "--partial", "--checksum", "-e", rsync_ssh, str(source), target], cwd=project)
        merge_file = temp_root / "merge_remote.py"
        merge_file.write_text(MERGE_CODE, encoding="utf-8")
        run(["rsync", "-a", "-e", rsync_ssh, str(merge_file), f"{host}:{incoming}/merge_remote.py"], cwd=project)
        encoded = base64.b64encode(symbols.encode()).decode()
        remote_cmd = (f"{shlex.quote(args.remote_python)} {shlex.quote(incoming + '/merge_remote.py')} "
                      f"{shlex.quote(remote_root)} {shlex.quote(incoming)} {shlex.quote(symbols)} && "
                      f"rm -rf {shlex.quote(incoming)}")
        run(ssh + [remote_cmd], cwd=project)
        print("REMOTE MERGE COMPLETED", flush=True)
    finally:
        if args.keep_local:
            print(f"Local delta kept at {temp_root}")
        else:
            shutil.rmtree(temp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
