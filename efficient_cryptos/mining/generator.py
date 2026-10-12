"""Generate factor metadata with OpenAI, DeepSeek, or an OpenAI-compatible Grok relay.

This script only generates and registers expressions.  Numeric evaluation is
performed locally by ``numba_engine.py`` and never sends market data to the API.

Example (safe prompt inspection):
    python codex_factor_generator.py --dry-run --blocks momentum reversal

Example (OpenAI):
    export OPENAI_API_KEY='...'
    export OPENAI_MODEL='your-code-generation-model'
    python codex_factor_generator.py --blocks all --fold-id fold_1 \
      --candidate-count 20 --batches 2

Example (DeepSeek):
    export DEEPSEEK_API_KEY='...'
    python codex_factor_generator.py --provider deepseek \
      --model deepseek-chat --blocks all --total-candidates 2000 --batches 20

Example (Grok/OpenAI-compatible relay):
    export GROK_API_KEY='...'
    export GROK_BASE_URL='https://relay.example/v1'
    python codex_factor_generator.py --provider grok --model grok-4 \
      --blocks all --total-candidates 2000 --batches 20
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any

try:
    from openai import OpenAI
except ImportError:  # Allow --dry-run without the optional API client.
    OpenAI = Any  # type: ignore[misc,assignment]

from efficient_cryptos.mining.dsl import compile_factor_record
from efficient_cryptos.mining.registry import read_jsonl, write_factor_registry


def _safe_display_path(path: str | os.PathLike[str]) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return os.path.normpath(os.fspath(candidate))
    try:
        return os.fspath(candidate.resolve())
    except FileNotFoundError:
        return os.path.normpath(os.path.join(os.path.dirname(__file__), os.fspath(candidate)))


ROOT = Path(__file__).resolve().parent
PROMPT_ROOT = ROOT / "prompts" / "crypto"
PROMPT_ROOTS = {name: ROOT / "prompts" / name for name in ("crypto", "dex", "meme")}
BLOCKS = [
    "momentum",
    "reversal",
    "volatility",
    "volume",
    "liquidity",
    "price_volume",
    "microstructure",
    "intraday",
    "nonlinear",
    "style",
    "sentiment",
    "flow",
]

FUNDING_EVENT_BLOCK = "funding_event"
FUNDING_EVENT_TARGET = "next_funding_event_excess_return"


FACTOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "factors": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "factor_id": {"type": "string"},
                    "block": {"type": "string"},
                    "name": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "target_horizon": {"type": "string"},
                    "direction": {"type": "string"},
                    "required_columns": {"type": "array", "items": {"type": "string"}},
                    "lookback_bars": {"type": "array", "items": {"type": "integer"}},
                    "formula": {"type": "string"},
                    "normalized_formula": {"type": "string"},
                    "complexity": {"type": "integer"},
                    "reusable_nodes": {"type": "array", "items": {"type": "string"}},
                    "leakage_check": {"type": "string"},
                    "code": {"type": "string"},
                    "status": {"type": "string"},
                    "rejection_reason": {"type": "string"},
                    "rejection_rules": {"type": "array", "items": {"type": "string"}},
                    "expected_failure_modes": {"type": "array", "items": {"type": "string"}},
                },
                "required": [
                    "factor_id", "block", "name", "hypothesis", "target_horizon",
                    "direction", "required_columns", "lookback_bars", "formula",
                    "normalized_formula", "complexity", "reusable_nodes", "leakage_check",
                    "code", "status", "rejection_reason", "rejection_rules",
                    "expected_failure_modes",
                ],
            },
        }
    },
    "required": ["factors"],
}


def _read_prompt(name: str, prompt_root: Path = PROMPT_ROOT) -> str:
    return (prompt_root / name).read_text(encoding="utf-8")


def _existing_summary(path: Path | None, limit: int = 300) -> str:
    if path is None or not path.exists():
        return "(no existing factor metadata)"
    records = read_jsonl(path)
    lines = []
    for record in records[-limit:]:
        lines.append(
            f"{record.get('factor_id')} | {record.get('block')} | "
            f"{record.get('normalized_formula') or record.get('formula')}"
        )
    return "\n".join(lines) or "(no existing factor metadata)"


def _records_summary(
    records: list[dict[str, Any]],
    block: str,
    limit: int = 300,
) -> str:
    """Prefer same-block formulas while keeping resume prompts token-efficient."""
    same_block = [record for record in records if record.get("block") == block]
    other = [record for record in records if record.get("block") != block]
    selected = (same_block[-limit:] + other[-50:])[-limit:]
    lines = [
        f"{record.get('factor_id')} | {record.get('block')} | "
        f"{record.get('normalized_formula') or record.get('formula')}"
        for record in selected
    ]
    return "\n".join(lines) or "(no existing factor metadata)"


def build_prompt(
    block: str,
    fold_id: str,
    train_start: str,
    train_end: str,
    validation_start: str,
    validation_end: str,
    candidate_count: int,
    batch_id: str,
    existing_summary: str,
    prompt_root: Path = PROMPT_ROOT,
) -> str:
    is_funding_event = block == FUNDING_EVENT_BLOCK
    target_horizon = FUNDING_EVENT_TARGET if is_funding_event else "next_30min"
    target_description = (
        "从当前 bar 到该 symbol 下一次真实 Binance funding_time 的市场中性超额收益。"
        "不要把 6 小时当成硬编码标签；以数据中的 funding_time 为准，实际间隔可能是 4/6/8 小时。"
        if is_funding_event else
        "当前 bar 之后约 30 分钟的市场中性超额收益。"
    )
    sections = [
            _read_prompt("base_factor_prompt.md", prompt_root),
            _read_prompt("economic_principles.md", prompt_root),
            _read_prompt(f"{block}.md", prompt_root),
            _read_prompt("fold_prompt_template.md", prompt_root),
        ]
    override = prompt_root / "crypto_overrides.md"
    if override.exists():
        # Put the target-specific contract after the inherited Dpsk prompts so
        # it wins over stock/A-share wording in the generic block templates.
        sections.append(_read_prompt("crypto_overrides.md", prompt_root))
    if is_funding_event:
        event_override = prompt_root / "funding_event_overrides.md"
        if event_override.exists():
            sections.append(_read_prompt("funding_event_overrides.md", prompt_root))
    sections.append(
        f"""Fill the following task values exactly:
fold_id: {fold_id}
train_start: {train_start}
train_end: {train_end}
validation_start: {validation_start}
validation_end: {validation_end}
block: {block}
batch_id: {batch_id}
candidate_count: {candidate_count}
target_horizon: {target_horizon}
target_definition: {target_description}

Existing factor summary (for de-duplication only; do not fit to validation/OOS):
{existing_summary}

Return exactly one JSON object matching the provided schema with exactly {candidate_count} factors.
Do not use Markdown fences or explanatory text outside the JSON object.
All factors must target {target_horizon} and must include a genuine economic hypothesis.
"""
    )
    return "\n\n".join(
        sections
    )


def _extract_json(response: Any) -> dict[str, Any]:
    text = response.output_text.strip()
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("factors"), list):
        raise ValueError("model response must be an object containing a factors array")
    return data


def _extract_deepseek_factors(content: str) -> list[dict[str, Any]]:
    """Parse DeepSeek JSON despite occasional fences/prose/control chars.

    ``response_format=json_object`` is advisory for some compatible endpoints:
    models may still wrap the object in a code fence, prepend a sentence, or
    emit literal newlines inside the generated ``code`` string.  Recover the
    first JSON object/array and validate the expected factors list locally.
    """
    raw = str(content or "").strip()
    if not raw:
        raise ValueError("DeepSeek returned empty content")
    candidates = [raw]
    if "```" in raw:
        fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE | re.DOTALL).strip()
        candidates.append(fenced)
    for candidate in candidates:
        try:
            data = json.loads(candidate, strict=False)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder(strict=False)
            starts = [pos for pos, char in enumerate(candidate) if char in "[{]"]
            data = None
            for start in starts:
                try:
                    data, _ = decoder.raw_decode(candidate[start:])
                    break
                except json.JSONDecodeError:
                    continue
            if data is None:
                continue
        if isinstance(data, list):
            factors = data
        elif isinstance(data, dict):
            factors = data.get("factors")
            if factors is None:
                for key in ("data", "items", "results"):
                    if isinstance(data.get(key), list):
                        factors = data[key]
                        break
            # Some compatible endpoints ignore the batch wrapper and return
            # one factor object directly.  It is still a valid candidate;
            # return it as a one-item batch so checkpoint accounting can ask
            # for the missing candidates on the next resume.
            if factors is None and data.get("factor_id"):
                factors = [data]
        else:
            factors = None
        if isinstance(factors, list) and all(isinstance(item, dict) for item in factors):
            return factors
    preview = re.sub(r"\s+", " ", raw)[:240]
    raise ValueError(f"DeepSeek response must contain a factors array; preview={preview!r}")


def request_batch(
    client: OpenAI,
    model: str,
    prompt: str,
    provider: str = "openai",
    reasoning_effort: str | None = None,
    max_retries: int = 4,
) -> list[dict[str, Any]]:
    for attempt in range(max_retries):
        try:
            if provider == "openai":
                request: dict[str, Any] = {
                    "model": model,
                    "input": prompt,
                    "text": {
                        "format": {
                            "type": "json_schema",
                            "name": "factor_batch",
                            "strict": True,
                            "schema": FACTOR_SCHEMA,
                        }
                    },
                }
                if reasoning_effort:
                    request["reasoning"] = {"effort": reasoning_effort}
                response = client.responses.create(**request)
                return _extract_json(response)["factors"]

            if provider in {"deepseek", "grok"}:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": "Return only valid JSON. The top-level object must contain a factors array.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return _extract_deepseek_factors(content)

            raise ValueError(f"unsupported provider: {provider}")
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code in {400, 401, 402, 403, 404}:
                raise
            if attempt >= max_retries - 1:
                raise
            time.sleep(min(30.0, (2**attempt) + random.random()))
    raise RuntimeError("unreachable")


def _validate_records(records: list[dict[str, Any]], block: str, fold_id: str) -> list[dict[str, Any]]:
    result = []
    for record in records:
        factor_id = str(record.get("factor_id", "")).strip()
        if not factor_id:
            continue
        record = dict(record)
        record["block"] = block
        record["fold"] = fold_id
        record.setdefault("status", "generated")
        result.append(record)
    return result


def _make_unique_ids(
    records: list[dict[str, Any]],
    used: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Avoid the model reusing an ID across batches or resumed runs."""
    used = used if used is not None else set()
    result: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        base = str(item["factor_id"]).strip()
        candidate = base
        suffix = 1
        while candidate in used:
            candidate = f"{base}__{item.get('batch_id', 'batch')}__{suffix}"
            suffix += 1
        item["factor_id"] = candidate
        used.add(candidate)
        result.append(item)
    return result


def _formula_signature(record: dict[str, Any]) -> str:
    """Create a stable exact-expression signature for generation-time de-duplication."""
    # Prefer the executable AST signature: it removes whitespace, commutative
    # operand order, and equivalent DSL spellings before falling back to the
    # model-provided mathematical formula.
    expression, reason = compile_factor_record(record)
    if reason is None and expression is not None:
        try:
            from numba_engine import canonical_formula
            return canonical_formula(expression)
        except Exception:
            pass
    value = record.get("normalized_formula") or record.get("formula") or ""
    text = str(value).strip().lower()
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    return re.sub(r"\s+", "", text)


def main() -> None:
    # ``spawn`` and Path.resolve() call os.getcwd().  If the project directory
    # was replaced during a long generation run, repair the stale shell cwd.
    try:
        os.getcwd()
    except FileNotFoundError:
        os.chdir(Path(__file__).resolve().parent)
    parser = argparse.ArgumentParser(description="Generate factor expressions with OpenAI")
    parser.add_argument("--blocks", nargs="+", default=BLOCKS)
    parser.add_argument("--fold-id", default="fold_1")
    parser.add_argument("--train-start", default="20220101")
    parser.add_argument("--train-end", default="20221231")
    parser.add_argument("--validation-start", default="20230101")
    parser.add_argument("--validation-end", default="20230630")
    parser.add_argument("--candidate-count", type=int, default=10)
    parser.add_argument("--total-candidates", type=int, help="total candidates across all selected blocks and batches")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--model",
        default=os.getenv("FACTOR_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("DEEPSEEK_MODEL"),
    )
    parser.add_argument(
        "--provider",
            choices=["openai", "deepseek", "grok"],
        default=os.getenv("FACTOR_PROVIDER", "openai"),
    )
    parser.add_argument("--base-url", help="API base URL; otherwise provider-specific environment/default is used")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=os.getenv("OPENAI_REASONING_EFFORT", "medium"),
    )
    parser.add_argument(
        "--existing-jsonl",
        type=Path,
        nargs="+",
        help="one or more prior JSONL files whose IDs/formulas must not be regenerated",
    )
    parser.add_argument("--output-jsonl", type=Path, default=Path("factor_metadata.jsonl"))
    parser.add_argument(
        "--rejected-jsonl",
        type=Path,
        help="duplicate/rejected expression audit; defaults beside output-jsonl",
    )
    parser.add_argument("--registry-md", type=Path, default=Path("factor_expression_registry.md"))
    parser.add_argument("--result-root", type=Path, default=Path("factor_results"))
    parser.add_argument("--prompt-root", type=Path, default=PROMPT_ROOT,
                        help="prompt directory (default: Binance crypto prompts); dex/meme prompts live beside it")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="do not skip completed batches already present in output-jsonl",
    )
    args = parser.parse_args()

    if args.blocks == ["all"]:
        args.blocks = BLOCKS.copy()
    # Target-specific prompt roots may define their own block names (for
    # example DEX flow/liquidity blocks).  Keep the stock allow-list strict,
    # while accepting a custom block only when its prompt file exists.
    custom_blocks = {
        block for block in args.blocks
        if (args.prompt_root / f"{block}.md").exists()
    }
    unknown = sorted(set(args.blocks) - set(BLOCKS) - custom_blocks)
    if unknown:
        parser.error(f"unknown blocks: {unknown}")
    if args.candidate_count <= 0 or args.batches <= 0 or args.concurrency <= 0:
        parser.error("candidate-count, batches and concurrency must be positive")
    if args.total_candidates is not None and args.total_candidates <= 0:
        parser.error("total-candidates must be positive")
    if not args.dry_run and not args.model:
        parser.error("set --model or OPENAI_MODEL")

    existing_output = read_jsonl(args.output_jsonl) if args.output_jsonl.exists() else []
    rejected_jsonl = args.rejected_jsonl or args.output_jsonl.with_name(
        f"{args.output_jsonl.stem}.rejected.jsonl"
    )
    existing_rejected = read_jsonl(rejected_jsonl) if rejected_jsonl.exists() else []
    if args.no_resume and existing_output:
        parser.error("--no-resume requires a new/empty --output-jsonl to avoid duplicate records")
    reference_records: list[dict[str, Any]] = []
    for reference_path in args.existing_jsonl or []:
        if not reference_path.exists():
            parser.error(f"existing-jsonl is not a file: {reference_path}")
        reference_records.extend(read_jsonl(reference_path))

    existing_batch_counts: dict[str, int] = {}
    for record in existing_output:
        batch_id = str(record.get("batch_id", "")).strip()
        if batch_id:
            existing_batch_counts[batch_id] = existing_batch_counts.get(batch_id, 0) + 1

    tasks: list[tuple[str, str, int, str]] = []
    if args.total_candidates is None:
        block_batch_counts = {
            (block, batch): args.candidate_count
            for block in args.blocks
            for batch in range(args.batches)
        }
    else:
        total_slots = len(args.blocks) * args.batches
        if args.total_candidates < total_slots:
            parser.error("total-candidates must be at least blocks × batches")
        base, remainder = divmod(args.total_candidates, total_slots)
        block_batch_counts = {}
        slot = 0
        for block in args.blocks:
            for batch in range(args.batches):
                block_batch_counts[(block, batch)] = base + (1 if slot < remainder else 0)
                slot += 1

    for block in args.blocks:
        for batch in range(args.batches):
            batch_id = f"{args.fold_id}_{block}_{batch + 1:04d}"
            expected_count = block_batch_counts[(block, batch)]
            completed_count = 0 if args.no_resume else existing_batch_counts.get(batch_id, 0)
            candidate_count = max(0, expected_count - completed_count)
            if candidate_count == 0:
                continue
            existing = _records_summary(reference_records + existing_output, block)
            tasks.append(
                (block, batch_id, candidate_count, build_prompt(
                    block, args.fold_id, args.train_start, args.train_end,
                    args.validation_start, args.validation_end,
                    candidate_count, batch_id, existing, args.prompt_root,
                ))
            )

    if args.dry_run:
        if tasks:
            print(tasks[0][3])
        remaining = sum(task[2] for task in tasks)
        print(
            f"\n# dry-run: {len(tasks)} batches, {remaining} candidates remaining, "
            f"{len(existing_output)} records already saved"
        )
        return

    if not tasks:
        write_factor_registry(existing_output + existing_rejected, args.registry_md, args.result_root)
        print(f"generated=0 total_registered={len(existing_output)} remaining=0")
        print("all requested batches are already complete")
        return

    if OpenAI is Any:
        parser.error("install the optional 'openai' package before API generation")

    if args.provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = args.base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    elif args.provider == "grok":
        api_key = os.getenv("GROK_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = args.base_url or os.getenv("GROK_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        required_key = {
            "deepseek": "DEEPSEEK_API_KEY",
            "grok": "GROK_API_KEY",
            "openai": "OPENAI_API_KEY",
        }[args.provider]
        parser.error(f"set {required_key}")
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)
    records: list[dict[str, Any]] = []
    errors: list[tuple[str, str, str]] = []
    completed = 0
    balance_exhausted = False
    used_ids = {
        str(record.get("factor_id", "")).strip()
        for record in reference_records + existing_output + existing_rejected
        if str(record.get("factor_id", "")).strip()
    }
    used_formulas = {
        signature
        for record in reference_records + existing_output + existing_rejected
        if (signature := _formula_signature(record))
    }
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.output_jsonl.open("a", encoding="utf-8") as output_handle,
        rejected_jsonl.open("a", encoding="utf-8") as rejected_handle,
        ThreadPoolExecutor(max_workers=args.concurrency) as executor,
    ):
        futures = {
                executor.submit(
                    request_batch,
                    client,
                    args.model,
                    prompt,
                    args.provider,
                    None if args.reasoning_effort == "none" else args.reasoning_effort,
                ): (block, batch_id, candidate_count)
            for block, batch_id, candidate_count, prompt in tasks
        }
        for future in as_completed(futures):
            block, batch_id, candidate_count = futures[future]
            try:
                raw_records = future.result()
            except Exception as exc:
                if future.cancelled():
                    continue
                errors.append((block, batch_id, repr(exc)))
                status_code = getattr(exc, "status_code", None)
                if status_code == 402 or "Insufficient Balance" in str(exc):
                    balance_exhausted = True
                    for pending in futures:
                        pending.cancel()
                completed += 1
                print(
                    f"progress={completed}/{len(tasks)} block={block} batch={batch_id} status=error",
                    flush=True,
                )
                continue
            batch_records = _validate_records(raw_records, block, args.fold_id)
            for record in batch_records:
                record["batch_id"] = batch_id
            batch_records = _make_unique_ids(batch_records, used_ids)
            unique_records: list[dict[str, Any]] = []
            duplicate_records: list[dict[str, Any]] = []
            for record in batch_records:
                signature = _formula_signature(record)
                if not signature or signature in used_formulas:
                    rejected = dict(record)
                    rejected["status"] = "rejected"
                    rejected["rejection_reason"] = (
                        "empty formula" if not signature else "duplicate normalized formula"
                    )
                    duplicate_records.append(rejected)
                    continue
                used_formulas.add(signature)
                unique_records.append(record)
            for record in unique_records:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            for record in duplicate_records:
                rejected_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()
            rejected_handle.flush()
            records.extend(unique_records)
            completed += 1
            print(
                f"progress={completed}/{len(tasks)} block={block} batch={batch_id} "
                f"unique={len(unique_records)}/{candidate_count} "
                f"duplicates={len(duplicate_records)}",
                flush=True,
            )

    all_records = read_jsonl(args.output_jsonl)
    rejected_records = read_jsonl(rejected_jsonl)
    write_factor_registry(all_records + rejected_records, args.registry_md, args.result_root)
    saved_batch_counts: dict[str, int] = {}
    for record in all_records:
        saved_batch_id = str(record.get("batch_id", "")).strip()
        if saved_batch_id:
            saved_batch_counts[saved_batch_id] = saved_batch_counts.get(saved_batch_id, 0) + 1
    remaining = 0
    for (block, batch), expected_count in block_batch_counts.items():
        expected_batch_id = f"{args.fold_id}_{block}_{batch + 1:04d}"
        remaining += max(0, expected_count - saved_batch_counts.get(expected_batch_id, 0))
    print(f"generated={len(records)} total_registered={len(all_records)} remaining={remaining}")
    print(f"rejected_duplicates={len(rejected_records)} rejected_jsonl={_safe_display_path(rejected_jsonl)}")
    print(f"registry={_safe_display_path(args.registry_md)}")
    print(f"result_root={_safe_display_path(args.result_root)}")
    if errors:
        if balance_exhausted:
            print(
                "STOP: DeepSeek balance is insufficient. Recharge, then rerun the same command; "
                "completed batches will be skipped automatically.",
                file=sys.stderr,
            )
        for block, batch_id, error in errors:
            print(f"ERROR block={block} batch={batch_id}: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
