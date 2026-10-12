"""Persist every generated factor expression and its result directory."""

from __future__ import annotations

import datetime as dt
import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def _safe_resolve(path: str | os.PathLike[str]) -> Path:
    """Resolve paths even if the caller's cwd was replaced during a run."""
    candidate = Path(path)
    if candidate.is_absolute():
        return Path(os.path.normpath(os.fspath(candidate)))
    try:
        return candidate.resolve()
    except FileNotFoundError:
        # Relative registry paths are interpreted next to this project when
        # the shell cwd no longer exists.
        return Path(os.path.normpath(os.path.join(os.path.dirname(__file__), os.fspath(candidate))))


def _safe_factor_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("factor_id must not be empty")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def _markdown_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _code_block(value: Any) -> str:
    code = "" if value is None else str(value)
    return code.replace("```", "`` `")


def write_factor_registry(
    records: Iterable[dict[str, Any]],
    markdown_path: str | os.PathLike[str],
    result_root: str | os.PathLike[str],
    *,
    title: str = "因子表达式注册表",
) -> Path:
    """Write a complete Markdown registry for generated factors.

    ``records`` may include accepted and rejected factors.  Every record gets a
    deterministic result directory under ``result_root``; rejected factors are
    still listed so that no generated expression is lost.
    """
    markdown_path = Path(markdown_path)
    result_root = Path(result_root)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    result_root.mkdir(parents=True, exist_ok=True)

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records:
        record = dict(raw)
        factor_id = _safe_factor_id(record.get("factor_id"))
        if factor_id in seen:
            raise ValueError(f"duplicate factor_id: {factor_id}")
        seen.add(factor_id)
        record["factor_id"] = factor_id
        record["result_dir"] = str(
            _safe_resolve(record.get("result_dir") or result_root / factor_id)
        )
        record.setdefault("status", "generated")
        record.setdefault("block", "unknown")
        record.setdefault("fold", "")
        record.setdefault("name", factor_id)
        record.setdefault("formula", "")
        record.setdefault("normalized_formula", "")
        record.setdefault("code", "")
        record.setdefault("rejection_reason", "")
        normalized.append(record)

    normalized.sort(key=lambda item: item["factor_id"])
    lines = [
        f"# {title}",
        "",
        f"> 生成时间：{dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "本文件记录所有生成过的因子表达式，包括最终淘汰的候选。每个因子都分配了独立的结果目录；只有通过筛选的因子才会实际写入信号文件。",
        "",
        "## 总览",
        "",
        "| factor_id | block | fold | status | result_dir | formula |",
        "|---|---|---|---|---|---|",
    ]
    for record in normalized:
        lines.append(
            "| "
            + " | ".join(
                _markdown_text(record.get(key))
                for key in ("factor_id", "block", "fold", "status", "result_dir", "formula")
            )
            + " |"
        )

    lines.extend(["", "## 因子详情", ""])
    for record in normalized:
        lines.extend(
            [
                f"### `{record['factor_id']}`",
                "",
                f"- **名称**：{_markdown_text(record.get('name'))}",
                f"- **板块**：{_markdown_text(record.get('block'))}",
                f"- **Fold**：{_markdown_text(record.get('fold'))}",
                f"- **状态**：{_markdown_text(record.get('status'))}",
                f"- **结果文件夹**：`{record['result_dir']}`",
                f"- **金融假设**：{_markdown_text(record.get('hypothesis'))}",
                f"- **预测目标**：{_markdown_text(record.get('target_horizon') or 'next_30min')}",
                f"- **字段依赖**：{_markdown_text(record.get('required_columns'))}",
                f"- **窗口**：{_markdown_text(record.get('lookback_bars'))}",
                f"- **公式**：`{_markdown_text(record.get('formula'))}`",
                f"- **规范化公式**：`{_markdown_text(record.get('normalized_formula'))}`",
                f"- **相关性/淘汰说明**：{_markdown_text(record.get('rejection_reason'))}",
                "",
                "```python",
                _code_block(record.get("code")),
                "```",
                "",
            ]
        )

    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read one factor metadata object per line."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} must contain a JSON object")
            records.append(record)
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write factor metadata JSONL to Markdown")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--result-root", required=True)
    args = parser.parse_args()
    write_factor_registry(
        read_jsonl(args.input_jsonl),
        args.markdown,
        args.result_root,
    )
