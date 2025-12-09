#!/usr/bin/env python
"""
Render a FinanceBench evaluation JSONL file to a single HTML report grouped by
`eval_classification`. Output is written next to the input file with a .html
extension.

Example:
    python scripts/export_eval_html.py --input data/logs/financebench_eval_20251208_100732.jsonl
"""
from __future__ import annotations

import argparse
import json
import html
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert FinanceBench eval JSONL to an HTML report grouped by eval_classification."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to financebench_eval_*.jsonl",
    )
    return parser.parse_args()


def _esc(text: Any) -> str:
    return html.escape(str(text) if text is not None else "")


def _slug(text: str) -> str:
    return (
        str(text).lower()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("#", "")
        .replace(":", "")
    )


def _format_text_block(title: str, body: str) -> str:
    return (
        f'<div class="block">'
        f'<h3 class="label">{_esc(title)}</h3>'
        f'<div class="text">{_esc(body)}</div>'
        f"</div>"
    )


def _render_evidence(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return '<div class="block"><h3 class="label">Evidence (dataset)</h3><div class="text muted">(none)</div></div>'
    lines = []
    for i, ev in enumerate(rows, 1):
        page = ev.get("evidence_page_num")
        page_str = f"p{page}" if page is not None else "p?"
        lines.append(f"<strong>Evidence {i} ({page_str}):</strong> { _esc(ev.get('evidence_text')) }")
    joined = "<br>".join(lines)
    return (
        '<div class="block"><h3 class="label">Evidence (dataset)</h3>'
        f'<div class="text">{joined}</div></div>'
    )


def _render_citations(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return '<div class="block"><h3 class="label">Citations</h3><div class="text muted">(none)</div></div>'
    items = []
    for row in rows:
        idx = row.get("i")
        text = row.get("text") or ""
        items.append(f"<li><strong>[{_esc(idx)}]</strong> { _esc(text) }</li>")
    return (
        '<div class="block"><h3 class="label">Citations</h3>'
        '<ul class="citations">'
        + "".join(items) +
        "</ul></div>"
    )


def _render_hits(rows: List[Dict[str, Any]], max_len: int = 400) -> str:
    if not rows:
        return '<div class="block"><h3 class="label">Top-K</h3><div class="text muted">(none)</div></div>'
    items = []
    for i, hit in enumerate(rows, 1):
        text = (hit.get("text") or "").replace("\n", " ")
        chunk_type = hit.get("type") or "text"
        page_start = hit.get("page_start")
        page_end = hit.get("page_end")
        if page_start is not None and page_end is not None and page_start != page_end:
            page_str = f"(p{page_start}-{page_end})"
        elif page_start is not None:
            page_str = f"(p{page_start})"
        else:
            page_str = ""
        items.append(
            f'<li><span class="hit-num">[{chunk_type}] {page_str}</span><span class="hit-body"> {text} </span></li>'
        )
    return (
        '<div class="block"><h3 class="label">Top-K</h3>'
        '<ol class="hits">'
        + "".join(items) +
        '</ol></div>'
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input JSONL not found: {args.input}")

    output_path = args.input.with_suffix(".html")

    # Optional dataset evidence backfill
    dataset_path = Path("data/financebench/financebench_open_source.jsonl")
    evidence_idx: Dict[tuple, List[Dict[str, Any]]] = {}
    if dataset_path.exists():
        with dataset_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (row.get("doc_name"), row.get("question"))
                evidence_idx[key] = row.get("evidence") or []
    else:
        print(f"[WARN] Dataset not found for evidence backfill: {dataset_path}")

    records: List[Dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # Backfill missing evidence from dataset
            if not rec.get("evidence"):
                key = (rec.get("doc_name"), rec.get("question"))
                if key in evidence_idx:
                    rec["evidence"] = evidence_idx.get(key)
            records.append(rec)

    grouped = defaultdict(list)
    for rec in records:
        cls = rec.get("eval_classification", "UNKNOWN")
        grouped[cls].append(rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    html_parts: List[str] = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        '<meta charset="utf-8" />',
        "<title>FinanceBench Evaluation Report</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; background: #f8f9fb; color: #111; margin: 2rem; }",
        "h1 { margin-bottom: 0.5rem; }",
        "h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.25rem; }",
        ".card { background: #fff; border: 1px solid #e6e6e6; border-radius: 8px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }",
        ".doc { font-weight: 700; margin-bottom: 8px; }",
        ".label { font-weight: 700; margin-bottom: 4px; }",
        ".block { margin: 10px 0; }",
        ".text { white-space: pre-wrap; line-height: 1.4; }",
        ".muted { color: #777; }",
        ".toc { margin: 0.75rem 0 1.5rem; }",
        ".toc a { margin-right: 12px; text-decoration: none; color: #0c5db9; font-weight: 600; }",
        ".hits { margin: 6px 0 0 1.25rem; padding-left: 0.5rem; }",
        ".hits li { margin: 0.35rem 0; list-style-position: outside; }",
        ".hit-num { font-weight: 700; margin-right: 6px; }",
        ".hit-body { white-space: pre-wrap; }",
        ".citations { margin: 6px 0 0 1.25rem; padding-left: 0.5rem; }",
        ".citations li { margin: 0.25rem 0; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>FinanceBench Evaluation Report</h1>",
        f"<div>Total records: {len(records)}</div>",
    ]

    # Table of contents
    html_parts.append('<div class="toc">')
    for cls in sorted(grouped.keys()):
        anchor = f"cls-{_slug(cls)}"
        html_parts.append(f'<a href="#{anchor}">{_esc(cls)} ({len(grouped[cls])})</a>')
    html_parts.append("</div>")

    for cls in sorted(grouped.keys()):
        items = grouped[cls]
        anchor = f"cls-{_slug(cls)}"
        html_parts.append(f'<h2 id="{anchor}">{_esc(cls)} ({len(items)})</h2>')
        for rec in items:
            html_parts.append('<div class="card">')
            html_parts.append(f'<h2 class="doc">{_esc(rec.get("doc_name", ""))}</h2>')
            
            qtype = rec.get("question_type")
            q_label = f"Question [{qtype}]" if qtype else "Question"
            html_parts.append(_format_text_block(q_label, rec.get("question", "")))
            
            html_parts.append(_format_text_block("Ground Truth", rec.get("ground_truth", "")))
            html_parts.append(_format_text_block("Generated Answer", rec.get("answer", "")))
            
            reasoning = rec.get("reasoning") or rec.get("eval_reasoning") or ""
            html_parts.append(_format_text_block("Eval Reasoning", reasoning))
            
            html_parts.append(_render_evidence(rec.get("evidence") or []))
            html_parts.append(_render_citations(rec.get("citations") or []))
            html_parts.append(_render_hits(rec.get("hits") or []))
            html_parts.append("</div>")

    html_parts.extend(["</body>", "</html>"])

    output_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"[OK] Wrote report to {output_path}")


if __name__ == "__main__":
    main()
