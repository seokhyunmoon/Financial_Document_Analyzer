import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """Write dict rows to a JSONL file, creating parent dirs if needed.

    Args:
        path: Output ``.jsonl`` path.
        rows: Iterable of dictionaries to serialize.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries.

    Args:
        path: Input ``.jsonl`` path.

    Returns:
        Parsed rows as a list of dictionaries.
    """
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
