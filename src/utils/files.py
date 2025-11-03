import json
from pathlib import Path
from typing import Iterable, Dict, Any, List


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    """
    Description:
        Write an iterable of dict rows to a JSON Lines file (UTF-8).
        Creates parent directories if not present.

    Args:
        path (str): Output .jsonl path.
        rows (Iterable[Dict[str, Any]]): Dict rows to write.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """
    Description:
        Read a JSON Lines file into a list of dicts.

    Args:
        path (str): Input .jsonl path.

    Returns:
        List[Dict[str, Any]]: Parsed rows.
    """
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out