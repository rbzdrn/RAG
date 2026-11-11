"""问答历史：每次问答追加到 history.jsonl，界面侧边栏可以回看。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .config import ROOT

HISTORY_PATH: Path = ROOT / "history.jsonl"


def append(question: str, answer: str, sources: list[str], elapsed: dict | None = None) -> None:
    """追加一条问答记录。"""
    record = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": sources,
        "elapsed": elapsed or {},
    }
    with HISTORY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def load(limit: int = 50) -> list[dict]:
    """读取最近的记录，最新的在前。"""
    if not HISTORY_PATH.exists():
        return []
    records: list[dict] = []
    with HISTORY_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return list(reversed(records[-limit:]))


def clear() -> None:
    """清空历史文件。"""
    HISTORY_PATH.write_text("", encoding="utf-8")
