from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


class Config:
    """config.yaml 的轻量封装，支持点号取值与环境变量覆盖。"""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self._data = data or {}
        self.path = path

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        p = Path(path) if path else ROOT / "config.yaml"
        if not p.exists():
            raise FileNotFoundError(f"找不到配置文件: {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return cls(data, p.resolve())

    def get(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self._data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        cur = self._data
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = value

    @property
    def vault_path(self) -> Path:
        raw = os.environ.get("OBSIDIAN_VAULT") or self.get("vault_path")
        if not raw:
            raise ValueError("未配置 vault_path：请在 config.yaml 填写，或设置环境变量 OBSIDIAN_VAULT")
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (self.path.parent / p) if self.path else (ROOT / p)
        if not p.exists():
            raise FileNotFoundError(f"Obsidian 仓库不存在: {p}")
        return p.resolve()

    @property
    def index_dir(self) -> Path:
        raw = os.environ.get("RAG_INDEX_DIR") or self.get("index_dir", "./.lancedb")
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (self.path.parent / p) if self.path else (ROOT / p)
        p.mkdir(parents=True, exist_ok=True)
        return p
