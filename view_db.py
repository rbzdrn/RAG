"""查看 LanceDB 索引内容的小工具（只读，不会改动数据）。

用法（在项目根目录执行）：
    python view_db.py info                       看有哪些表、多少行、每篇笔记多少块
    python view_db.py find 单例                   按关键词在块正文里搜
    python view_db.py show "错题/状态码.md#s2c0"   看某一条块的完整内容（含所属父块）
    python view_db.py export                     导出 csv，可直接用 Excel 打开
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from rag.config import Config
from rag.store import VectorStore

# 导出时跳过的列：向量太长，Excel 里看不了
SKIP_COLS = {"vector"}


def _esc(text: str) -> str:
    """转义 SQL 里的单引号，防止拼串报错。"""
    return text.replace("'", "''")


def cmd_info(store: VectorStore) -> None:
    print(f"索引目录: {store.dir}")
    for name in ("chunks", "parents"):
        tbl = getattr(store, name)
        rows = tbl.count_rows()
        cols = [f.name for f in tbl.schema]
        print(f"\n[{name}] {rows} 行")
        print("  字段:", ", ".join(cols))

    rows = store.chunks.search().select(["note_title", "id"]).limit(100000).to_list()
    per_note = Counter(r["note_title"] for r in rows)
    print(f"\n共 {len(per_note)} 篇笔记，块数最多的 5 篇：")
    for title, n in per_note.most_common(5):
        print(f"  {n:5d}  {title}")


def cmd_find(store: VectorStore, keyword: str, limit: int = 10) -> None:
    cond = f"text LIKE '%{_esc(keyword)}%'"
    rows = (
        store.chunks.search()
        .where(cond)
        .select(["id", "note_title", "heading_path", "body", "parent_id"])
        .limit(limit)
        .to_list()
    )
    print(f"关键词「{keyword}」命中 {len(rows)} 条（最多显示 {limit} 条）\n")
    for r in rows:
        preview = r["body"].replace("\n", " / ")[:90]
        print(f"- {r['id']}")
        print(f"  标题: {r['note_title']} | 章节: {r['heading_path']}")
        print(f"  内容: {preview}...")


def cmd_show(store: VectorStore, chunk_id: str) -> None:
    rows = store.chunks.search().where(f"id = '{_esc(chunk_id)}'").limit(1).to_list()
    if not rows:
        print(f"没找到 id = {chunk_id}")
        return
    row = rows[0]
    print("=" * 70)
    for k, v in row.items():
        if k in SKIP_COLS:
            print(f"{k:12s}= <{len(v)} 个浮点数，已省略>")
        else:
            print(f"{k:12s}= {v}")
    parent_id = row.get("parent_id")
    if parent_id:
        parents = store.parents.search().where(f"id = '{_esc(parent_id)}'").limit(1).to_list()
        if parents:
            print("=" * 70)
            print(f"所属父块 {parent_id}（送给大模型的就是这段）：\n")
            print(parents[0]["text"][:1200])


def cmd_export(store: VectorStore) -> None:
    out = Path(".") / "lancedb_export"
    out.mkdir(exist_ok=True)
    for name in ("chunks", "parents"):
        cols = [f.name for f in getattr(store, name).schema if f.name not in SKIP_COLS]
        rows = getattr(store, name).search().select(cols).limit(1000000).to_list()
        path = out / f"{name}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in cols})
        print(f"已导出 {len(rows):5d} 行 -> {path.resolve()}")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        return
    cfg = Config.load(None)
    store = VectorStore(cfg.index_dir)
    action = sys.argv[1]
    if action == "info":
        cmd_info(store)
    elif action == "find":
        cmd_find(store, sys.argv[2])
    elif action == "show":
        cmd_show(store, sys.argv[2])
    elif action == "export":
        cmd_export(store)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
