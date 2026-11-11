"""构建 / 增量更新 Obsidian 知识库索引。

用法:
    python build_index.py --vault "D:/Obsidian/MyVault"
    python build_index.py --full          # 全量重建
    python build_index.py                 # 使用 config.yaml 中的 vault_path，仅更新变更笔记
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from rag.chunker import chunk_note
from rag.config import Config
from rag.embedder import Embedder
from rag.parser import scan_vault
from rag.store import VectorStore, load_bm25


def main() -> None:
    parser = argparse.ArgumentParser(description="为 Obsidian 仓库构建 RAG 索引")
    parser.add_argument("--config", default=None, help="配置文件路径")
    parser.add_argument("--vault", default=None, help="覆盖 config.yaml 的 vault_path")
    parser.add_argument("--full", action="store_true", help="忽略增量清单，全量重建")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 篇笔记（调试用）")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    vault = Path(args.vault).expanduser().resolve() if args.vault else cfg.vault_path
    if not vault.exists():
        sys.exit(f"仓库不存在: {vault}")

    print(f"扫描仓库: {vault}")
    notes = scan_vault(vault, cfg.get("exclude_globs", []))
    if args.limit:
        notes = notes[: args.limit]
    if not notes:
        sys.exit("没有找到任何 Markdown 笔记")

    embedder = Embedder(
        model_name=cfg.get("embed.model", "BAAI/bge-m3"),
        device=cfg.get("embed.device", "auto"),
        batch_size=int(cfg.get("embed.batch_size", 16)),
        max_seq_length=int(cfg.get("embed.max_seq_length", 512)),
        query_prefix=cfg.get("embed.query_prefix", ""),
    )
    print(f"嵌入模型: {embedder.model_name} (dim={embedder.dim}, device={embedder.device})")

    store = VectorStore(cfg.index_dir)
    if args.full:
        store.reset()
    store.ensure(embedder.dim, embedder.model_name)

    manifest = {} if args.full else store.load_manifest()
    current_ids = {note.note_id for note in notes}

    deleted = [nid for nid in manifest if nid not in current_ids]
    for nid in deleted:
        store.delete_note(nid)
    if deleted:
        print(f"清理已删除/排除笔记 {len(deleted)} 篇")

    pending = [
        note for note in notes
        if manifest.get(note.note_id, {}).get("hash") != note.content_hash
    ]
    print(f"共 {len(notes)} 篇笔记，待更新 {len(pending)} 篇")

    new_manifest = {nid: meta for nid, meta in manifest.items() if nid in current_ids}
    started = time.time()
    added_chunks = 0

    for idx, note in enumerate(pending, 1):
        result = chunk_note(note, cfg)
        if not result.chunks:
            store.delete_note(note.note_id)
            new_manifest[note.note_id] = {"hash": note.content_hash, "chunks": 0, "mtime": note.mtime}
            print(f"  [{idx}/{len(pending)}] {note.note_id} -> 空笔记，跳过")
            continue

        vectors = embedder.encode_documents([chunk.text for chunk in result.chunks])
        store.upsert_note(note.note_id, result.parents, result.chunks, vectors)
        added_chunks += len(result.chunks)
        new_manifest[note.note_id] = {
            "hash": note.content_hash,
            "chunks": len(result.chunks),
            "mtime": note.mtime,
        }
        print(f"  [{idx}/{len(pending)}] {note.note_id} -> {len(result.chunks)} 块")

    store.save_manifest(new_manifest)

    print("同步 BM25 索引 ...")
    load_bm25(cfg, store)

    elapsed = time.time() - started
    print(
        f"完成：新增/更新 {len(pending)} 篇，写入 {added_chunks} 块，"
        f"当前共 {store.count_rows()} 块，耗时 {elapsed:.1f}s"
    )
    print("启动问答页: streamlit run app.py")


if __name__ == "__main__":
    main()
