"""向量库（LanceDB）+ 中文 BM25 索引。"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Callable

import numpy as np
import pyarrow as pa

from .chunker import Chunk, Parent


def _chunk_schema(dim: int) -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
        pa.field("text", pa.string()),
        pa.field("body", pa.string()),
        pa.field("note_id", pa.string()),
        pa.field("note_title", pa.string()),
        pa.field("heading_path", pa.string()),
        pa.field("tags_text", pa.string()),
        pa.field("parent_id", pa.string()),
        pa.field("mtime", pa.float64()),
    ])


def _parent_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("note_id", pa.string()),
        pa.field("note_title", pa.string()),
        pa.field("heading_path", pa.string()),
        pa.field("tags_text", pa.string()),
    ])


class VectorStore:
    CHUNKS = "chunks"
    PARENTS = "parents"

    def __init__(self, index_dir: str | Path):
        import lancedb

        self.dir = Path(index_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(self.dir))
        self.dim: int | None = None
        self._parents_cache: dict[str, str] | None = None

    # ---------- 建库 ----------
    def _table_names(self) -> set[str]:
        try:
            return set(self.db.table_names())
        except TypeError:
            return set(self.db.table_names)
        except Exception:
            return set()

    def ensure(self, dim: int, model_name: str) -> None:
        meta_path = self.dir / "meta.json"
        meta: dict = {}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        old_dim = meta.get("dim")
        if old_dim and int(old_dim) != int(dim):
            raise RuntimeError(
                f"索引维度({old_dim})与当前嵌入模型({model_name}, dim={dim})不一致，"
                f"请用 --full 重建索引"
            )
        old_model = meta.get("embedding_model")
        if old_model and old_model != model_name:
            print(f"[warn] 嵌入模型从 {old_model} 变为 {model_name}，建议 --full 重建")
        meta.update({"dim": dim, "embedding_model": model_name})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        self.dim = dim
        names = self._table_names()
        if self.CHUNKS not in names:
            self.db.create_table(self.CHUNKS, schema=_chunk_schema(dim))
        if self.PARENTS not in names:
            self.db.create_table(self.PARENTS, schema=_parent_schema())

    def reset(self) -> None:
        for name in (self.CHUNKS, self.PARENTS):
            try:
                self.db.drop_table(name)
            except Exception:
                pass

    @property
    def chunks(self):
        return self.db.open_table(self.CHUNKS)

    @property
    def parents(self):
        return self.db.open_table(self.PARENTS)

    def count_rows(self) -> int:
        try:
            return int(self.chunks.count_rows())
        except Exception:
            return 0

    # ---------- 写入 ----------
    def delete_note(self, note_id: str) -> None:
        esc = note_id.replace("'", "''")
        for name in (self.CHUNKS, self.PARENTS):
            try:
                self.db.open_table(name).delete(f"note_id = '{esc}'")
            except Exception:
                pass
        self._parents_cache = None

    def upsert_note(self, note_id: str, parents: list[Parent], chunks: list[Chunk],
                    vectors: np.ndarray) -> None:
        self.delete_note(note_id)
        if chunks:
            records = [
                {
                    "id": c.id,
                    "vector": [float(x) for x in vec],
                    "text": c.text,
                    "body": c.body,
                    "note_id": c.note_id,
                    "note_title": c.note_title,
                    "heading_path": c.heading_path,
                    "tags_text": c.tags_text,
                    "parent_id": c.parent_id,
                    "mtime": c.mtime,
                }
                for c, vec in zip(chunks, vectors)
            ]
            self.chunks.add(pa.Table.from_pylist(records, schema=_chunk_schema(self.dim or len(vectors[0]))))
        if parents:
            records = [
                {
                    "id": p.id,
                    "text": p.text,
                    "note_id": p.note_id,
                    "note_title": p.note_title,
                    "heading_path": p.heading_path,
                    "tags_text": p.tags_text,
                }
                for p in parents
            ]
            self.parents.add(pa.Table.from_pylist(records, schema=_parent_schema()))

    # ---------- 查询 ----------
    def search_dense(self, vector: np.ndarray, k: int, where: str | None = None) -> list[dict]:
        query = self.chunks.search([float(x) for x in vector]).limit(k)
        if where:
            query = query.where(where, prefilter=True)
        rows = query.to_list()
        out = []
        for row in rows:
            row.pop("vector", None)
            distance = row.pop("_distance", None)
            row["_dense_score"] = float(1.0 - distance) if distance is not None else 0.0
            out.append(row)
        return out

    def get_parents(self, ids: list[str]) -> dict[str, str]:
        if not ids:
            return {}
        if self._parents_cache is None:
            arrow = self.parents.to_arrow()
            self._parents_cache = dict(zip(
                arrow.column("id").to_pylist(),
                arrow.column("text").to_pylist(),
            ))
        return {i: self._parents_cache[i] for i in ids if i in self._parents_cache}

    def chunk_records(self) -> list[dict]:
        """导出不含向量的紧凑记录，用于构建 BM25。"""
        arrow = self.chunks.to_arrow().drop_columns(["vector"])
        keep = [
            "id", "text", "body", "note_id", "note_title", "heading_path", "tags_text", "parent_id",
        ]
        return [{k: row[k] for k in keep} for row in arrow.to_pylist()]

    # ---------- 清单 ----------
    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    def load_manifest(self) -> dict:
        if self.manifest_path.exists():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        return {}

    def save_manifest(self, manifest: dict) -> None:
        self.manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------- BM25 ----------------

def tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in jieba.lcut_for_search((text or "").lower()) if t.strip()]


class BM25Index:
    def __init__(self, records: list[dict]):
        from rank_bm25 import BM25Okapi

        self.records = records
        self.tokens = [tokenize(r.get("text", "")) for r in records]
        self.bm25 = BM25Okapi(self.tokens) if self.tokens else None

    def search(self, query: str, k: int = 20,
               predicate: Callable[[dict], bool] | None = None) -> list[dict]:
        if not self.bm25:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        order = np.argsort(scores)[::-1]
        hits: list[dict] = []
        for idx in order:
            score = float(scores[int(idx)])
            if score <= 0:
                break
            record = self.records[int(idx)]
            if predicate and not predicate(record):
                continue
            hits.append({**record, "_bm25_score": score})
            if len(hits) >= k:
                break
        return hits

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({"records": self.records, "tokens": self.tokens}, f, protocol=4)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.records = data["records"]
        obj.tokens = data["tokens"]
        obj.bm25 = BM25Okapi(obj.tokens) if obj.tokens else None
        return obj

    @classmethod
    def load_or_build(cls, path: str | Path, records: list[dict]) -> "BM25Index":
        p = Path(path)
        if p.exists():
            try:
                return cls.load(p)
            except Exception:
                pass
        return cls(records)


def load_bm25(cfg, store: VectorStore) -> BM25Index:
    """按索引块数判断 BM25 是否过期，过期则重建并落盘。"""
    path = Path(cfg.index_dir) / "bm25.pkl"
    count_path = Path(cfg.index_dir) / "bm25.count"
    count = store.count_rows()

    if path.exists() and count_path.exists():
        try:
            if int(count_path.read_text(encoding="utf-8").strip()) == count:
                return BM25Index.load(path)
        except Exception:
            pass

    index = BM25Index(store.chunk_records())
    index.save(path)
    count_path.write_text(str(count), encoding="utf-8")
    return index
