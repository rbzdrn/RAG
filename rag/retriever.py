"""混合检索：向量 + BM25 -> RRF 融合 -> 交叉编码器重排 -> 父块回填。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .embedder import Embedder, resolve_device
from .store import BM25Index, VectorStore


@dataclass
class Hit:
    id: str
    text: str
    body: str
    note_id: str
    note_title: str
    heading_path: str
    parent_id: str
    score: float = 0.0
    dense_score: float = 0.0
    parent_text: str = ""
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float = 0.0


class Reranker:
    def __init__(self, model_name: str, device: str = "auto"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=resolve_device(device))

    def rerank(self, query: str, docs: list[str], top_k: int) -> list[tuple[int, float]]:
        if not docs:
            return []
        scores = self.model.predict(
            [(query, d) for d in docs], batch_size=8, show_progress_bar=False
        )
        order = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[int(i)])) for i in order]


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, bm25: BM25Index, cfg):
        self.store = store
        self.embedder = embedder
        self.bm25 = bm25
        self.cfg = cfg
        self.dense_k = int(cfg.get("retrieval.dense_k", 20))
        self.bm25_k = int(cfg.get("retrieval.bm25_k", 20))
        self.rrf_k = int(cfg.get("retrieval.rrf_k", 60))
        self.final_k = int(cfg.get("retrieval.final_k", 5))
        self.rerank_enabled = bool(cfg.get("rerank.enabled", True))
        self.rerank_top_n = int(cfg.get("rerank.top_n", 20))
        # 第一道闸：砍掉分数垫底的块。绝对阈值不可靠（实测无关问题也能到 0.55），
        # 所以只保留与最高分差距在 dense_margin 以内的
        self.min_dense_score = float(cfg.get("retrieval.min_dense_score", 0.30))
        self.dense_margin = float(cfg.get("retrieval.dense_margin", 0.10))
        self._reranker: Reranker | None = None

    @property
    def reranker(self) -> Reranker | None:
        if not self.rerank_enabled:
            return None
        if self._reranker is None:
            self._reranker = Reranker(
                self.cfg.get("rerank.model", "BAAI/bge-reranker-base"),
                self.cfg.get("embed.device", "auto"),
            )
        return self._reranker

    def search(self, query: str, final_k: int | None = None, tag: str | None = None) -> list[Hit]:
        final_k = final_k or self.final_k
        where = None
        predicate = None
        if tag:
            normalized = tag.strip().lstrip("#").lower()
            where = f"tags_text LIKE '% {normalized.replace(chr(39), chr(39) * 2)} %'"

            def predicate(record: dict, _t=normalized) -> bool:
                return f" {_t} " in (record.get("tags_text") or "")

        dense_rows = self.store.search_dense(self.embedder.encode_query(query), self.dense_k, where)
        bm25_rows = self.bm25.search(query, self.bm25_k, predicate)

        merged: dict[str, Hit] = {}
        for rank, row in enumerate(dense_rows):
            hit = merged.get(row["id"])
            if hit is None:
                hit = self._to_hit(row)
                merged[row["id"]] = hit
            hit.dense_rank = rank + 1
            hit.rrf_score += 1.0 / (self.rrf_k + rank + 1)

        for rank, row in enumerate(bm25_rows):
            hit = merged.get(row["id"])
            if hit is None:
                hit = self._to_hit(row)
                merged[row["id"]] = hit
            hit.bm25_rank = rank + 1
            hit.rrf_score += 1.0 / (self.rrf_k + rank + 1)

        candidates = self._prune(
            sorted(merged.values(), key=lambda h: h.rrf_score, reverse=True)
        )
        pool = candidates[: max(self.rerank_top_n, final_k)]

        reranker = self.reranker
        if reranker and pool:
            ranked = reranker.rerank(query, [h.text for h in pool], min(len(pool), final_k * 3))
            ordered: list[Hit] = []
            for idx, score in ranked:
                hit = pool[idx]
                hit.score = score
                ordered.append(hit)
        else:
            ordered = list(pool)
            for hit in ordered:
                hit.score = hit.rrf_score

        # 同一父章节只保留最相关的一个子块，避免上下文重复；
        # 被丢弃的同章节子块正文记下来，稍后补进上下文（超长章节回填会截断）
        results: list[Hit] = []
        seen_parents: set[str] = set()
        dropped: dict[str, list[str]] = {}
        for hit in ordered:
            key = hit.parent_id or hit.id
            if key in seen_parents:
                dropped.setdefault(key, []).append(hit.body or hit.text)
                continue
            seen_parents.add(key)
            results.append(hit)
            if len(results) >= final_k:
                break

        parent_map = self.store.get_parents([h.parent_id for h in results])
        for hit in results:
            key = hit.parent_id or hit.id
            parts = [parent_map.get(hit.parent_id) or hit.text]
            for body in [hit.body or hit.text, *dropped.get(key, [])]:
                if body and body[:30] not in parts[0]:
                    parts.append(f"[命中片段]\n{body}")
            hit.parent_text = "\n\n".join(p for p in parts if p)
        return results

    def _prune(self, hits: list[Hit]) -> list[Hit]:
        """第一道闸：只保留和最高分接近的块；纯关键词命中的块不受此限。"""
        if not hits:
            return []
        best = max(h.dense_score for h in hits)
        floor = max(self.min_dense_score, best - self.dense_margin)
        return [h for h in hits if h.dense_rank is None or h.dense_score >= floor]

    @staticmethod
    def _to_hit(row: dict) -> Hit:
        return Hit(
            id=row.get("id", ""),
            text=row.get("text", ""),
            body=row.get("body", row.get("text", "")),
            note_id=row.get("note_id", ""),
            note_title=row.get("note_title", ""),
            heading_path=row.get("heading_path", ""),
            parent_id=row.get("parent_id", ""),
            dense_score=float(row.get("_dense_score", 0.0)),
        )
