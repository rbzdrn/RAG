from __future__ import annotations

import numpy as np


def resolve_device(device: str | None) -> str:
    if device and device != "auto":
        return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Embedder:
    """基于 sentence-transformers 的本地向量化，默认 BAAI/bge-m3。"""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16,
                 max_seq_length: int = 512, query_prefix: str = ""):
        from sentence_transformers import SentenceTransformer

        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.query_prefix = query_prefix or ""
        self.model_name = model_name
        try:
            self.dim = int(self.model.get_embedding_dimension())
        except AttributeError:
            self.dim = int(self.model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                texts,
                batch_size=batch_size or self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(texts) > 64,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vec[0], dtype=np.float32)
