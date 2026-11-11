from __future__ import annotations

import json
import urllib.request

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
    """本地向量化：sentence-transformers 或 Ollama（/api/embed），默认 bge-m3。"""

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16,
                 max_seq_length: int = 512, query_prefix: str = "",
                 backend: str = "sentence-transformers",
                 base_url: str = "http://127.0.0.1:11434",
                 dim: int | None = None):
        self.backend = (backend or "sentence-transformers").lower()
        self.query_prefix = query_prefix or ""
        self.batch_size = batch_size
        self.base_url = base_url.rstrip("/")
        self.device = resolve_device(device)
        if self.backend == "ollama":
            self.model = model_name
            self.model_name = f"ollama:{model_name}"
            self.dim = int(dim or 1024)
            return
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=self.device)
        if max_seq_length:
            self.model.max_seq_length = max_seq_length
        self.model_name = model_name
        try:
            self.dim = int(self.model.get_embedding_dimension())
        except AttributeError:
            self.dim = int(self.model.get_sentence_embedding_dimension())

    def _embed_ollama(self, texts: list[str]) -> np.ndarray:
        """调用本机 Ollama 的 /api/embed，返回 L2 归一化后的向量。"""
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
        vectors = np.asarray(data["embeddings"], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.backend == "ollama":
            size = batch_size or self.batch_size
            batches = [
                self._embed_ollama(texts[i:i + size])
                for i in range(0, len(texts), size)
            ]
            return np.vstack(batches)
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
        if self.backend == "ollama":
            return self._embed_ollama([self.query_prefix + text])[0]
        vec = self.model.encode(
            [self.query_prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vec[0], dtype=np.float32)
