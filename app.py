"""本地 Web 问答页：streamlit run app.py"""

from __future__ import annotations

import os
import re
from pathlib import Path

import streamlit as st

from rag.config import ROOT, Config
from rag.embedder import Embedder
from rag.generator import Generator
from rag.retriever import Hit, Retriever
from rag.store import VectorStore, load_bm25


def _strip_breadcrumb(text: str, note_title: str) -> str:
    """父块正文第一行是「笔记标题 > 章节」的面包屑，展示时去掉避免标题重复。"""
    head, sep, rest = (text or "").partition("\n\n")
    if sep and (">" in head or head.strip() == (note_title or "").strip()):
        return rest.strip()
    return (text or "").strip()


def _cited_sources(answer: str, hits: list[Hit]) -> list[tuple[int, Hit]]:
    """答案里 [n] 实际引用了哪几段就只展示哪几段；一个都没引用则全展示。"""
    refs = {int(n) for n in re.findall(r"\[(\d{1,2})\]", answer or "")}
    picked = [(i, h) for i, h in enumerate(hits, 1) if i in refs]
    return picked or list(enumerate(hits, 1))


st.set_page_config(page_title="Obsidian 知识库问答", page_icon=None, layout="wide")


@st.cache_resource(show_spinner=False)
def load_engine(config_path: str):
    cfg = Config.load(config_path)
    embedder = Embedder(
        model_name=cfg.get("embed.model", "BAAI/bge-m3"),
        device=cfg.get("embed.device", "auto"),
        batch_size=int(cfg.get("embed.batch_size", 16)),
        max_seq_length=int(cfg.get("embed.max_seq_length", 512)),
        query_prefix=cfg.get("embed.query_prefix", ""),
    )
    store = VectorStore(cfg.index_dir)
    store.ensure(embedder.dim, embedder.model_name)
    if store.count_rows() == 0:
        raise RuntimeError("索引为空，请先在终端运行：python build_index.py")
    bm25 = load_bm25(cfg, store)
    retriever = Retriever(store, embedder, bm25, cfg)
    return cfg, store, retriever


def main() -> None:
    config_path = os.environ.get("RAG_CONFIG", str(ROOT / "config.yaml"))

    with st.sidebar:
        st.header("设置")
        try:
            cfg, store, retriever = load_engine(config_path)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        st.caption(f"索引目录：{store.dir}")
        st.metric("已索引块数", store.count_rows())
        st.caption(f"嵌入模型：{cfg.get('embed.model')}")

        tag = st.text_input("只看某个标签（可选）", placeholder="例如 算法")
        top_k = st.slider("引用片段数", 1, 10, int(cfg.get("retrieval.final_k", 5)))

        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()

    st.title("Obsidian 知识库问答")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for source in message.get("sources", []):
                st.caption(source)

    question = st.chat_input("向你的笔记提问…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = retriever.search(question, final_k=top_k, tag=tag.strip() or None)
        if not hits:
            st.warning("没有检索到相关笔记。")
            st.session_state.messages.append({"role": "assistant", "content": "没有检索到相关笔记。"})
            return

        answer = ""
        try:
            generator = Generator(cfg)
            answer = st.write_stream(generator.stream(question, hits))
        except Exception as exc:
            st.error(f"生成失败：{exc}")
            answer = "（生成失败）"

        cited = _cited_sources(answer, hits)
        sources = [
            f"[{i}] 《{hit.note_title}》 · {hit.heading_path} · score={hit.score:.3f}"
            for i, hit in cited
        ]
        with st.expander(f"引用来源（{len(cited)}）", expanded=False):
            if len(cited) < len(hits):
                st.caption(f"只显示答案实际引用的 {len(cited)} 段，其余 {len(hits) - len(cited)} 段未被采用。")
            for i, hit in cited:
                st.markdown(f"**[{i}] {hit.note_title}** — `{hit.note_id}`")
                st.caption(hit.heading_path)
                st.text(_strip_breadcrumb(hit.parent_text or hit.text, hit.note_title)[:600])

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer or "",
        "sources": sources,
    })


if __name__ == "__main__":
    main()
