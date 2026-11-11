"""本地 Web 问答页：streamlit run app.py"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st

from rag.config import ROOT, Config
from rag import history
from rag.embedder import Embedder
from rag.generator import Generator, validate_citations
from rag.retriever import Retriever
from rag.store import VectorStore, load_bm25

st.set_page_config(page_title="Obsidian 知识库问答", page_icon=None, layout="wide")


@st.cache_resource(show_spinner="正在加载嵌入模型（首次约 30 秒）…")
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


@st.cache_data(show_spinner=False)
def load_tags(index_dir: str, row_count: int) -> list[str]:
    """库里出现过的标签（row_count 只用来让缓存随重建索引失效）。"""
    return VectorStore(index_dir).list_tags()


def main() -> None:
    config_path = os.environ.get("RAG_CONFIG", str(ROOT / "config.yaml"))

    with st.sidebar:
        records = history.load(50)
        st.header("设置")
        try:
            cfg, store, retriever = load_engine(config_path)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        st.caption(f"索引目录：{store.dir}")
        st.metric("已索引块数", store.count_rows())
        st.caption(f"嵌入模型：{cfg.get('embed.model')}")

        all_tags = load_tags(str(store.dir), store.count_rows())
        if all_tags:
            picked_tag = st.selectbox("只看某个标签（可选）", ["（不过滤）", *all_tags])
            tag = "" if picked_tag == "（不过滤）" else picked_tag
        else:
            st.caption("库里还没有标签，可在笔记 frontmatter 写 tags: [xxx]")
            tag = ""
        top_k = st.slider("引用片段数", 1, 10, int(cfg.get("retrieval.final_k", 5)))

        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()

        st.divider()
        st.subheader(f"历史记录（{len(records)}）")
        if records:
            for i, rec in enumerate(records[:30]):
                label = f"{rec['time'][5:16]}  {rec['question'][:13]}"
                if st.button(label, key=f"hist_{i}", use_container_width=True):
                    st.session_state.view = "history"
                    st.session_state.hist_index = i
                    st.rerun()
            if len(records) > 30:
                st.caption(f"仅显示最近 30 条，共 {len(records)} 条")
            if st.button("清空历史", use_container_width=True):
                history.clear()
                st.rerun()
        else:
            st.caption("还没有记录")

    st.title("Obsidian 知识库问答")

    if st.session_state.get("view") == "history" and records:
        rec = records[min(st.session_state.get("hist_index", 0), len(records) - 1)]
        if st.button("← 返回对话"):
            st.session_state.view = "chat"
            st.rerun()
        st.caption(f"提问时间：{rec['time']}")
        st.subheader(rec["question"])
        st.markdown(rec["answer"])
        elapsed = rec.get("elapsed") or {}
        if elapsed:
            st.caption("耗时：" + " · ".join(f"{k} {v}s" for k, v in elapsed.items()))
        sources = rec.get("sources") or []
        if sources:
            with st.expander(f"引用来源（{len(sources)}）", expanded=False):
                for src in sources:
                    st.caption(src)
        return

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
        t0 = time.perf_counter()
        with st.spinner("正在检索笔记…"):
            hits = retriever.search(question, final_k=top_k, tag=tag.strip() or None)
        t_search = time.perf_counter() - t0
        if not hits:
            if tag:
                msg = f"标签「{tag}」下没有检索到相关片段。换个标签或选「（不过滤）」再试。"
            else:
                msg = "笔记里没有检索到相关片段，这次没有调用大模型。"
            st.warning(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
            history.append(question, msg, [], {"search": round(t_search, 1)})
            return

        answer = ""
        t_verify = 0.0
        t_gen0 = time.perf_counter()
        try:
            generator = Generator(cfg)
            best = hits[0].dense_score
            # 向量分特别高时直接跳过预检，省掉一次往返
            need_verify = (
                bool(cfg.get("retrieval.verify_with_llm", True))
                and best < float(cfg.get("retrieval.skip_verify_score", 0.65))
            )
            if need_verify:
                with st.spinner("正在判断笔记里的内容是否够用…"):
                    ok = generator.can_answer(question, hits)
                t_verify = time.perf_counter() - t0 - t_search
                if not ok:
                    msg = (
                        "笔记里没有能回答这个问题的内容，所以这次没有生成答案——"
                        "避免模型拿网上或训练时的知识来充数。"
                    )
                    st.warning(msg)
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    history.append(question, msg, [], {"search": round(t_search, 1), "verify": round(t_verify, 1)})
                    return
            t_gen0 = time.perf_counter()
            answer = st.write_stream(generator.stream(question, hits))
        except Exception as exc:
            st.error(f"生成失败：{exc}")
            answer = "（生成失败）"
        t_gen = time.perf_counter() - t_gen0

        sources = [
            f"[{i}] 《{hit.note_title}》 · {hit.heading_path} · score={hit.score:.3f}"
            for i, hit in enumerate(hits, 1)
        ]
        with st.expander(f"引用来源（{len(hits)}）", expanded=False):
            for i, hit in enumerate(hits, 1):
                st.markdown(f"**[{i}] {hit.note_title}** — `{hit.note_id}`")
                st.caption(hit.heading_path)
                st.text((hit.parent_text or hit.text)[:600])

        bad_refs = validate_citations(answer, len(hits))
        if bad_refs:
            st.warning(
                f"答案里出现了不存在的引用 {bad_refs}，本次上下文只有 {len(hits)} 段。"
                "这部分内容可能不是来自你的笔记，建议核对。"
            )

        st.caption(f"耗时：检索 {t_search:.1f}s · 预检 {t_verify:.1f}s · 生成 {t_gen:.1f}s")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer or "",
        "sources": sources,
    })
    history.append(
        question,
        answer or "",
        sources,
        {"search": round(t_search, 1), "verify": round(t_verify, 1), "gen": round(t_gen, 1)},
    )


if __name__ == "__main__":
    main()
