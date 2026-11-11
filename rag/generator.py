"""上下文拼装与云端大模型生成（OpenAI 兼容接口）。"""

from __future__ import annotations

import os

from .retriever import Hit

SYSTEM_PROMPT = (
    "你是我的第二大脑助手，基于我的 Obsidian 笔记回答提问。\n"
    "规则：\n"
    "1. 只使用提供的笔记片段作答，不要编造片段之外的事实。\n"
    "2. 片段不足以回答时，直接说明缺少哪部分信息。\n"
    "3. 引用来源时用 [编号] 标注，例如 [1][2]。\n"
    "4. 用简体中文回答，先给结论，再给必要细节。\n"
    "5. 片段可能包含与问题无关的内容：只引用真正支撑答案的片段，"
    "不要因为片段里出现了相似的字眼就把它写进答案；一个都不相关时直接说没有找到。"
)


def build_context(hits: list[Hit], max_chars: int = 7000) -> tuple[str, list[Hit]]:
    """按父块去重拼装上下文，返回文本与被实际引用的片段。"""
    blocks: list[str] = []
    used: list[Hit] = []
    seen: set[str] = set()
    total = 0

    for hit in hits:
        key = hit.parent_id or hit.id
        if key in seen:
            continue
        seen.add(key)
        idx = len(used) + 1
        body = hit.parent_text or hit.text
        block = f"[{idx}] 《{hit.note_title}》 {hit.heading_path}\n{body}"
        if total + len(block) > max_chars and used:
            break
        blocks.append(block)
        used.append(hit)
        total += len(block)

    return "\n\n".join(blocks), used


class Generator:
    def __init__(self, cfg):
        from openai import OpenAI

        api_key_env = cfg.get("llm.api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"未设置环境变量 {api_key_env}，无法调用大模型")
        self.client = OpenAI(api_key=api_key, base_url=cfg.get("llm.base_url"))
        self.model = cfg.get("llm.model", "deepseek-chat")
        self.temperature = float(cfg.get("llm.temperature", 0.2))
        self.context_max_chars = int(cfg.get("retrieval.context_max_chars", 7000))

    def build_messages(self, question: str, hits: list[Hit]) -> tuple[list[dict], list[Hit]]:
        context, used = build_context(hits, self.context_max_chars)
        if not context:
            context = "（没有检索到相关笔记片段）"
        user = f"【笔记片段】\n{context}\n\n【问题】\n{question}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ], used

    def stream(self, question: str, hits: list[Hit]):
        messages, used = self.build_messages(question, hits)
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def answer(self, question: str, hits: list[Hit]) -> str:
        messages, _ = self.build_messages(question, hits)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""
