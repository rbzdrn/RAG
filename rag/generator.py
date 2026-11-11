"""上下文拼装与云端大模型生成（OpenAI 兼容接口）。"""

from __future__ import annotations

import os
import re

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


_CITE_RE = re.compile(r"\[(\d{1,2})\]")


def validate_citations(answer: str, used_count: int) -> list[int]:
    """第二道闸：找出答案里越界的引用编号（不在 1..used_count 范围内）。"""
    refs = {int(m) for m in _CITE_RE.findall(answer or "")}
    return sorted(n for n in refs if not 1 <= n <= used_count)


class Generator:
    def __init__(self, cfg):
        from openai import OpenAI

        api_key_env = cfg.get("llm.api_key_env", "DEEPSEEK_API_KEY")
        api_key = os.environ.get(api_key_env) or cfg.get("llm.api_key", "")
        if not api_key:
            raise RuntimeError(f"未设置环境变量 {api_key_env}，且在配置中找不到 llm.api_key")
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

    def can_answer(self, question: str, hits: list[Hit]) -> bool:
        """第二道闸的前置检查：先问一句「这些片段能回答吗」，答不了就不生成。

        分数阈值挡不住「库里没有」的问题（无关问题也能拿到 0.55），
        所以这里用一次极短调用做硬判定。
        """
        context, _ = build_context(hits, 3000)
        if not context.strip():
            return False
        messages = [
            {
                "role": "system",
                "content": (
                    "你是检索质量检查员。判断给出的笔记片段是否包含回答该问题所需的具体信息。"
                    "只有当片段直接说明了问题所问的内容（事实、定义、步骤、数据等）时才回答 YES；"
                    "仅仅出现相同字眼、在举例中顺带提到、或只能回答相关话题，都必须回答 NO。"
                    "只回答 YES 或 NO，不要输出其他文字。"
                ),
            },
            {"role": "user", "content": f"【片段】\n{context}\n\n【问题】\n{question}"},
        ]
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0, max_tokens=4
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        return text.startswith("YES")

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
