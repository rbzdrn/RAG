"""标题结构感知的切块 + 父块回填（Small-to-Big）。

设计要点
1. 按 Markdown 标题层级切章节，每个章节是一个「父块」；
2. 章节正文再按「原子块」（段落/代码/表格/callout/列表）贪心打包成「子块」；
3. 子块正文前拼接面包屑（笔记标题 > 章节 > 小节），提升 embedding 的语义定位能力；
4. 检索命中子块后回填父块，兼顾召回精度与上下文完整度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import Note, Section, build_sections, iter_sections

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
TABLE_RE = re.compile(r"^\s*\|")
QUOTE_RE = re.compile(r"^\s*>")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
SENTENCE_RE = re.compile(r"(?<=[。！？!?；;])\s*")


@dataclass
class Chunk:
    id: str
    text: str            # 面包屑 + 正文，用于向量化与重排
    body: str            # 原始正文，用于展示
    note_id: str
    note_title: str
    heading_path: str
    tags_text: str
    parent_id: str
    mtime: float


@dataclass
class Parent:
    id: str
    text: str
    note_id: str
    note_title: str
    heading_path: str
    tags_text: str


@dataclass
class ChunkResult:
    parents: list[Parent] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)


def iter_blocks(lines: list[str]):
    """把章节正文切成不可再分的原子块，代码块 / 表格 / 引用 / 列表保持完整。"""
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            buf = [line]
            i += 1
            while i < n:
                buf.append(lines[i])
                if lines[i].strip().startswith(marker):
                    i += 1
                    break
                i += 1
            yield "\n".join(buf)
            continue

        if TABLE_RE.match(line):
            buf = []
            while i < n and lines[i].strip() and TABLE_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            yield "\n".join(buf)
            continue

        if QUOTE_RE.match(line):
            buf = []
            while i < n and lines[i].strip() and QUOTE_RE.match(lines[i]):
                buf.append(lines[i])
                i += 1
            yield "\n".join(buf)
            continue

        if LIST_RE.match(line):
            buf = []
            while i < n:
                cur = lines[i]
                if LIST_RE.match(cur):
                    buf.append(cur)
                elif cur.startswith(("  ", "\t")) and cur.strip():
                    buf.append(cur)
                else:
                    break
                i += 1
            yield "\n".join(buf)
            continue

        buf = []
        while i < n:
            cur = lines[i]
            if not cur.strip():
                break
            if FENCE_RE.match(cur) or TABLE_RE.match(cur) or QUOTE_RE.match(cur) or LIST_RE.match(cur):
                break
            buf.append(cur)
            i += 1
        if buf:
            yield "\n".join(buf)


def split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """超长原子块按句子边界拆分，必要时硬切。"""
    sentences = [s for s in SENTENCE_RE.split(text) if s.strip()]
    pieces: list[str] = []
    cur = ""

    for sentence in sentences:
        if len(sentence) > max_chars:
            if cur:
                pieces.append(cur)
                cur = ""
            step = max(1, max_chars - overlap)
            for start in range(0, len(sentence), step):
                pieces.append(sentence[start:start + max_chars])
            continue

        if cur and len(cur) + len(sentence) > max_chars:
            pieces.append(cur)
            cur = (cur[-overlap:] if overlap > 0 else "") + sentence
        else:
            cur += sentence

    if cur.strip():
        pieces.append(cur)
    return [p.strip() for p in pieces if p.strip()]


def pack_blocks(blocks: list[str], target: int, maximum: int, minimum: int, overlap: int) -> list[str]:
    units: list[str] = []
    for block in blocks:
        block = block.strip("\n")
        if not block.strip():
            continue
        if len(block) <= maximum:
            units.append(block)
        else:
            units.extend(split_long_text(block, maximum, overlap))

    packed: list[str] = []
    cur = ""
    for unit in units:
        if not cur:
            cur = unit
        elif len(cur) + 2 + len(unit) <= target or len(cur) < minimum:
            cur = f"{cur}\n\n{unit}"
        else:
            packed.append(cur)
            cur = unit
    if cur:
        packed.append(cur)

    # 合并过短的块，避免碎片化
    merged: list[str] = []
    for piece in packed:
        if merged and len(piece) < minimum and len(merged[-1]) + len(piece) + 2 <= maximum:
            merged[-1] = f"{merged[-1]}\n\n{piece}"
        else:
            merged.append(piece)
    return merged


def _breadcrumb(path: list[str], enabled: bool) -> str:
    if not enabled or not path:
        return ""
    parts: list[str] = []
    for item in path:
        item = item.strip()
        if item and (not parts or parts[-1] != item):
            parts.append(item)
    return " > ".join(parts)


def chunk_note(note: Note, cfg) -> ChunkResult:
    result = ChunkResult()
    target = int(cfg.get("chunk.target_chars", 420))
    maximum = int(cfg.get("chunk.max_chars", 900))
    minimum = int(cfg.get("chunk.min_chars", 120))
    overlap = int(cfg.get("chunk.overlap_chars", 80))
    parent_max = int(cfg.get("chunk.parent_max_chars", 2000))
    use_crumb = bool(cfg.get("chunk.breadcrumb_prefix", True))
    tags_text = " " + " ".join(note.tags) + " " if note.tags else " "

    root = build_sections(note.body, note.title)

    for sec_idx, section in enumerate(iter_sections(root)):
        raw_body = "\n".join(section.body).strip()
        if not raw_body:
            continue

        heading_path = " > ".join(section.path)
        prefix = _breadcrumb(section.path, use_crumb)
        parent_id = f"{note.note_id}#s{sec_idx}"

        parent_text = f"{prefix}\n\n{raw_body}" if prefix else raw_body
        if len(parent_text) > parent_max:
            parent_text = parent_text[:parent_max].rstrip() + " …"

        result.parents.append(Parent(
            id=parent_id,
            text=parent_text,
            note_id=note.note_id,
            note_title=note.title,
            heading_path=heading_path,
            tags_text=tags_text,
        ))

        blocks = list(iter_blocks(section.body))
        for chunk_idx, body in enumerate(pack_blocks(blocks, target, maximum, minimum, overlap)):
            text = f"{prefix}\n\n{body}" if prefix else body
            result.chunks.append(Chunk(
                id=f"{note.note_id}#s{sec_idx}c{chunk_idx}",
                text=text,
                body=body,
                note_id=note.note_id,
                note_title=note.title,
                heading_path=heading_path,
                tags_text=tags_text,
                parent_id=parent_id,
                mtime=note.mtime,
            ))

    return result
