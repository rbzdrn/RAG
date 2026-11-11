"""Obsidian 笔记解析：frontmatter、标签、双链、正文清洗。"""

from __future__ import annotations

import fnmatch
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.S)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
EMBED_RE = re.compile(r"!\[\[[^\]]*\]\]")
WIKILINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:#[^\]\|]*)?(?:\|([^\]]*))?\]\]")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
INLINE_TAG_RE = re.compile(r"(?<![\w/#])#([\w\u4e00-\u9fff][\w\u4e00-\u9fff/\-]*)")
BLOCK_COMMENT_RE = re.compile(r"%%.*?%%", re.S)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)
HIGHLIGHT_RE = re.compile(r"==([^=\n]+)==")
CALLOUT_RE = re.compile(r"^(\s*)>\s*\[!(\w+)\][+-]?\s*(.*)$", re.M)
MULTI_BLANK_RE = re.compile(r"\n{3,}")


@dataclass
class Note:
    note_id: str          # 相对 vault 的 posix 路径
    title: str
    frontmatter: dict
    tags: list[str]
    links: list[str]
    body: str             # 清洗后的 Markdown 正文（不含 frontmatter）
    mtime: float
    content_hash: str


@dataclass
class Section:
    level: int
    title: str
    path: list[str]
    body: list[str] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)


def _split_frontmatter(raw: str) -> tuple[dict, str]:
    m = FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, raw[m.end():]


def _normalize_tags(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        items = re.split(r"[,\s]+", value)
    elif isinstance(value, (list, tuple, set)):
        items = []
        for v in value:
            items.extend(re.split(r"[,\s]+", str(v)))
    else:
        items = [str(value)]
    return [t.strip().lstrip("#").lower() for t in items if t and t.strip()]


def clean_markdown(raw: str) -> tuple[str, list[str], list[str]]:
    """把 Obsidian 语法清洗成适合向量化的纯文本，同时收集标签与双链。"""
    text = BLOCK_COMMENT_RE.sub("", raw)
    text = HTML_COMMENT_RE.sub("", text)

    links: list[str] = []

    def _wikilink(m: re.Match) -> str:
        target = (m.group(1) or "").strip()
        alias = (m.group(2) or "").strip()
        if target:
            links.append(target.lower())
        return alias or target

    # 附件嵌入先移除，避免图片文件名污染语义
    text = EMBED_RE.sub("", text)
    text = WIKILINK_RE.sub(_wikilink, text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = HIGHLIGHT_RE.sub(r"\1", text)
    # callout 标记 > [!note] 标题  ->  > 标题
    text = CALLOUT_RE.sub(lambda m: f"{m.group(1)}> {m.group(3)}".rstrip(), text)

    inline_tags = [t.lower() for t in INLINE_TAG_RE.findall(text)]
    # 保留标签词本身参与向量化，只去掉 # 号；不压缩空格以免破坏代码缩进
    text = INLINE_TAG_RE.sub(r"\1", text)
    text = MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip(), inline_tags, links


def read_note(path: Path, vault_root: Path) -> Note:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    meta, body_raw = _split_frontmatter(raw)
    body, inline_tags, links = clean_markdown(body_raw)

    title = str(meta.get("title") or path.stem).strip()
    fm_tags = _normalize_tags(meta.get("tags") or meta.get("tag"))
    tags = sorted(set(fm_tags + inline_tags))

    rel = path.relative_to(vault_root).as_posix()
    return Note(
        note_id=rel,
        title=title,
        frontmatter=meta,
        tags=tags,
        links=sorted(set(links)),
        body=body,
        mtime=path.stat().st_mtime,
        content_hash=hashlib.sha1(raw.encode("utf-8")).hexdigest(),
    )


def _is_excluded(rel_posix: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if fnmatch.fnmatch(rel_posix, pat):
            return True
        # 兼容 "dir/**" 针对目录本身的匹配
        if pat.endswith("/**") and fnmatch.fnmatch(rel_posix, pat[:-3]):
            return True
    return False


def scan_vault(vault_root: Path, exclude_globs: list[str] | None = None) -> list[Note]:
    patterns = exclude_globs or []
    notes: list[Note] = []
    for path in sorted(vault_root.rglob("*.md")):
        rel = path.relative_to(vault_root).as_posix()
        if any(part.startswith(".") for part in rel.split("/")[:-1]):
            continue
        if _is_excluded(rel, patterns):
            continue
        try:
            notes.append(read_note(path, vault_root))
        except Exception as exc:  # 单篇笔记失败不影响整体
            print(f"[warn] 跳过 {rel}: {exc}")
    return notes


def build_sections(text: str, root_title: str) -> Section:
    """按 Markdown 标题层级构建章节树，代码块内的 # 不会被误判为标题。"""
    root = Section(level=0, title=root_title, path=[root_title])
    stack: list[Section] = [root]
    in_fence = False

    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            stack[-1].body.append(line)
            continue

        if not in_fence:
            m = HEADING_RE.match(line)
            if m:
                level = len(m.group(1))
                title = m.group(2).strip()
                # H1 与笔记标题相同且位于文首时，视为文档标题，不再单独成节
                if (level == 1 and title == root_title and not root.children
                        and not any(t.strip() for t in root.body)):
                    continue
                while len(stack) > 1 and stack[-1].level >= level:
                    stack.pop()
                parent = stack[-1]
                section = Section(level=level, title=title, path=parent.path + [title])
                parent.children.append(section)
                stack.append(section)
                continue

        stack[-1].body.append(line)

    return root


def iter_sections(root: Section):
    yield root
    for child in root.children:
        yield from iter_sections(child)
