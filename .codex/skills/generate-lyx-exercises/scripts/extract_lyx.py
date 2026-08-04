#!/usr/bin/env python3
"""Extract readable body text or an outline from a LyX source file."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


HEADING_LEVELS = {
    "Part": 1,
    "Part*": 1,
    "Chapter": 1,
    "Chapter*": 1,
    "Section": 1,
    "Section*": 1,
    "Subsection": 2,
    "Subsection*": 2,
    "Subsubsection": 3,
    "Subsubsection*": 3,
    "Paragraph": 4,
    "Paragraph*": 4,
    "Minisec": 4,
}
LIST_LAYOUTS = {"Itemize": "-", "Enumerate": "1."}
SKIP_DIRECTIVES = (
    "\\emph ",
    "\\family ",
    "\\series ",
    "\\shape ",
    "\\color ",
    "\\lang ",
    "\\noun ",
    "\\bar ",
    "\\strikeout ",
    "\\uuline ",
    "\\uwave ",
)


def consume_inset(lines: list[str], start: int) -> tuple[str, list[str], int]:
    header = lines[start].strip()
    match = re.match(r"\\begin_inset\s+(\S+)(?:\s+(.*))?$", header)
    kind = match.group(1) if match else "Unknown"
    first = match.group(2) if match and match.group(2) else ""
    content = [first] if first else []
    depth = 1
    index = start + 1
    while index < len(lines) and depth:
        stripped = lines[index].strip()
        if stripped.startswith("\\begin_inset"):
            depth += 1
        elif stripped == "\\end_inset":
            depth -= 1
            if depth == 0:
                index += 1
                break
        if depth:
            content.append(lines[index])
        index += 1
    return kind, content, index


def clean_inline(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([({])\s+", r"\1", text)
    return text


def formula_text(content: list[str]) -> str:
    kept = [line.rstrip() for line in content if line.strip()]
    return "\n".join(kept).strip()


def render_layout(layout: str, content: list[str], include_notes: bool) -> str:
    tokens: list[str] = []
    index = 0
    while index < len(content):
        stripped = content[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("\\begin_inset"):
            kind, inset_content, index = consume_inset(content, index)
            if kind == "Formula":
                formula = formula_text(inset_content)
                if formula:
                    tokens.append(formula)
            elif kind == "Note" and include_notes:
                tokens.append(extract_lines(inset_content, include_notes))
            continue
        if stripped.startswith(SKIP_DIRECTIVES) or stripped.startswith("\\end_"):
            index += 1
            continue
        if stripped.startswith("\\"):
            index += 1
            continue
        tokens.append(stripped)
        index += 1

    if not tokens:
        return ""

    blocks: list[str] = []
    inline: list[str] = []
    for token in tokens:
        if "\n" in token or token.startswith("\\["):
            if inline:
                blocks.append(clean_inline(" ".join(inline)))
                inline = []
            blocks.append(token)
        else:
            inline.append(token)
    if inline:
        blocks.append(clean_inline(" ".join(inline)))
    text = "\n\n".join(block for block in blocks if block)

    if layout in HEADING_LEVELS:
        return f"{'#' * HEADING_LEVELS[layout]} {clean_inline(text)}"
    if layout in LIST_LAYOUTS:
        return f"{LIST_LAYOUTS[layout]} {text}"
    if layout not in {"Standard", "Plain Layout"}:
        return f"**{layout}.** {text}"
    return text


def extract_lines(lines: list[str], include_notes: bool = False) -> str:
    output: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("\\begin_inset"):
            kind, inset_content, index = consume_inset(lines, index)
            if kind == "Note" and include_notes:
                note = extract_lines(inset_content, include_notes)
                if note:
                    output.append(note)
            continue
        if not stripped.startswith("\\begin_layout "):
            index += 1
            continue
        layout = stripped.removeprefix("\\begin_layout ")
        start = index
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index].strip() != "\\end_layout":
            content.append(lines[index])
            index += 1
        rendered = render_layout(layout, content, include_notes)
        if rendered:
            output.append(rendered)
        index += 1
        if index <= start:
            raise RuntimeError("LyX parser did not advance")
    return "\n\n".join(output)


def body_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("#LyX"):
        raise ValueError(f"{path} does not look like a LyX source file")
    lines = text.splitlines()
    try:
        start = lines.index("\\begin_body") + 1
        end = lines.index("\\end_body", start)
    except ValueError as error:
        raise ValueError(f"{path} has no complete LyX body") from error
    return lines[start:end]


def outline(lines: list[str]) -> str:
    items: list[str] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped.startswith("\\begin_layout "):
            index += 1
            continue
        layout = stripped.removeprefix("\\begin_layout ")
        index += 1
        content: list[str] = []
        while index < len(lines) and lines[index].strip() != "\\end_layout":
            content.append(lines[index])
            index += 1
        if layout in HEADING_LEVELS:
            title = clean_inline(render_layout("Standard", content, False))
            items.append(f"{'  ' * (HEADING_LEVELS[layout] - 1)}- {layout}: {title}")
        index += 1
    return "\n".join(items)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="LyX source file")
    parser.add_argument("--outline", action="store_true", help="print headings only")
    parser.add_argument(
        "--include-notes",
        action="store_true",
        help="include Note insets in full-text output",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        lines = body_lines(args.source)
        result = outline(lines) if args.outline else extract_lines(lines, args.include_notes)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
