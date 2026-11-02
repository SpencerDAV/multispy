"""Render the project's Markdown docs as pages the app can link to.

A small Markdown subset rather than a dependency: the whole point of the web
layer here is that it needs nothing but the standard library, and the docs only
use headings, code fences, tables, lists, quotes, rules and the usual inline
marks. Anything outside that subset degrades to plain text rather than breaking
the page.

Relative links between the docs (``README.md`` <-> ``TUTORIAL.md``) are rewritten
to the routes the server actually exposes, so the cross-references work in the
browser as well as on disk.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# route name -> file on disk
DOCS = {
    "tutorial": ("TUTORIAL.md", "Tutorial"),
    "readme": ("README.md", "Reference"),
}

# how a relative .md link in the source maps onto a route
LINK_MAP = {
    "readme.md": "/docs/readme",
    "tutorial.md": "/docs/tutorial",
}


def slug(text: str) -> str:
    s = re.sub(r"<[^>]+>", "", text).lower()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    return re.sub(r"[\s-]+", "-", s).strip("-") or "section"


def _rewrite_link(href: str) -> str:
    bare = href.split("#", 1)[0].strip()
    anchor = href[len(bare):] if "#" in href else ""
    mapped = LINK_MAP.get(bare.lower())
    return (mapped + anchor) if mapped else href


def _inline(text: str) -> str:
    """Inline marks. Code spans are stashed first so nothing formats inside them."""
    spans: list[str] = []

    def stash(m: re.Match) -> str:
        spans.append(m.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![*\w])\*([^*]+?)\*(?![*\w])", r"<em>\1</em>", text)

    def link(m: re.Match) -> str:
        label, href = m.group(1), _rewrite_link(m.group(2))
        external = href.startswith(("http://", "https://"))
        extra = ' target="_blank" rel="noopener"' if external else ""
        return f'<a href="{html.escape(href, quote=True)}"{extra}>{label}</a>'

    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)
    return re.sub(
        "\x00(\\d+)\x00",
        lambda m: f"<code>{html.escape(spans[int(m.group(1))], quote=False)}</code>",
        text,
    )


@dataclass
class Heading:
    level: int
    text: str
    anchor: str


def render_markdown(src: str) -> tuple[str, list[Heading]]:
    """Markdown subset -> (html, headings) for building a contents sidebar."""
    lines = src.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    headings: list[Heading] = []
    i = 0
    used: set[str] = set()

    def close_list(stack: list[str]) -> None:
        while stack:
            out.append(f"</{stack.pop()}>")

    list_stack: list[str] = []

    while i < len(lines):
        line = lines[i]

        # fenced code
        if line.lstrip().startswith("```"):
            close_list(list_stack)
            lang = line.strip().strip("`").strip()
            i += 1
            body: list[str] = []
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            cls = f' class="lang-{html.escape(lang, quote=True)}"' if lang else ""
            out.append(f"<pre{cls}><code>{html.escape(chr(10).join(body), quote=False)}</code></pre>")
            continue

        stripped = line.strip()

        if not stripped:
            close_list(list_stack)
            i += 1
            continue

        # horizontal rule
        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            close_list(list_stack)
            out.append("<hr>")
            i += 1
            continue

        # heading
        m = re.match(r"(#{1,6})\s+(.*)", stripped)
        if m:
            close_list(list_stack)
            level = len(m.group(1))
            text = _inline(m.group(2).strip())
            anchor = slug(m.group(2))
            n = 2
            while anchor in used:
                anchor, n = f"{slug(m.group(2))}-{n}", n + 1
            used.add(anchor)
            if level >= 2:
                headings.append(Heading(level, re.sub(r"<[^>]+>", "", text), anchor))
            out.append(f'<h{level} id="{anchor}">{text}'
                       f'<a class="anchor" href="#{anchor}" aria-label="link">#</a></h{level}>')
            i += 1
            continue

        # table: a pipe row followed by a |---| separator
        if stripped.startswith("|") and i + 1 < len(lines) and re.fullmatch(
            r"\|[\s:|-]+\|", lines[i + 1].strip()
        ):
            close_list(list_stack)

            def cells(row: str) -> list[str]:
                return [c.strip() for c in row.strip().strip("|").split("|")]

            header = cells(stripped)
            i += 2
            body_rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                body_rows.append(cells(lines[i].strip()))
                i += 1
            head_html = "".join(f"<th>{_inline(c)}</th>" for c in header)
            rows_html = "".join(
                "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                for r in body_rows
            )
            out.append(f"<table><thead><tr>{head_html}</tr></thead><tbody>{rows_html}</tbody></table>")
            continue

        # blockquote (possibly several lines, possibly containing list items)
        if stripped.startswith(">"):
            close_list(list_stack)
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(re.sub(r"^\s*>\s?", "", lines[i]))
                i += 1
            inner, _ = render_markdown("\n".join(quote))
            out.append(f"<blockquote>{inner}</blockquote>")
            continue

        # lists
        m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)", line)
        if m:
            indent, marker, content = len(m.group(1)), m.group(2), m.group(3)
            kind = "ol" if marker.endswith(".") else "ul"
            depth = indent // 2
            while len(list_stack) > depth + 1:
                out.append(f"</{list_stack.pop()}>")
            if len(list_stack) == depth + 1 and list_stack[-1] != kind:
                out.append(f"</{list_stack.pop()}>")
            if len(list_stack) < depth + 1:
                out.append(f"<{kind}>")
                list_stack.append(kind)
            # continuation lines belong to the same item
            i += 1
            while (
                i < len(lines)
                and lines[i].strip()
                and not re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i])
                and not lines[i].strip().startswith(("#", ">", "|", "```"))
                and len(lines[i]) - len(lines[i].lstrip()) > indent
            ):
                content += " " + lines[i].strip()
                i += 1
            out.append(f"<li>{_inline(content)}</li>")
            continue

        # paragraph
        close_list(list_stack)
        para = [stripped]
        i += 1
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i])
            and not lines[i].strip().startswith(("#", ">", "|", "```", "---"))
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline(' '.join(para))}</p>")

    close_list(list_stack)
    return "\n".join(out), headings


PAGE = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · MultiSpy</title>
<style>
  :root {{
    --bg:#15171b; --surface:#1c1f25; --surface-2:#23272f; --border:#32373f;
    --text:#e4e7ec; --dim:#969ca8; --accent:#5a9cff; --good:#43c47c; --warn:#e5a53f;
  }}
  *{{box-sizing:border-box}}
  html,body{{margin:0;height:100%}}
  body{{background:var(--bg);color:var(--text);
    font:15px/1.65 "Segoe UI",system-ui,-apple-system,sans-serif}}
  a{{color:var(--accent);text-decoration:none}}
  a:hover{{text-decoration:underline}}

  header{{position:sticky;top:0;z-index:10;display:flex;align-items:center;gap:14px;
    padding:10px 20px;background:var(--surface);border-bottom:1px solid var(--border)}}
  header .brand{{font-weight:600}}
  header .brand span{{color:var(--dim);font-weight:400;margin-left:6px}}
  header nav{{display:flex;gap:1px;margin-left:auto}}
  header nav a{{background:var(--surface-2);border:1px solid var(--border);
    padding:5px 12px;color:var(--text)}}
  header nav a:first-child{{border-radius:5px 0 0 5px}}
  header nav a:last-child{{border-radius:0 5px 5px 0}}
  header nav a:hover{{background:#2c313a;text-decoration:none}}
  header nav a.on{{background:var(--accent);border-color:var(--accent);color:#0d1016;font-weight:600}}

  .wrap{{display:grid;grid-template-columns:260px minmax(0,1fr);gap:34px;
    max-width:1180px;margin:0 auto;padding:26px 20px 90px}}
  nav.toc{{position:sticky;top:64px;align-self:start;max-height:calc(100vh - 90px);
    overflow-y:auto;font-size:13px;border-left:1px solid var(--border);padding-left:14px}}
  nav.toc div{{padding:3px 0}}
  nav.toc .l3{{padding-left:14px;font-size:12.5px}}
  nav.toc a{{color:var(--dim)}}
  nav.toc a:hover,nav.toc a.active{{color:var(--text)}}

  article{{min-width:0}}
  article h1{{font-size:30px;margin:0 0 6px;letter-spacing:-.2px}}
  article h2{{font-size:22px;margin:38px 0 12px;padding-top:10px;
    border-top:1px solid var(--border)}}
  article h3{{font-size:17px;margin:26px 0 8px}}
  article h4{{font-size:15px;margin:20px 0 6px;color:var(--dim);
    text-transform:uppercase;letter-spacing:.6px}}
  article p{{margin:11px 0}}
  article ul,article ol{{margin:11px 0;padding-left:22px}}
  article li{{margin:5px 0}}
  article hr{{border:0;border-top:1px solid var(--border);margin:30px 0}}
  article code{{background:var(--surface-2);border:1px solid var(--border);
    border-radius:4px;padding:1px 5px;font-size:13px;
    font-family:"Cascadia Mono",Consolas,monospace}}
  article pre{{background:#101217;border:1px solid var(--border);border-radius:7px;
    padding:13px 15px;overflow-x:auto;margin:14px 0}}
  article pre code{{background:none;border:0;padding:0;font-size:13px;line-height:1.55}}
  article blockquote{{margin:16px 0;padding:10px 16px;background:var(--surface);
    border-left:3px solid var(--warn);border-radius:0 6px 6px 0}}
  article blockquote p{{margin:5px 0}}
  article table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:13.5px}}
  article th,article td{{border:1px solid var(--border);padding:7px 11px;
    text-align:left;vertical-align:top}}
  article th{{background:var(--surface-2);font-weight:600}}
  article tbody tr:nth-child(even){{background:#1a1d23}}
  article strong{{color:#fff}}
  .anchor{{opacity:0;margin-left:8px;color:var(--dim);font-weight:400}}
  h1:hover .anchor,h2:hover .anchor,h3:hover .anchor{{opacity:1}}

  @media (max-width:900px){{
    .wrap{{grid-template-columns:1fr;gap:0}}
    nav.toc{{position:static;max-height:none;border-left:0;border-bottom:1px solid var(--border);
      padding:0 0 14px;margin-bottom:18px}}
  }}
</style></head><body>
<header>
  <div class="brand">MultiSpy<span>{title}</span></div>
  <nav>
    <a href="/docs/tutorial"{tut_on}>Tutorial</a>
    <a href="/docs/readme"{ref_on}>Reference</a>
    <a href="/">&larr; Back to app</a>
  </nav>
</header>
<div class="wrap">
  <nav class="toc">{toc}</nav>
  <article>{body}</article>
</div>
<script>
  // highlight whichever section is currently on screen
  const links = [...document.querySelectorAll('nav.toc a')];
  const byId = new Map(links.map(a => [a.getAttribute('href').slice(1), a]));
  const heads = [...document.querySelectorAll('h2[id],h3[id]')];
  const onScreen = new Set();
  const spy = new IntersectionObserver((entries) => {{
    for (const e of entries) {{
      if (e.isIntersecting) onScreen.add(e.target.id);
      else onScreen.delete(e.target.id);
    }}
    links.forEach((a) => a.classList.remove('active'));
    for (const h of heads) {{
      if (onScreen.has(h.id)) {{
        const a = byId.get(h.id);
        if (a) a.classList.add('active');
        break;
      }}
    }}
  }}, {{ rootMargin: '-70px 0px -75% 0px' }});
  heads.forEach((h) => spy.observe(h));
</script>
</body></html>"""


def render_page(name: str) -> str | None:
    """Full HTML page for a doc route, or None if there is no such doc."""
    entry = DOCS.get(name)
    if entry is None:
        return None
    path = ROOT / entry[0]
    if not path.is_file():
        return None
    body, headings = render_markdown(path.read_text(encoding="utf-8"))
    toc = "".join(
        f'<div class="l{h.level}"><a href="#{h.anchor}">{h.text}</a></div>'
        for h in headings
        if h.level <= 3
    )
    return PAGE.format(
        title=entry[1],
        body=body,
        toc=toc,
        tut_on=' class="on"' if name == "tutorial" else "",
        ref_on=' class="on"' if name == "readme" else "",
    )
