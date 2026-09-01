#!/usr/bin/env python3
"""Build ``docs/paper/paper.pdf`` from ``docs/paper/paper.md``.

    python scripts/build_paper.py

No TeX distribution is required, which matters because none is installable in
the environment this repository is developed in (Debian mirrors are behind a
blocking egress proxy). The chain instead is:

    paper.md  --python-markdown-->  HTML
              --KaTeX (npm)------>  MathML for every $...$ and $$...$$
              --Chromium --print-to-pdf-->  paper.pdf

KaTeX is asked for **MathML output specifically**, not its usual HTML+CSS
output. MathML is rendered natively by Chromium 109+, so the resulting page
needs no web fonts and no stylesheet, and stays a single self-contained file
that can also be published as an artifact.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO / "docs" / "paper" / "paper.md"

#: Where the KaTeX install lives. Kept out of the repository: it is a build
#: dependency, not source.
NODE_CACHE = Path(os.environ.get("ASTRO_NODE_CACHE", "/tmp/astro-paper-node"))

_RENDER_JS = r"""
const katex = require(process.env.KATEX_MODULE);
let raw = "";
process.stdin.on("data", d => raw += d);
process.stdin.on("end", () => {
  const items = JSON.parse(raw);
  const out = items.map(({tex, display}) => {
    try {
      return katex.renderToString(tex, {
        displayMode: display, output: "mathml", throwOnError: false, strict: false,
      });
    } catch (e) {
      return "<code>" + tex.replace(/</g, "&lt;") + "</code>";
    }
  });
  process.stdout.write(JSON.stringify(out));
});
"""

STYLE = """
:root {
  --ink: #111418; --muted: #55606b; --rule: #d7dde3; --accent: #1f4e79;
  --bg: #ffffff; --panel: #f6f8fa; --panel-rule: #dce3ea;
}
* { box-sizing: border-box; }
body {
  background: var(--bg); color: var(--ink); margin: 0 auto; padding: 2.2rem 1.6rem 4rem;
  max-width: 52rem; font: 10.6pt/1.58 "Iowan Old Style", "Palatino Linotype", Palatino,
  "Book Antiqua", Georgia, serif; text-rendering: optimizeLegibility;
}
h1 { font-size: 1.72rem; line-height: 1.22; margin: 0 0 .3rem; letter-spacing: -.01em; }
h2 { font-size: 1.12rem; margin: 2.1rem 0 .6rem; padding-bottom: .22rem;
     border-bottom: 1px solid var(--rule); letter-spacing: .01em; }
h3 { font-size: .99rem; margin: 1.35rem 0 .4rem; color: var(--accent); }
h4 { font-size: .93rem; margin: 1rem 0 .3rem; color: var(--muted);
     text-transform: uppercase; letter-spacing: .06em; }
p, li { orphans: 3; widows: 3; }
a { color: var(--accent); }
code, pre { font-family: "SF Mono", "DejaVu Sans Mono", ui-monospace, monospace; }
code { font-size: .87em; background: var(--panel); padding: .08em .3em; border-radius: 3px; }
pre { background: var(--panel); border: 1px solid var(--panel-rule);
      border-left: 3px solid var(--accent); border-radius: 4px; padding: .75rem .9rem;
      overflow-x: auto; font-size: .8rem; line-height: 1.45; }
pre code { background: none; padding: 0; font-size: inherit; }
blockquote { margin: 1rem 0; padding: .5rem 0 .5rem .9rem; border-left: 3px solid var(--rule);
             color: var(--muted); }
table { border-collapse: collapse; width: 100%; margin: .9rem 0; font-size: .84rem;
        font-family: system-ui, -apple-system, sans-serif; }
th, td { border-bottom: 1px solid var(--rule); padding: .34rem .5rem; text-align: left;
         vertical-align: top; }
th { background: var(--panel); font-weight: 600; border-bottom: 1.5px solid var(--muted); }
td:not(:first-child), th:not(:first-child) { text-align: right; white-space: nowrap; }
.table-wrap { overflow-x: auto; }
math { font-size: 1.02em; }
mstyle, math[display="block"] { margin: .5rem 0; }
.abstract { background: var(--panel); border: 1px solid var(--panel-rule); border-radius: 5px;
            padding: .9rem 1.1rem; margin: 1.4rem 0 1.8rem; font-size: .95rem; }
.abstract h4 { margin-top: 0; }
.byline { color: var(--muted); font-size: .92rem; margin: 0 0 1.2rem; }
.algorithm { border: 1px solid var(--muted); border-radius: 4px;
             padding: .1rem .9rem .7rem; margin: 1.1rem 0; background: #fff;
             page-break-inside: avoid; }
.algorithm > .algorithm-title { background: var(--ink); color: #fff; font-size: .78rem;
             letter-spacing: .07em; text-transform: uppercase; padding: .3rem .9rem;
             margin: 0 -.9rem .6rem; font-family: system-ui, sans-serif; }
.algorithm pre { background: none; border: none; border-left: none; padding: 0;
                 font-size: .78rem; }
.footnote { font-size: .84rem; color: var(--muted); }
hr { border: none; border-top: 1px solid var(--rule); margin: 2rem 0; }
@media print {
  body { max-width: none; padding: 0; font-size: 10pt; }
  h2 { page-break-after: avoid; } h3 { page-break-after: avoid; }
  pre, table, .algorithm, .abstract { page-break-inside: avoid; }
}
@page { size: A4; margin: 17mm 16mm 18mm; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --ink: #e8ecf1; --muted: #9aa7b4; --rule: #2c333b; --accent: #7fb3e0;
    --bg: #14181d; --panel: #1c222a; --panel-rule: #2c333b;
  }
  :root:not([data-theme="light"]) .algorithm { background: var(--panel); }
  :root:not([data-theme="light"]) .algorithm > .algorithm-title {
    background: var(--accent); color: #0d1117; }
}
:root[data-theme="dark"] {
  --ink: #e8ecf1; --muted: #9aa7b4; --rule: #2c333b; --accent: #7fb3e0;
  --bg: #14181d; --panel: #1c222a; --panel-rule: #2c333b;
}
"""


def ensure_katex() -> Path:
    """Install KaTeX into the build cache if absent; return its module path."""
    module = NODE_CACHE / "node_modules" / "katex"
    if module.exists():
        return module
    if shutil.which("npm") is None:
        raise SystemExit("npm is required to render math; install Node.js")
    NODE_CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "npm", "install", "--silent", "--no-fund", "--no-audit",
            "--prefix", str(NODE_CACHE), "katex",
        ],
        check=True,
        capture_output=True,
    )
    return module


def render_math(items: list[tuple[str, bool]]) -> list[str]:
    """Render ``(tex, display)`` pairs to MathML using KaTeX under Node."""
    if not items:
        return []
    module = ensure_katex()
    payload = json.dumps([{"tex": tex, "display": display} for tex, display in items])
    result = subprocess.run(
        ["node", "-e", _RENDER_JS],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
        # Passed via the environment rather than argv: `node -e` does not place
        # extra arguments where a normal script would, and the resulting
        # off-by-one is silent until KaTeX fails to load.
        env={**os.environ, "KATEX_MODULE": str(module)},
    )
    return json.loads(result.stdout)


_MATH = re.compile(r"\$\$(.+?)\$\$|(?<![\\$])\$(?!\s)((?:[^$\\]|\\.)+?)(?<!\s)\$", re.S)


def extract_math(text: str) -> tuple[str, list[tuple[str, bool]]]:
    """Replace math spans with placeholders that Markdown will not touch."""
    found: list[tuple[str, bool]] = []

    def swap(match: re.Match[str]) -> str:
        display = match.group(1) is not None
        found.append(((match.group(1) or match.group(2)).strip(), display))
        # Placeholder must survive Markdown's inline processing untouched.
        return f"\x00MATH{len(found) - 1}\x00"

    return _MATH.sub(swap, text), found


_ALGORITHM = re.compile(r"^:::algorithm\s+(.+?)\n(.*?)^:::\s*$", re.S | re.M)


def extract_algorithms(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Pull ``:::algorithm Title ... :::`` blocks out as verbatim pseudocode."""
    found: list[tuple[str, str]] = []

    def swap(match: re.Match[str]) -> str:
        found.append((match.group(1).strip(), match.group(2).rstrip("\n")))
        return f"\n\x00ALGO{len(found) - 1}\x00\n"

    return _ALGORITHM.sub(swap, text), found


def build_html(source: Path) -> str:
    """Render the paper source to a single self-contained HTML document."""
    import markdown

    text = source.read_text(encoding="utf-8")
    text, algorithms = extract_algorithms(text)
    text, math_items = extract_math(text)

    body = markdown.markdown(
        text, extensions=["tables", "fenced_code", "attr_list", "toc", "footnotes", "md_in_html"]
    )

    for index, rendered in enumerate(render_math(math_items)):
        body = body.replace(f"\x00MATH{index}\x00", rendered)
    for index, (title, code) in enumerate(algorithms):
        block = (
            f'<div class="algorithm"><div class="algorithm-title">{html.escape(title)}</div>'
            f"<pre><code>{html.escape(code)}</code></pre></div>"
        )
        body = re.sub(rf"<p>\x00ALGO{index}\x00</p>|\x00ALGO{index}\x00", block, body)

    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body)
    body = re.sub(r"</table>", "</table></div>", body)

    title_match = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    title = re.sub(r"<[^>]+>", "", title_match.group(1)) if title_match else "ASTRO"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(title)}</title><style>{STYLE}</style></head>"
        f"<body>{body}</body></html>"
    )


def chromium_binary() -> str:
    """Locate the pre-installed Chromium used for printing."""
    candidates = [
        os.environ.get("CHROMIUM_BINARY", ""),
        "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
        shutil.which("chromium") or "",
        shutil.which("chromium-browser") or "",
        shutil.which("google-chrome") or "",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    roots = sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))
    if roots:
        return str(roots[-1])
    raise SystemExit("no Chromium binary found; set CHROMIUM_BINARY")


def write_pdf(html_path: Path, pdf_path: Path) -> None:
    """Print ``html_path`` to ``pdf_path`` with headless Chromium."""
    subprocess.run(
        [
            chromium_binary(),
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.source.exists():
        raise SystemExit(f"missing paper source: {args.source}")

    document = build_html(args.source)
    html_path = args.source.with_suffix(".html")
    html_path.write_text(document, encoding="utf-8")
    print(f"wrote {html_path} ({len(document) / 1024:.0f} KB)")

    if args.html_only:
        return 0

    pdf_path = args.source.with_suffix(".pdf")
    write_pdf(html_path, pdf_path)
    size = pdf_path.stat().st_size
    print(f"wrote {pdf_path} ({size / 1024:.0f} KB)")
    if size < 2048:
        raise SystemExit("PDF looks empty; check the Chromium invocation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
