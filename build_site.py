#!/usr/bin/env python3
"""Render competition READMEs for GitHub Pages, or into an isolated --out tree."""

import argparse
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import re
import shutil

import markdown


ROOT = Path(__file__).resolve().parent
COMPETITIONS = (
    "amexpert", "aptos", "avito", "doodle", "imet", "jigsaw", "jigsawml",
    "leaf", "pronoun", "quest", "rfcx", "rsna", "salt", "siim", "spooky",
    "statoil", "talking", "toxic", "tweet", "whale",
)
GITHUB = "https://github.com/brightertiger/kaggle/tree/master"
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title} | Kaggle Portfolio</title>
    <link rel="stylesheet" href="../styles.css">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&amp;display=swap">
</head>
<body>
    <header class="header writeup-header">
        <div class="container">
            <nav class="links" aria-label="Portfolio navigation">
                <a href="../">← Back to portfolio</a>
            </nav>
        </div>
    </header>
    <main class="container writeup">
        <article>
{content}
        </article>
    </main>
    <footer class="footer">
        <div class="container footer-links">
            <a href="{code_url}">View code on GitHub</a>
        </div>
    </footer>
{mermaid_script}
</body>
</html>
"""
MERMAID_SCRIPT = """    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
        mermaid.initialize({ startOnLoad: true, securityLevel: 'strict', theme: 'neutral' });
    </script>"""


class HeadingText(HTMLParser):
    """Extract a plain browser title from the first rendered level-one heading."""

    def __init__(self) -> None:
        super().__init__()
        self.parts = []
        self.in_heading = False
        self.finished = False

    def handle_starttag(self, tag, attrs):
        if tag == "h1" and not self.finished:
            self.in_heading = True

    def handle_endtag(self, tag):
        if tag == "h1":
            self.in_heading = False
            self.finished = True

    def handle_data(self, data):
        if self.in_heading:
            self.parts.append(data)


def render_page(source: str, competition: str) -> str:
    """Preserve folder-relative URLs and turn fenced Mermaid into diagram elements."""
    content = markdown.markdown(
        source, extensions=["fenced_code", "tables", "toc"], output_format="html"
    )
    # Work on Markdown's escaped code output, so diagram labels cannot become HTML.
    content, diagrams = re.subn(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<pre class="mermaid">\1</pre>',
        content,
        flags=re.DOTALL,
    )
    heading = HeadingText()
    heading.feed(content)
    title = "".join(heading.parts).strip() or competition
    return TEMPLATE.format(
        title=escape(title),
        content=content,
        code_url=f"{GITHUB}/{competition}",
        mermaid_script=MERMAID_SCRIPT if diagrams else "",
    )


def build(output: Path) -> None:
    # Read and render everything before writing, so a missing README fails clearly.
    pages = {
        competition: render_page(
            (ROOT / competition / "README.md").read_text(encoding="utf-8"), competition
        )
        for competition in COMPETITIONS
    }
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != ROOT:
        for asset in ("index.html", "styles.css", "script.js"):
            shutil.copy2(ROOT / asset, output / asset)
    for competition, page in pages.items():
        destination = output / competition / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(page, encoding="utf-8")
        print(f"Built {competition}/index.html")
    print(f"Rendered {len(pages)} write-ups in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT,
        help="Output root (default: repository root, as used by GitHub Pages CI)",
    )
    args = parser.parse_args()
    try:
        build(args.out.resolve())
    except OSError as error:
        parser.exit(1, f"Site build failed: {error}\n")


if __name__ == "__main__":
    main()
