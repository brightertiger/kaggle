#!/usr/bin/env python3
"""Render competition READMEs for GitHub Pages, or into an isolated --out tree."""

import argparse
from datetime import datetime, timezone
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
    <link rel="stylesheet" href="../styles.css?v={version}">
</head>
<body>
    <a class="skip-link" href="#main">Skip to content</a>
    <header class="site-header">
        <div class="container">
            <div>
                <p class="name">Ujjwal Singh Rao</p>
                <p class="tagline">Kaggle Master · ML Engineer · Data Scientist</p>
            </div>
            <nav class="profile-links" aria-label="Profiles">
                <a href="https://www.kaggle.com/brightertiger" target="_blank" rel="noopener noreferrer">Kaggle</a>
                <a href="https://github.com/brightertiger" target="_blank" rel="noopener noreferrer">GitHub</a>
                <a href="https://linkedin.com/in/brightertiger" target="_blank" rel="noopener noreferrer">LinkedIn</a>
                <a href="https://brightertiger.xyz" target="_blank" rel="noopener noreferrer">Website</a>
            </nav>
        </div>
    </header>
    <main id="main" class="container writeup">
        <nav class="article-nav" aria-label="Write-up navigation">
            <a href="../">← Back to portfolio</a>
            <a href="{code_url}">View code on GitHub</a>
        </nav>
        <article class="prose">
{content}
        </article>
    </main>
    <footer class="site-footer">
        <div class="container">
            <span>Ujjwal Singh Rao · Competition notes</span>
            <nav class="profile-links" aria-label="More about the author">
                <a href="https://github.com/brightertiger" target="_blank" rel="noopener noreferrer">GitHub</a>
                <a href="https://linkedin.com/in/brightertiger" target="_blank" rel="noopener noreferrer">LinkedIn</a>
                <a href="https://brightertiger.xyz" target="_blank" rel="noopener noreferrer">Website</a>
                <a href="https://kaggle.com/brightertiger" target="_blank" rel="noopener noreferrer">Kaggle</a>
            </nav>
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


def render_page(source: str, competition: str, version: str) -> str:
    """Preserve folder-relative URLs and turn fenced Mermaid into diagram elements."""
    content = markdown.markdown(
        source, extensions=["fenced_code", "tables", "toc"], output_format="html"
    )
    # Work on Markdown's escaped code output, so diagram labels cannot become HTML.
    content, diagrams = re.subn(
        r'<pre><code class="language-mermaid">(.*?)</code></pre>',
        r'<pre class="mermaid" tabindex="0">\1</pre>',
        content,
        flags=re.DOTALL,
    )
    # Keep wide tables within a keyboard-accessible horizontal scroll region.
    content = content.replace(
        "<table>", '<div class="table-scroll" role="region" aria-label="Data table" tabindex="0"><table>'
    ).replace("</table>", "</table></div>")
    content = content.replace("<pre>", '<pre tabindex="0">')
    heading = HeadingText()
    heading.feed(content)
    title = "".join(heading.parts).strip() or competition
    return TEMPLATE.format(
        title=escape(title),
        version=version,
        content=content,
        code_url=f"{GITHUB}/{competition}",
        mermaid_script=MERMAID_SCRIPT if diagrams else "",
    )


def build(output: Path) -> None:
    version = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    # Read and render everything before writing, so a missing README fails clearly.
    pages = {
        competition: render_page(
            (ROOT / competition / "README.md").read_text(encoding="utf-8"), competition, version
        )
        for competition in COMPETITIONS
    }
    home = re.sub(
        r'href="styles\.css(?:\?v=[^" ]*)?"',
        f'href="styles.css?v={version}"',
        (ROOT / "index.html").read_text(encoding="utf-8"),
    )
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != ROOT:
        shutil.copy2(ROOT / "styles.css", output / "styles.css")
    (output / "index.html").write_text(home, encoding="utf-8")
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
