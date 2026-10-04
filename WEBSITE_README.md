# Portfolio website

The [live portfolio](https://brightertiger.github.io/kaggle/) is a static GitHub Pages
site. Each of the 20 solution cards links to a rendered README at
`/kaggle/<competition>/` and to the corresponding code on GitHub. Five additional
competition cards link only to Kaggle because they have no solution folder.
Medal filters cover all 25 cards, including unranked entries under Other.

## Files

- `index.html`: portfolio cards, medal filters, and profile links.
- `styles.css`: shared fonts, color variables, responsive layout, and write-up typography.
- `script.js`: filtering and optional scroll animations.
- `build_site.py`: shared HTML template and README renderer for the 20 competition folders.
- `requirements-site.txt`: the Python Markdown dependency; the build otherwise uses the standard library.
- `package.json`: optional local server commands and `npm run build`.
- `.github/workflows/deploy.yml`: builds the write-ups and deploys the repository root.

## Build and validate locally

Install the site dependency into your Python environment, then build outside the
repository to leave competition folders untouched:

```bash
python -m pip install -r requirements-site.txt
python build_site.py --out /tmp/site_build_test
```

In this workspace, the equivalent commands are:

```bash
UV_CACHE_DIR=/tmp/kaggle-site-uv-cache uv pip install --python .venv/bin/python -r requirements-site.txt
.venv/bin/python build_site.py --out /tmp/site_build_test
```

The isolated output contains the homepage, shared CSS and JavaScript, and all 20
`<competition>/index.html` pages. You can inspect the generated HTML without a server.
The build preserves README-relative links and image paths; it does not copy competition
source files or images to the isolated output. CI uploads the original competition
folders alongside the generated pages, so existing relative assets retain their URLs.

For an optional browser preview, serve the output with
`python -m http.server 8000 --directory /tmp/site_build_test`.

Running `python build_site.py` (or `npm run build`) without `--out` writes pages into
the repository, as CI does. Those generated `*/index.html` files are gitignored;
edit the source READMEs instead of generated pages. Missing READMEs fail the build.

## Markdown support

Write-ups support headings with anchor IDs, lists, links, images, fenced code, and
Markdown tables. Fenced `mermaid` blocks become `<pre class="mermaid">` elements.
Pages with diagrams load Mermaid 11 from jsDelivr as an ES module and initialize it
with strict security settings. Diagram rendering needs JavaScript and CDN access;
the diagram source remains readable without them. Google Fonts supplies Inter,
with system fonts as a fallback.

The shared template loads `../styles.css`, links back to `../`, and adds a
“View code on GitHub” footer pointing to
`https://github.com/brightertiger/kaggle/tree/master/<competition>`.

## Deployment

On pushes to `main` or `master`, or a manual workflow dispatch, GitHub Actions:

1. Checks out the repository and sets up Python 3.12.
2. Installs `requirements-site.txt` and runs `python build_site.py`.
3. Uploads the repository root as the Pages artifact.
4. Deploys it to GitHub Pages.

The homepage remains at `/kaggle/`, and write-ups live at `/kaggle/<competition>/`.
No generated pages are committed. Changes to any competition README are rendered
on the next deployment.

## Updating the portfolio

Edit root `index.html` for cards and keep its ranks and medal tiers aligned with
root `README.md`. Use a local Write-up link and a GitHub Code link for each solution
folder. Leave undocumented ranks off cards. Add new competition folders to
`COMPETITIONS` in `build_site.py` and update the root competition tables as needed.
All cards use the same classes, so the medal filters include new cards automatically.
