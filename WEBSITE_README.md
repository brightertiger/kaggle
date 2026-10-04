# Portfolio website

The [live portfolio](https://brightertiger.github.io/kaggle/) is a static academic-style
portfolio with 25 numbered competition entries grouped by Gold, Silver, Bronze, and
Other. All entries link to their competition; 20 also link to a rendered solution
README and the code on GitHub. The five entries without solution folders retain
competition links only. AmExpert links to its original Analytics Vidhya competition.
Medal navigation uses native anchors and works without JavaScript.

## Design and source files

- `index.html`: all competition titles, ranks, team counts, descriptions, tags, links,
  medal totals, and profile links.
- `styles.css`: the only stylesheet, shared by the homepage and write-ups.
- `script.js`: explanatory comments only; the homepage requires no script.
- `build_site.py`: shared page template, README renderer, and stylesheet versioning.
- `requirements-site.txt`: Python Markdown dependency.
- `package.json`: optional local server and build commands.
- `.github/workflows/deploy.yml`: builds write-ups and deploys the repository root.

The design follows the author's personal website: system sans-serif body text,
white background, `#242424` ink, `#4c4c4c` body text, `#707070` metadata,
`#e8e8e8` rules, understated underlined links, and the same responsive reading width
(`clamp(808px, 56vw, 1040px)`). Georgia headings give the competition list an academic
feel. Spacing follows the reference's 8/12/16/24/32/36/40/52/64/72px rhythm.
Mobile gutters are 16px. Like the reference, the site is light-only. No web fonts,
frameworks, card effects, or animations are loaded.

## Build and validate locally

Install `requirements-site.txt` into your Python environment, then build outside the
repository to leave competition folders untouched:

```bash
.venv/bin/python build_site.py --out /tmp/site_overhaul_test
python -m http.server 8000 --directory /tmp/site_overhaul_test
```

The isolated output contains the homepage, shared stylesheet, and all 20
`<competition>/index.html` pages. Open the preview at `http://localhost:8000`.
README-relative links and image paths are preserved. The isolated output does not
copy competition source files or images. CI uploads the original competition folders
alongside generated pages, so those assets retain their URLs in the deployed site.

Running `python build_site.py` (or `npm run build`) without `--out` writes pages into
the repository, as CI does. Generated `*/index.html` files are gitignored and should
not be committed. Missing READMEs fail the build before output is written.

## Markdown and accessibility

Write-ups share the author's header, profile links, and footer with the homepage,
and include Back to portfolio and View code on GitHub links. Markdown supports
heading anchors, lists, links, images, fenced code, blockquotes, and tables. Wide
tables and code blocks scroll horizontally within the reading column. Tables have
focusable, labeled scroll regions; code blocks are keyboard focusable. Both page
types include skip links and visible keyboard focus outlines.

Fenced `mermaid` blocks become `<pre class="mermaid">` elements. Only pages with
diagrams load Mermaid 11 from jsDelivr, with strict security and a neutral theme.
Diagram rendering requires JavaScript and CDN access; the source remains readable
without them. No other external scripts are needed.

## Deployment and cache busting

The existing GitHub Actions workflow builds the site on pushes to `main` or `master`,
or on manual dispatch, then deploys the repository root to GitHub Pages.

Every build creates one UTC timestamp and appends `?v=<timestamp>` to the stylesheet
URL in all 20 write-ups and the output homepage. With `--out`, only the output copy
of the homepage is rewritten; CI's default build rewrites the root homepage in its
checkout. The source homepage also has a version suffix for direct local previews.
This gives each deployment a new stylesheet URL so browsers request the new design.

## Updating entries

Edit root `index.html` to update entries. Preserve ranks, medals, descriptions, tags,
and existing competition, Write-up, and Code destinations. Keep the numbered lists'
`start` values and summary counts aligned when adding entries. Do not invent ranks
or solution links for entries without recorded results or a solution folder.
