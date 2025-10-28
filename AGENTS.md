# Repository Guidelines

## Project Structure & Module Organization
This MkDocs site keeps documentation at the repository root because `docs_dir: .` in `mkdocs.yml`. Global guides such as `bridge.md` and `setup_nuclio.md` sit alongside configuration files. Language-specific content lives in `en/` and `fr/`; create matching directories when adding translations. Generated static files appear in `site/` (gitignored). Update navigation in `mkdocs.yml` whenever you add or rename pages.

## Build, Test, and Development Commands
```bash
python -m venv .venv && source .venv/bin/activate  # isolate dependencies
pip install -r requirements.txt                   # install MkDocs + theme
mkdocs serve                                      # live preview with reload
mkdocs build --strict                             # production build + link checks
```
Run commands from the repository root. `mkdocs serve` exposes docs at http://127.0.0.1:8000/. Use `--dirtyreload` for faster rebuilds during long sessions.

## Coding Style & Naming Conventions
Write Markdown in English or French with sentence-case headings. Keep filenames lowercase with hyphens (`setup_nuclio.md`), and mirror the navigation depth. Prefer short sections, bullet lists, and fenced code blocks with language hints. Use relative links across locales and Material callouts (e.g., `!!! note`) sparingly but consistently for warnings or prerequisites.

## Testing Guidelines
Before opening a PR, run `mkdocs build --strict` to catch broken links, missing assets, or configuration drift. Verify new screenshots or diagrams by checking them in the generated `site/` output. For localization work, diff the English and French pages to ensure parity and update anchors referenced elsewhere.

## Commit & Pull Request Guidelines
Use imperative, scoped commit messages such as `docs: add nuclio troubleshooting tips`. Squash work-in-progress commits locally. PRs should describe the change, reference related tasks, and mention affected locales. Include screenshots or the live preview URL for visual updates, plus confirmation that `mkdocs build --strict` passes.
