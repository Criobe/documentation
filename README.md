# CRIOBE Documentation

A MkDocs site that centralizes deployment and operations guides for the CRIOBE tooling stack: CVAT bridge workflows, Nuclio/serverless functions, and related dashboards in both English and French.

## Requirements
- Python 3.10+
- `pip` (or another installer such as `uv`)

## Quick start
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install the MkDocs toolchain:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the live preview server:
   ```bash
   mkdocs serve
   ```
   The documentation will be available at `http://127.0.0.1:8000/` with live reload on save.

## Project layout
- `mkdocs.yml` – MkDocs configuration including theme selection and navigation.
- `requirements.txt` – pinned dependencies for reproducible builds.
- `en/` – English documentation (CVAT bridge, Nuclio setup, serverless workflow).
- `fr/` – French documentation.
- `coral_segmentation_pipeline.md` – Shared overview referenced on the landing page.

## Deployment
To publish a static build, run:
```bash
mkdocs build
```
The generated site will be written to the `site/` directory (ignored by git). Deploy those files to any static hosting provider (GitHub Pages, Netlify, S3, etc.).
