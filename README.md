# CRIOBE Documentation

A MkDocs site that centralizes deployment and operations guides for the CRIOBE tooling stack: CVAT bridge workflows, Nuclio/serverless functions, and related dashboards.

**Note**: Currently only the English version is available. French translation is planned for future releases.

## Requirements
- [Pixi](https://pixi.sh/) (recommended) or Python 3.10+

## Quick start

### Using Pixi (recommended)
1. Install dependencies and activate the environment:
   ```bash
   pixi shell
   ```
2. Launch the live preview server:
   ```bash
   pixi run serve
   ```
   The documentation will be available at `http://127.0.0.1:8000/` with live reload on save.

### Using pip
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

## Available Commands

Using Pixi:
- `pixi run serve` – Start development server at http://127.0.0.1:8000
- `pixi run serve-8010` – Start development server at http://127.0.0.1:8010
- `pixi run build` – Build static site to `site/` directory
- `pixi run build-strict` – Build with strict mode (fail on warnings)

Using MkDocs directly (after `pixi shell` or in virtual environment):
- `mkdocs serve` – Start development server
- `mkdocs build` – Build static site
- `mkdocs build --strict` – Build with strict mode

## Project layout

```
documentation/
├── .github/
│   └── workflows/
│       └── docs.yml                      # GitHub Actions workflow for automated builds and deployment
├── docs/                                 # Documentation source files (Markdown)
│   ├── index.md                          # Landing page
│   ├── assets/                           # Images, diagrams, CSS/JS customizations
│   ├── quickstart/                       # Quick start guides for users and developers
│   ├── setup/                            # Installation and configuration guides
│   │   ├── requirements.md
│   │   ├── installation/                 # Installation steps (end-users vs developers)
│   │   └── configuration/                # Configuration guides
│   ├── user-guide/                       # User documentation
│   │   ├── concepts/                     # Core concepts and architecture
│   │   ├── data-preparation/             # Data pipeline workflows
│   │   ├── training-and-deployment/      # Model training and deployment guides
│   │   ├── reference/                    # Technical references (CVAT templates, etc.)
│   │   └── tutorials/                    # Step-by-step tutorials
│   ├── developer-guide/                  # Developer documentation
│   └── community/                        # Community and contribution guides
├── mkdocs.yml                            # MkDocs configuration (theme, navigation, plugins)
├── pixi.toml                             # Pixi environment and dependency configuration
├── pixi.lock                             # Pixi lock file (auto-generated)
├── requirements.txt                      # Pip-compatible dependencies (generated from pixi.toml for GitHub Actions)
└── README.md                             # This file
```

**Key Files**:
- **`pixi.toml`**: Primary dependency source. Defines Python version and all MkDocs dependencies.
- **`requirements.txt`**: Generated from `pixi.toml` pypi-dependencies section for GitHub Actions compatibility. Keep in sync with pixi.toml when updating dependencies.
- **`mkdocs.yml`**: MkDocs configuration including Material theme, navigation structure, and plugins.
- **`.github/workflows/docs.yml`**: Automated CI/CD pipeline that builds and deploys to GitHub Pages.

## Deployment

### Local Build
To publish a static build, run:
```bash
pixi run build
# or
mkdocs build
```
The generated site will be written to the `site/` directory (ignored by git).

### GitHub Actions
The repository includes a GitHub Actions workflow (`.github/workflows/docs.yml`) that automatically:
- Builds the documentation on push to `main` or pull requests
- Deploys to GitHub Pages on merge to `main`
- Runs link checks on pull requests

To enable GitHub Pages deployment:
1. Go to repository Settings → Pages
2. Set Source to "GitHub Actions"
3. Push to `main` branch
