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

#### Setting Up GitHub Pages

**Step 1: Enable GitHub Pages with GitHub Actions**

1. Go to your repository on GitHub: `https://github.com/Criobe/documentation`
2. Click on **Settings** (top menu bar)
3. In the left sidebar, scroll down and click on **Pages** (under "Code and automation")
4. Under **"Build and deployment"** section:
    - **Source**: Select **"GitHub Actions"** from the dropdown
    - (You don't need to select a branch when using GitHub Actions)
5. Click **Save** if prompted

**Step 2: Verify Workflow Permissions**

1. Still in **Settings**, go to **Actions** > **General** (in the left sidebar)
2. Scroll down to **"Workflow permissions"**
3. Ensure that **"Read repository contents and packages permissions"** is selected
4. Click **Save** if you made changes

**Step 3: Push Your Changes**

```bash
cd documentation/

# Check current status
git status

# Add all changes
git add .

# Commit with a descriptive message
git commit -m "Configure GitHub Actions workflow for documentation deployment"

# Push to main branch
git push origin main
```

**Step 4: Monitor the Deployment**

1. Go to the **Actions** tab in your GitHub repository
2. You should see a workflow run called "Deploy Documentation"
3. Click on it to watch the progress:
    - **Build Documentation** job should complete first
    - **Deploy to GitHub Pages** job will run after the build succeeds
4. Once complete, you'll see a green checkmark

**Step 5: Access Your Documentation**

After successful deployment, your documentation will be available at:
```
https://criobe.github.io/documentation/
```

Or check the exact URL in **Settings** > **Pages** - you'll see a box that says **"Your site is live at [URL]"**

Add this url into the website field in the About section (use the settings icon in the top right corner of the About section)


#### Troubleshooting

**Check the workflow logs**:
1. Go to Actions tab
2. Click on the failed workflow run
3. Click on the failed job to see error details

**Common issues**:
- **404 on deployment**: Make sure the `site/` directory was created in the build step
- **Permission denied**: Check that workflow permissions are set to "Read and write"
- **Build warnings with strict mode**: The workflow uses `--strict` flag, which fails on warnings. Check the build output for any warnings.

**Manual trigger**:
1. Go to Actions tab
2. Click "Deploy Documentation" in the left sidebar
3. Click "Run workflow" button
4. Select `main` branch and click "Run workflow"

#### Future Updates

After the initial setup, any push to the `main` branch that modifies `docs/**`, `mkdocs.yml`, or `.github/workflows/docs.yml` will automatically trigger a rebuild and redeployment.
