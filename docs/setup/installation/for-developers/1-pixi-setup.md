# Pixi Setup

Install and configure Pixi, the package manager for Python environments used across all QUADRATSEG modules.

!!! info "For Developers"
    This guide is for **AI researchers and developers** who want to train, evaluate, and experiment with ML models. If you only want to process coral images, see the [End User Installation Guide](../for-end-users/1-docker-deployment.md) instead.

**Time Required**: 15-20 minutes
**Prerequisites**: Linux or macOS system with sudo access

## What is Pixi?

Pixi is a fast, cross-platform package manager that replaces conda/mamba for managing Python environments. All QUADRATSEG modules use Pixi for dependency management.

**Why Pixi?**
- **Fast**: 10-100x faster than conda
- **Reproducible**: Lock files ensure consistent environments
- **Simple**: Single `pixi.toml` file per module
- **Cross-platform**: Works on Linux, macOS, and Windows (WSL2)
- **Multi-environment**: Each module can have multiple environments (train, dev, deploy)

## System Requirements

### Operating System

| OS | Support | Notes |
|----|---------|-------|
| Linux | ✅ Full support | Recommended for training |
| macOS | ✅ Full support | CPU-only (no CUDA) |
| Windows | ⚠️ Via WSL2 | Use Ubuntu 22.04 in WSL2 |

### Software Prerequisites

Before installing Pixi, ensure you have:

- **bash or zsh shell**: For installation script
- **curl or wget**: For downloading installer
- **sudo access**: For system-wide installation (optional)
- **Git**: For cloning the repository

### Hardware Requirements

For development and training:

- **CPU**: 4+ cores recommended (8+ for faster training)
- **RAM**: 16GB minimum, 32GB+ recommended
- **Disk**: 50GB+ free space for datasets and models
- **GPU**: NVIDIA GPU with CUDA support (for GPU training)

## Step 1: Install Pixi

Pixi can be installed system-wide or per-user.

### Option A: User Installation (Recommended)

Install Pixi for your user only (no sudo required):

```bash
# Download and run installer
curl -fsSL https://pixi.sh/install.sh | bash

# Or using wget
wget -qO- https://pixi.sh/install.sh | bash
```

**Installation Location**: `~/.pixi/bin/pixi`

### Option B: System-Wide Installation

Install Pixi for all users:

```bash
# Download and run installer with sudo
curl -fsSL https://pixi.sh/install.sh | sudo bash

# Or using wget
wget -qO- https://pixi.sh/install.sh | sudo bash
```

**Installation Location**: `/usr/local/bin/pixi`

### Verify Installation

After installation, restart your shell or source your profile:

```bash
# For bash users
source ~/.bashrc

# For zsh users
source ~/.zshrc

# Verify pixi is available
pixi --version

# Expected output: pixi 0.x.x
```

!!! success "Pixi Installed"
    If you see the version number, Pixi is correctly installed!

## Step 2: Clone Repository

Clone the QUADRATSEG repository to your local machine:

```bash
# Navigate to your projects directory
cd ~/Projects  # or your preferred location

# Clone repository
git clone https://github.com/criobe/coral-segmentation.git

# Enter repository
cd coral-segmentation

# Verify repository structure
ls -la
# You should see: bridge/, coral_seg_yolo/, DINOv2_mmseg/,
#                 data_engineering/, grid_inpainting/,
#                 grid_pose_detection/, documentation/, etc.
```

## Step 3: Understand Repository Structure

The repository is organized into multiple modules, each with its own Pixi environment:

```
coral-segmentation/
├── coral_seg_yolo/           # YOLO-based coral segmentation
│   └── pixi.toml            # Environments: coral-seg-yolo, coral-seg-yolo-dev
├── DINOv2_mmseg/            # DINOv2-based segmentation
│   └── pixi.toml            # Environment: dinov2-mmseg
├── grid_pose_detection/     # Grid corner and pose detection
│   └── pixi.toml            # Environments: grid-pose, grid-pose-dev
├── grid_inpainting/         # Grid removal via inpainting
│   └── pixi.toml            # Environments: grid-inpainting, grid-inpainting-deploy
├── data_engineering/        # Dataset management with FiftyOne
│   └── pixi.toml            # Environment: default
├── bridge/                  # Webhook automation service
│   └── pixi.toml            # Environment: bridge
└── documentation/           # This documentation
    └── pixi.toml            # Environment: default (MkDocs)
```

**Each module is independent** with its own dependencies and environments.

## Step 4: Configure Pixi (Optional)

### Set Default Python Version

Pixi uses micromamba to manage environments. You can configure default settings:

```bash
# View current configuration
pixi config list

# Set default channels (already configured in pixi.toml files)
pixi config set default-channels conda-forge,pytorch,nvidia

# Set parallel downloads
pixi config set concurrent-downloads 4
```

### Configure Cache Location

By default, Pixi caches packages in `~/.pixi/`. To use a different location:

```bash
# Set custom cache directory
pixi config set cache-dir /path/to/cache

# View cache location
pixi config get cache-dir
```

!!! tip "Large Cache"
    Pixi's cache can grow to several GB. Ensure your home directory or cache location has sufficient space.

## Step 5: Test Pixi Installation

Test that Pixi can create and manage environments:

### Create Test Environment

```bash
# Navigate to a module
cd grid_pose_detection/

# Install environment (this will take a few minutes the first time)
pixi install

# Expected output:
# ✔ Project in /path/to/coral-segmentation/grid_pose_detection is ready to use!
```

### Activate Environment

There are two ways to use Pixi environments:

**Option A: Shell Activation (Interactive)**:
```bash
# Activate environment shell
pixi shell

# Your prompt will change to show environment name
# (grid-pose) user@host:~/coral-segmentation/grid_pose_detection$

# Test Python
python --version
# Expected: Python 3.9.x

# List installed packages
pip list

# Exit shell
exit
```

**Option B: Run Commands Directly (Non-Interactive)**:
```bash
# Run single command in environment
pixi run python --version

# Run with specific environment name
pixi run -e grid-pose python --version

# Run a script
pixi run python src/gridpose_inference.py --help
```

!!! tip "Pixi Shell vs Run"
    Use `pixi shell` for interactive work (Jupyter, debugging). Use `pixi run` for automated scripts and CI/CD pipelines.

## Step 6: Verify Module Environments

Check that Pixi can access all module environments:

```bash
# From repository root
cd ~/Projects/coral-segmentation

# List all environments in current module
cd coral_seg_yolo/
pixi info
# Shows: coral-seg-yolo, coral-seg-yolo-dev

cd ../DINOv2_mmseg/
pixi info
# Shows: dinov2-mmseg

cd ../grid_pose_detection/
pixi info
# Shows: grid-pose, grid-pose-dev

cd ../data_engineering/
pixi info
# Shows: default environment
```

## Pixi Common Commands

### Environment Management

```bash
# Install environment from pixi.toml
pixi install

# Install and activate specific environment
pixi install -e <env-name>

# Update environment dependencies
pixi update

# Clean and reinstall environment
pixi clean
pixi install
```

### Running Commands

```bash
# Run command in default environment
pixi run <command>

# Run command in specific environment
pixi run -e <env-name> <command>

# Run Python script
pixi run python script.py

# Run with environment variables
pixi run --env VAR=value python script.py
```

### Environment Information

```bash
# Show project info and available environments
pixi info

# List installed packages in environment
pixi list

# Show environment's Python path
pixi run which python
```

### Adding Dependencies

```bash
# Add package to default environment
pixi add <package-name>

# Add package to specific environment
pixi add -e <env-name> <package-name>

# Add package with version constraint
pixi add "package-name>=1.0,<2.0"

# Add from specific channel
pixi add --channel conda-forge <package-name>
```

## Understanding pixi.toml Files

Each module has a `pixi.toml` file defining its environments. Here's an example:

```toml
[project]
name = "coral-seg-yolo"
version = "0.1.0"
channels = ["conda-forge", "pytorch", "nvidia"]
platforms = ["linux-64", "osx-arm64"]

[dependencies]
python = "3.9.*"
pytorch = "2.5.0"
pytorch-cuda = "12.1"
ultralytics = "8.3.45"

[feature.dev.dependencies]
jupyter = "*"
fiftyone = "1.0.1"
ipywidgets = "*"

[environments]
default = ["base"]
coral-seg-yolo-dev = ["base", "dev"]
```

**Key Sections**:
- `[project]`: Metadata and channels
- `[dependencies]`: Core dependencies for all environments
- `[feature.X.dependencies]`: Additional dependencies for feature "X"
- `[environments]`: Environment definitions combining features

## Troubleshooting

### Pixi Command Not Found

**Symptoms**: `bash: pixi: command not found`

**Solutions**:
```bash
# Check if pixi is in PATH
echo $PATH | grep -i pixi

# If not found, add to PATH manually
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
pixi --version

# If still not working, reinstall
curl -fsSL https://pixi.sh/install.sh | bash
```

### Installation Fails with SSL Error

**Symptoms**: `SSL certificate verification failed`

**Solutions**:
```bash
# Update CA certificates
sudo apt-get update
sudo apt-get install ca-certificates

# Or bypass SSL verification (not recommended for production)
export PIXI_INSECURE=1
pixi install
unset PIXI_INSECURE
```

### Environment Installation Very Slow

**Symptoms**: `pixi install` takes >30 minutes

**Solutions**:
```bash
# Use multiple concurrent downloads
pixi config set concurrent-downloads 8

# Clear cache and retry
pixi clean
rm -rf ~/.pixi/envs/*
pixi install

# Check network connectivity to conda-forge
ping conda.anaconda.org
```

### Lock File Conflicts

**Symptoms**: `Lock file is incompatible with platform`

**Solutions**:
```bash
# Remove lock file and regenerate
rm pixi.lock
pixi install

# Or update lock file for current platform
pixi update
```

### Permission Denied Errors

**Symptoms**: `Permission denied` when installing packages

**Solutions**:
```bash
# Use user installation (not system-wide)
curl -fsSL https://pixi.sh/install.sh | bash

# Check file permissions
ls -la ~/.pixi/

# Fix ownership if needed
sudo chown -R $USER:$USER ~/.pixi/
```

### Module Import Errors After Installation

**Symptoms**: `ModuleNotFoundError` even after `pixi install`

**Solutions**:
```bash
# Verify you're in the right environment
pixi shell
which python
python -c "import sys; print(sys.prefix)"

# Check if package is actually installed
pixi list | grep <package-name>

# If missing, install it
pixi add <package-name>

# Try clean reinstall
pixi clean
pixi install
```

## Pixi vs Conda/Mamba

If you're familiar with conda, here's how Pixi compares:

| Task | Conda | Pixi |
|------|-------|------|
| Create environment | `conda create -n myenv python=3.9` | `pixi install` (from pixi.toml) |
| Activate environment | `conda activate myenv` | `pixi shell` |
| Install package | `conda install numpy` | `pixi add numpy` |
| Run command | `conda run -n myenv python` | `pixi run python` |
| List packages | `conda list` | `pixi list` |
| Export environment | `conda env export > env.yml` | `pixi.lock` (automatic) |
| Update packages | `conda update --all` | `pixi update` |

**Key Differences**:
- Pixi environments are **per-project**, not global
- Pixi uses **lock files** for reproducibility
- Pixi is **much faster** (written in Rust vs Python)
- Pixi supports **multiple environments per project**

## Best Practices

### Project Organization

1. **One pixi.toml per module**: Don't share environments across modules
2. **Use lock files**: Commit `pixi.lock` to git for reproducibility
3. **Separate environments**: Use `-dev` environments for development dependencies
4. **Pin versions**: Use version constraints in pixi.toml for stability

### Workflow Tips

1. **Always use pixi run**: For scripts and automation
2. **Use pixi shell**: For interactive development and debugging
3. **Keep environments clean**: Don't install packages with pip inside pixi environments
4. **Update regularly**: Run `pixi update` monthly to get bug fixes

### Performance Optimization

1. **Use local cache**: Keep cache on fast SSD
2. **Parallel downloads**: Set `concurrent-downloads` to 4-8
3. **Minimal environments**: Only install what you need
4. **Reuse environments**: Lock files prevent re-downloads

## Next Steps

!!! success "Pixi Setup Complete!"
    Pixi is now installed and ready to manage module environments!

**What's next**:

1. **[Install Module Environments](2-module-environments.md)** - Set up environments for all modules
2. **[Prepare Datasets](3-data-preparation.md)** - Download and configure training data
3. **[Configure GPU](4-gpu-configuration.md)** - Set up CUDA for GPU training

## Quick Reference

### Essential Commands

```bash
# Install environment
pixi install

# Activate shell
pixi shell

# Run command
pixi run <command>

# Add package
pixi add <package>

# Update environment
pixi update

# Show info
pixi info

# List packages
pixi list

# Clean environment
pixi clean
```

### Useful Aliases

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# Pixi shortcuts
alias pi='pixi install'
alias pr='pixi run'
alias ps='pixi shell'
alias pa='pixi add'
alias pu='pixi update'
alias pinfo='pixi info'
```

---

**Questions?** See [module environments guide](2-module-environments.md) or [Getting Help](../../../community/index.md).
