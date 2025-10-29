# Pixi Environment Setup

Install and configure Pixi package manager for managing Python environments across all QUADRATSEG modules developed by CRIOBE.

!!! tip "Why Pixi?"
    - **Isolated environments** per module with correct Python versions
    - **Automatic CUDA management** - handles CUDA 11.x and 12.x separately
    - **Fast dependency resolution** with conda-forge and PyPI
    - **Reproducible** across machines
    - **No manual Python installation** required

**Time Required**: 10-15 minutes
**Prerequisites**: Linux, macOS, or WSL2 on Windows

## What is Pixi?

[Pixi](https://pixi.sh/) is a modern package manager that creates isolated, reproducible Python environments. Each QUADRATSEG module maintained by CRIOBE (e.g., `coral_seg_yolo`, `grid_pose_detection`) has its own `pixi.toml` configuration defining dependencies and Python versions.

**Key Features**:
- Manages Python interpreters automatically
- Handles CUDA toolkit versions per module
- Supports multiple environments per project (e.g., `dev` vs `prod`)
- Uses conda, PyPI, and custom wheels simultaneously

## Step 1: Install Pixi

### Linux / macOS

```bash
# Download and install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Reload shell configuration
source ~/.bashrc  # or ~/.zshrc for zsh

# Verify installation
pixi --version
# Expected: pixi 0.30.0 or newer
```

### Windows (WSL2)

Use the same Linux installation method inside WSL2:

```bash
# From WSL2 terminal
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
pixi --version
```

!!! warning "Native Windows Support"
    While Pixi supports native Windows, QUADRATSEG is designed for Linux. **Use WSL2 with Ubuntu 22.04** for best results.

### Alternative: Manual Installation

If the install script fails:

```bash
# Download specific version
wget https://github.com/prefix-dev/pixi/releases/download/v0.30.0/pixi-x86_64-unknown-linux-musl

# Make executable
chmod +x pixi-x86_64-unknown-linux-musl

# Move to PATH
sudo mv pixi-x86_64-unknown-linux-musl /usr/local/bin/pixi

# Verify
pixi --version
```

## Step 2: Verify System Requirements

Before installing module environments, ensure system requirements are met:

```bash
# Check available disk space (need 10-20 GB for all environments)
df -h ~

# Check CUDA is available (optional, for GPU support)
nvidia-smi

# Expected: CUDA version 11.7+ or 12.x
```

!!! info "CPU-Only Mode"
    You can use Pixi environments without a GPU. Module installation will work, but inference will be 10-50x slower.

## Step 3: Clone Repository

Clone the QUADRATSEG repository maintained by CRIOBE:

```bash
# Clone repository
cd ~/
git clone https://github.com/criobe/coral-segmentation.git
cd coral-segmentation

# List modules
ls -d */
# Expected: bridge/ coral_seg_yolo/ DINOv2_mmseg/ grid_pose_detection/ grid_inpainting/ data_engineering/ documentation/
```

## Step 4: Install Module Environments

Each module has its own Pixi environment. Install them as needed:

### Coral Segmentation (YOLO)

```bash
cd coral_seg_yolo

# Install core environment (CUDA 12.1, PyTorch 2.5.0)
pixi install -e coral-seg-yolo

# OR install dev environment (includes Jupyter, FiftyOne, TensorBoard)
pixi install -e coral-seg-yolo-dev

# Verify installation
pixi list -e coral-seg-yolo-dev
```

**Expected packages**:
- Python 3.9
- PyTorch 2.5.0 (CUDA 12.1)
- Ultralytics (YOLOv11)
- FiftyOne 1.8.0 (dev env only)

**Environments**:
- `coral-seg-yolo`: Core training/inference
- `coral-seg-yolo-dev`: Development with Jupyter and FiftyOne

### Grid Pose Detection

```bash
cd ../grid_pose_detection

# Install dev environment
pixi install -e grid-pose-dev

# Verify
pixi list -e grid-pose-dev
```

**Expected packages**:
- Python 3.9
- PyTorch 2.5.0 (CUDA 12.1)
- Ultralytics (YOLO for keypoint detection)

**Environments**:
- `grid-pose`: Core environment
- `grid-pose-dev`: Development environment

### Grid Inpainting

```bash
cd ../grid_inpainting

# Install environment
pixi install

# Verify
pixi list
```

**Expected packages**:
- Python 3.9+
- PyTorch with CUDA 12.x
- SimpleLama for inpainting

### DINOv2 Segmentation

```bash
cd ../DINOv2_mmseg

# Install environment (CUDA 11.7, PyTorch 2.0.0)
pixi install -e dinov2-mmseg

# Verify
pixi list -e dinov2-mmseg
```

!!! warning "CUDA 11.7 Required"
    DINOv2 uses CUDA 11.7 (older than other modules). Pixi isolates this environment so it won't conflict with CUDA 12.x modules.

**Expected packages**:
- Python 3.9
- PyTorch 2.0.0 (CUDA 11.7)
- MMSegmentation
- DINOv2 (editable install from `src/dinov2/`)

**Environment**:
- `dinov2-mmseg`: Single environment for all tasks

### Data Engineering

```bash
cd ../data_engineering

# Install environment
pixi install

# Verify
pixi list
```

**Expected packages**:
- Python 3.9
- FiftyOne for dataset management
- PyTorch for model evaluation

## Step 5: Test Environments

Verify each environment works correctly:

### Test YOLO Environment

```bash
cd ~/coral-segmentation/coral_seg_yolo

# Activate shell
pixi shell -e coral-seg-yolo-dev

# Test PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Expected output:
# PyTorch: 2.5.0+cu121
# CUDA available: True

# Test YOLO
python -c "from ultralytics import YOLO; print('YOLO imported successfully')"

# Exit shell
exit
```

### Test DINOv2 Environment

```bash
cd ~/coral-segmentation/DINOv2_mmseg

# Activate shell
pixi shell -e dinov2-mmseg

# Test PyTorch version
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

# Expected output:
# PyTorch: 2.0.0+cu117
# CUDA: 11.7

# Test MMSegmentation
python -c "import mmseg; print(f'MMSeg version: {mmseg.__version__}')"

# Exit shell
exit
```

## Using Pixi Environments

There are two main ways to use Pixi environments:

### Method 1: Run Commands Directly

**Best for**: Quick commands, scripts, one-off tasks

```bash
# Run command in specific environment
pixi run -e <environment-name> <command>

# Examples:
pixi run -e coral-seg-yolo-dev python src/inference_demo.py
pixi run -e dinov2-mmseg python train.py configs/my_config.py
pixi run -e grid-pose-dev jupyter notebook
```

### Method 2: Activate Shell

**Best for**: Interactive work, multiple commands, debugging

```bash
# Activate environment shell
pixi shell -e <environment-name>

# Now in activated environment
python script.py
jupyter notebook
# ... run multiple commands ...

# Exit when done
exit
```

### Environment Name Reference

| Module | Environment Names | Use Case |
|--------|-------------------|----------|
| `coral_seg_yolo` | `coral-seg-yolo`<br/>`coral-seg-yolo-dev` | Coral segmentation (YOLO) |
| `grid_pose_detection` | `grid-pose`<br/>`grid-pose-dev` | Grid detection |
| `grid_inpainting` | (default) | Grid removal |
| `DINOv2_mmseg` | `dinov2-mmseg` | Semantic segmentation |
| `data_engineering` | (default) | Dataset management |
| `documentation` | (default) | Documentation build |

!!! tip "Default Environment"
    If a module has no `-e` flag specified, it uses the default environment defined in `pixi.toml`.

## Understanding Pixi Configuration

Each module's `pixi.toml` defines its environment. Here's an example breakdown:

```toml
[project]
name = "coral_seg_yolo"
version = "0.1.0"
platforms = ["linux-64"]
channels = ["conda-forge", "pytorch", "nvidia"]

[dependencies]
python = "3.9.*"                    # Python version

[system-requirements]
cuda = "12"                         # CUDA version requirement

[pypi-dependencies]
torch = { version = "==2.5.0", index = "https://download.pytorch.org/whl/cu121" }
ultralytics = "*"                  # Latest from PyPI

[feature.dev.pypi-dependencies]    # Dev-only dependencies
jupyter = "*"
fiftyone = "==1.8.0"

[environments]
coral-seg-yolo = { solve-group = "default" }
coral-seg-yolo-dev = { features = ["dev"], solve-group = "default" }
```

**Key sections**:
- `[dependencies]`: Core packages (conda-forge)
- `[pypi-dependencies]`: Python packages from PyPI or custom indexes
- `[feature.*.pypi-dependencies]`: Optional feature sets
- `[environments]`: Named environments combining features

## Advanced Configuration

### Environment Variables

Some modules use `.env` files for configuration. Create them after installing:

```bash
# Coral YOLO module
cd coral_seg_yolo
cp .env.example .env
nano .env  # Edit CVAT credentials, paths

# DINOv2 module
cd ../DINOv2_mmseg
cp .env.example .env
nano .env  # Edit paths, CVAT credentials
```

See [Environment Variables](../configuration/environment-variables.md) for details.

### Cache Location

Pixi stores environments in `~/.pixi/`:

```bash
# View cache size
du -sh ~/.pixi

# Typical size: 5-15 GB for all modules
```

### Cleaning Cache

Remove unused environments to free space:

```bash
# Clean unused packages
pixi clean cache

# Remove specific environment (from module directory)
cd coral_seg_yolo
pixi clean -e coral-seg-yolo

# Remove all environments in current project
pixi clean
```

## Troubleshooting

### Environment Installation Fails

**Error**: `Failed to solve dependencies`

**Solution**:
```bash
# Try with verbose output
pixi install -e <env-name> -v

# Common issue: conflicting CUDA versions
# Solution: Ensure correct CUDA version in system-requirements
```

**Error**: `Package not found`

**Solution**:
```bash
# Update Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clear cache and retry
pixi clean cache
pixi install -e <env-name>
```

### CUDA Not Available in Environment

**Check**:
```bash
pixi shell -e coral-seg-yolo-dev
python -c "import torch; print(torch.cuda.is_available())"
```

**If False**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA requirement in `pixi.toml`: `[system-requirements] cuda = "12"`
3. Reinstall environment:
   ```bash
   pixi clean -e coral-seg-yolo-dev
   pixi install -e coral-seg-yolo-dev
   ```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'fiftyone'`

**Solution**: Wrong environment or missing dev features
```bash
# Ensure using dev environment
pixi run -e coral-seg-yolo-dev python script.py

# Or install dev environment if not done
pixi install -e coral-seg-yolo-dev
```

### Slow Installation

**Issue**: Conda dependency resolution is slow

**Solution**:
```bash
# Use mamba solver (faster)
pixi global install pixi-mamba
pixi config set dependency-solver mamba

# Retry installation
pixi install -e <env-name>
```

### Out of Disk Space

**Check space**:
```bash
df -h ~
du -sh ~/.pixi
```

**Free space**:
```bash
# Remove all cached packages
pixi clean cache

# Remove specific project environments
cd <module>
pixi clean

# Remove unused Docker images (if applicable)
docker system prune -a
```

## Multiple CUDA Versions

The QUADRATSEG platform uses different CUDA versions across modules. Pixi handles this automatically:

| Module | CUDA Version | PyTorch Version | Isolated? |
|--------|--------------|-----------------|-----------|
| `coral_seg_yolo` | 12.1 | 2.5.0 | ✅ |
| `grid_pose_detection` | 12.x | 2.5.0 | ✅ |
| `grid_inpainting` | 12.1 | Latest | ✅ |
| `DINOv2_mmseg` | 11.7 | 2.0.0 | ✅ |

**How it works**:
- Each environment downloads its own CUDA libraries
- No system-wide CUDA installation conflicts
- Only requires NVIDIA driver (compatible with all CUDA versions)

**Verify isolation**:
```bash
# Check CUDA in YOLO env
pixi run -e coral-seg-yolo-dev python -c "import torch; print(torch.version.cuda)"
# Output: 12.1

# Check CUDA in DINOv2 env
pixi run -e dinov2-mmseg python -c "import torch; print(torch.version.cuda)"
# Output: 11.7
```

## IDE Integration

### VSCode

Install Pixi extension for VSCode:

1. Open VSCode
2. Install **Pixi** extension
3. Open module directory (e.g., `coral_seg_yolo/`)
4. VSCode will detect `pixi.toml` and offer to select environment

**Manual Python interpreter selection**:
1. `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose interpreter from `.pixi/envs/<env-name>/bin/python`

### PyCharm

Pixi includes `pixi-pycharm` dependency for integration:

1. Open PyCharm
2. File → Settings → Project → Python Interpreter
3. Add Interpreter → Existing Environment
4. Select: `<project>/.pixi/envs/<env-name>/bin/python`

## Best Practices

### ✅ Do:
- Install `-dev` environments for development (includes Jupyter, FiftyOne)
- Use `pixi run -e <env>` for automation scripts
- Use `pixi shell -e <env>` for interactive work
- Update Pixi regularly: `curl -fsSL https://pixi.sh/install.sh | bash`

### ❌ Don't:
- Mix `pip install` in Pixi environments (use `pixi add` instead)
- Manually install Python (Pixi manages this)
- Share `pixi.lock` files across different platforms
- Use `sudo` with Pixi commands

## Next Steps

!!! success "Pixi Environments Ready!"
    You can now run Python scripts in any module using Pixi.

**Continue with**:

- [ML Models Installation](ml-models.md) - Download pre-trained models
- [CVAT Projects Configuration](../configuration/cvat-projects.md) - Set up annotation projects
- [First Annotation Tutorial](../../quickstart/first-annotation.md) - Test the complete pipeline

## Quick Reference

### Common Commands

```bash
# Install environment
pixi install -e <env-name>

# Run command
pixi run -e <env-name> <command>

# Activate shell
pixi shell -e <env-name>

# List environments
pixi project info

# List packages in environment
pixi list -e <env-name>

# Add package to environment
pixi add -e <env-name> <package>

# Clean cache
pixi clean cache

# Update Pixi
curl -fsSL https://pixi.sh/install.sh | bash
```

### Environment Quick Reference

```bash
# YOLO Coral Segmentation
cd coral_seg_yolo
pixi run -e coral-seg-yolo-dev python src/inference_demo.py ...

# Grid Pose Detection
cd grid_pose_detection
pixi run -e grid-pose-dev python src/gridpose_inference.py ...

# Grid Inpainting
cd grid_inpainting
pixi run python grid_rem_with_kp.py ...

# DINOv2 Segmentation
cd DINOv2_mmseg
pixi run -e dinov2-mmseg python inference_with_coralscop.py ...
```

---

**Questions?** See [Getting Help](../../community/getting-help.md) or [Pixi Documentation](https://pixi.sh/latest/).
