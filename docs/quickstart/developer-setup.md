# Developer Setup

Set up the QUADRATSEG development environment for **AI researchers and developers** who want to train models, experiment with architectures, and iterate on the ML pipeline.

!!! info "For AI Researchers & Developers"
    This setup installs **Pixi** for managing Python environments across all modules. Use this path if you want to train models, run experiments, or develop new features.

**Time Required**: 30-45 minutes
**Target Users**: AI researchers, ML engineers, software developers

!!! tip "What You'll Accomplish"
    - Install Pixi package manager for environment management
    - Set up development environments for all pipeline modules
    - Download test data and pre-trained models
    - Run inference and training experiments locally
    - Understand the codebase structure for development

**Prerequisites**: Git, Pixi installed, Python 3.9+, NVIDIA GPU recommended

## Repository Architecture

QUADRATSEG uses a **multi-repository architecture** where each module is an independent Git repository:

- **Clone only what you need**: Work on specific modules without the full codebase
- **Independent versioning**: Each module has its own release cycle
- **Shared data via symlinks**: Centralized data directory avoids duplication
- **Flexible development**: Mix and match modules for your workflow

!!! tip "Modular Development"
    You don't need to clone all repositories. Start with the modules relevant to your work (e.g., just `coral_seg_yolo` for YOLO segmentation).

## Why Use Pixi for Development?

Pixi is a fast, cross-platform package manager that handles all dependencies for each module:

- **Isolated Environments**: Each module repository has its own environment
- **Reproducible Builds**: Exact versions specified in `pixi.toml` files
- **Fast Installation**: Parallel downloads and caching
- **CUDA Management**: Handles PyTorch + CUDA dependencies automatically
- **Cross-Platform**: Works on Linux, macOS, and Windows

!!! warning "Production vs Development"
    **End users** (coral researchers) do NOT need Pixi or module repositories. The production system uses Docker with pre-packaged models.

    **Developers** need Pixi and individual module repositories to train, evaluate, and experiment with models before packaging them for deployment.

## Development Workflow Overview

This setup enables the complete development workflow:

```mermaid
graph LR
    A[1. Raw Image] --> B[2. Corner<br/>Detection]
    B --> C[3. Warped<br/>Image]
    C --> D[4. Grid<br/>Detection]
    D --> E[5. Grid<br/>Removal]
    E --> F[6. Coral<br/>Segmentation]

    style A fill:#e1f5ff
    style F fill:#c8e6c9
```

## Prerequisites Check

Before starting, ensure you have:

### Required Software

- [x] **Git**: For version control ([install](https://git-scm.com/downloads))
- [x] **Pixi**: Package manager ([install guide](https://pixi.sh/latest/#installation))
- [x] **Python 3.9+**: Managed by Pixi, but system Python helpful for Pixi itself
- [x] **NVIDIA GPU**: Recommended for training and fast inference
- [x] **NVIDIA Drivers**: CUDA 11.7+ or 12.x depending on module
- [x] **Disk Space**: 50GB+ for datasets, models, and experiments

### Install Pixi

If you haven't installed Pixi yet:

**Linux & macOS**:
```bash
# Install via curl
curl -fsSL https://pixi.sh/install.sh | bash

# Restart terminal or source the config
source ~/.bashrc  # or ~/.zshrc for zsh

# Verify installation
pixi --version
```

**Windows**:
```powershell
# Install via PowerShell
iwr -useb https://pixi.sh/install.ps1 | iex

# Verify installation
pixi --version
```

### Verify GPU (Optional but Recommended)

```bash
# Check NVIDIA driver
nvidia-smi

# Expected: GPU information with CUDA version
# If error, install NVIDIA drivers from nvidia.com
```

## Step 1: Set Up Centralized Data Directory

Create a shared data directory that all modules will access via symlinks:

```bash
# Create centralized data directory
mkdir -p ~/Projects/criobe_data

# Set environment variable (add to ~/.bashrc for persistence)
export DATA_ROOT=~/Projects/criobe_data
echo 'export DATA_ROOT=~/Projects/criobe_data' >> ~/.bashrc
```

!!! info "Why Centralized Data?"
    This approach allows all modules to share test data, models, and datasets without duplication. Each module will have a `data` symlink pointing to `$DATA_ROOT`.

## Step 2: Clone Module Repositories

Clone only the modules you need. Here we'll clone the most common ones:

```bash
# Navigate to projects directory
cd ~/Projects

# Clone ML module repositories (work on main branch by default)
git clone https://github.com/criobe/coral_seg_yolo.git
git clone https://github.com/criobe/grid_pose_detection.git
git clone https://github.com/criobe/grid_inpainting.git
git clone https://github.com/criobe/data_engineering.git

# Optional: Clone additional modules as needed
# git clone https://github.com/criobe/DINOv2_mmseg.git
```

**All module repositories work on the `main` branch** (the default).

!!! tip "Choose Your Modules"
    You don't need all modules. Common combinations:

    - **Coral segmentation only**: `coral_seg_yolo`, `data_engineering`
    - **Grid detection only**: `grid_pose_detection`, `grid_inpainting`
    - **Full pipeline**: All modules above

## Step 3: Create Data Symlinks

Link each module to the centralized data directory:

```bash
# Create symlink in each module
cd ~/Projects/coral_seg_yolo
ln -s $DATA_ROOT data

cd ~/Projects/grid_pose_detection
ln -s $DATA_ROOT data

cd ~/Projects/grid_inpainting
ln -s $DATA_ROOT data

cd ~/Projects/data_engineering
ln -s $DATA_ROOT data

# Verify symlinks
ls -l ~/Projects/coral_seg_yolo/data
# Should show: data -> /home/user/Projects/criobe_data
```

## Step 4: Download Test Data

Download shared test data once to the centralized directory:

```bash
# Option 1: Use the automated script (recommended)
cd ~/Projects/coral_seg_yolo  # or grid_pose_detection, grid_inpainting, DINOv2_mmseg
./download_test_samples.sh

# Option 2: Manual download
cd $DATA_ROOT

# Download demo datasets (contains 5 sample images for all pipeline stages)
wget https://storage.googleapis.com/data_criobe/test_samples.zip
unzip test_samples.zip

# Verify structure
ls -la test_samples/
# Should show: 1-raw_jpg/, 2-quadrat_corner_export/, etc.
```

**All modules now have access to demo datasets via their `data` symlinks!**

The demo datasets include subdirectories for each pipeline stage (5 sample images each):

| Directory | Stage | Content |
|-----------|-------|---------|
| `test_samples/1-raw_jpg/` | Raw images | Original underwater quadrat photos |
| `test_samples/2-quadrat_corner_export/` | Corner annotations | Corner keypoint labels |
| `test_samples/3-image_warping/` | Warped images | Perspective-corrected quadrats |
| `test_samples/4-grid_pose_export/` | Grid annotations | Grid intersection keypoints |
| `test_samples/5-grid_removal/` | Clean images | Grid-removed coral images |

## Demo A: Complete YOLO Pipeline (Fastest)

Run the fastest end-to-end demo using YOLOv11 segmentation.

### Setup

```bash
# Navigate to module repository
cd ~/Projects/coral_seg_yolo

# Install environment
pixi install -e coral-seg-yolo-dev

# Download models to shared data directory
./download_models.sh  # Models will be in $DATA_ROOT/models/
```

### Run Inference

```bash
# Run coral segmentation on grid-removed images
pixi run -e coral-seg-yolo-dev python src/inference_demo.py \
    data/test_samples/5-grid_removal/ \
    results/demo_yolo/ \
    models/coralsegv4_yolo11m_best.pt \
    --method overlapping
```

**Expected Output**:
```
Loading model: models/coralsegv4_yolo11m_best.pt
Processing images from: data/test_samples/5-grid_removal/
Found 10 images
Processing: Tetiaroa_1994_01.jpg
  └─ Detected 58 coral instances in 7.4s
Processing: Tetiaroa_1994_02.jpg
  └─ Detected 42 coral instances in 6.8s
...
✅ Results saved to: results/demo_yolo/
```

**View Results**:
```bash
# Results include:
ls results/demo_yolo/
# - annotated_images/  (images with drawn polygons)
# - predictions.json    (raw predictions)
# - summary.txt        (statistics)
```

**Performance**: ~7s per image (1920x1920px on GTX 1070 Mobile)

## Demo B: Grid Detection & Removal

See how grid patterns are detected and removed.

### Grid Corner Detection (4 points)

```bash
# Navigate to grid detection module
cd ~/Projects/grid_pose_detection

pixi install
./download_models.sh

# Detect 4 corner points on raw images
pixi run python src/gridpose_inference.py predict_as_yolo_txt \
    --test_dir data/test_samples/1-raw_jpg \
    --model_path assets/gridcorners_yolov11n_best.pt \
    --template_path assets/kp_template_corners.npy \
    --output_dir results/demo_corners \
    --max_cost 0.3
```

**Expected Output**:
```
Loading model and template...
Processing 10 images...
✅ Tetiaroa_1994_01.jpg: 4/4 corners detected (cost: 0.12)
✅ Tetiaroa_1994_02.jpg: 4/4 corners detected (cost: 0.15)
...
Results saved to: results/demo_corners/
```

### Grid Pose Detection (117 points)

```bash
# Detect all 117 grid intersection points on warped images
pixi run python src/gridpose_inference.py predict_as_yolo_txt \
    --test_dir data/test_samples/3-image_warping \
    --model_path assets/gridpose_yolov11n_best.pt \
    --template_path assets/kp_template.npy \
    --output_dir results/demo_gridpose \
    --max_cost 0.12
```

**Expected Output**:
```
Processing 10 images...
✅ Tetiaroa_1994_01.jpg: 117/117 keypoints detected (cost: 0.08)
...
```

### Grid Removal (Inpainting)

```bash
# Navigate to grid inpainting module
cd ~/Projects/grid_inpainting

pixi install
./download_model.sh

# Remove grid lines using detected keypoints
pixi run python grid_rem_with_kp.py remove_grid_from_coco_dataset \
    --data_path data/test_samples/3-image_warping/ \
    --labels_path data/test_samples/4-grid_pose_export/person_keypoints_default.json \
    --output_dir results/demo_grid_removal
```

**Expected Output**:
```
Loading SimpleLama model...
Processing batch 1/1 (10 images)...
  ✓ Tetiaroa_1994_01.jpg inpainted (5.2s)
  ✓ Tetiaroa_1994_02.jpg inpainted (5.4s)
...
✅ Grid removed from 10 images
Results: results/demo_grid_removal/
```

**Performance**: ~5-8s per image

## Demo C: DINOv2 Two-Stage Segmentation (Highest Accuracy)

Run the most accurate segmentation approach.

### Setup

```bash
# Clone DINOv2 module if not already done
cd ~/Projects
git clone https://github.com/criobe/DINOv2_mmseg.git
cd DINOv2_mmseg

# Create data symlink
ln -s $DATA_ROOT data

pixi install -e dinov2-mmseg
./download_models.sh
```

### Run Two-Stage Inference

```bash
# Semantic segmentation + instance refinement
pixi run -e dinov2-mmseg python inference_with_coralscop.py \
    --input-dir data/test_samples/5-grid_removal/ \
    --segformer-config assets/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
    --segformer-weights assets/best_mIoU_epoch_140.pth \
    --sam-checkpoint assets/pretrained_models/vit_b_coralscop.pth \
    --output-dir results/demo_dinov2/
```

**Expected Output**:
```
Loading DINOv2 SegFormer model...
Loading CoralSCoP (SAM) model...
Processing: Tetiaroa_1994_01.jpg
  Stage 1: Semantic segmentation (8.2s)
  Stage 2: Instance refinement (12.4s)
  └─ Generated 56 coral instances
...
✅ Results saved to: results/demo_dinov2/
```

**Performance**: ~15-25s per image
**Accuracy**: 49.53% mIoU (highest accuracy)