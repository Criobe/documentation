# Grid Detection Training

Train YOLOv11 keypoint detection models for quadrat corner and grid intersection detection.

## Introduction

This guide covers training **YOLOv11-pose** models for detecting keypoints in underwater quadrat images. The `grid_pose_detection` module handles two distinct tasks:

1. **GridCorners**: Detect 4 quadrat corner points for perspective correction
2. **GridPose**: Detect 117 grid intersection points for grid removal

### When to Train Grid Detection Models

**Train GridCorners when:**

- Using a custom quadrat frame design different from CRIOBE/Banggai datasets
- Corner markers have different appearance (color, shape, material)
- Working with different underwater conditions (turbidity, lighting)
- Pre-trained model has low accuracy on your specific setup

**Train GridPose when:**

- Using a different grid size (not 9×13 = 117 points)
- Grid material or design differs significantly
- Working with irregular or adaptive grid patterns
- Need higher precision for your specific grid type

### What You'll Learn

- Convert skeleton annotations to YOLO keypoint format
- Configure separate training for corners and grid pose
- Train YOLOv11-pose models with template matching
- Evaluate keypoint detection accuracy
- Deploy corner and grid detection models to Nuclio

### Expected Outcomes

- GridCorners model detecting 4 corners with <5 pixel error
- GridPose model detecting 117 points with <3 pixel error
- Template matching ensuring correct point ordering
- Deployment-ready models for preprocessing pipeline

### Time Required

- **Data preparation**: ~1 hour
- **GridCorners training**: 4-6 hours
- **GridPose training**: 4-6 hours
- **Evaluation**: ~1 hour
- **Deployment**: ~30 minutes

## Prerequisites

Ensure you have:

- [x] Completed [Guide B](../data-preparation/2-two-stage-banggai.md) or [Guide C](../data-preparation/3-three-stage-criobe.md)
- [x] FiftyOne datasets with skeleton annotations:
    - Corner dataset (4-point skeletons)
    - Grid dataset (117-point skeletons)
- [x] CUDA-capable GPU with 8GB+ VRAM
- [x] Pixi installed and configured
- [x] At least 50GB free disk space

!!! info "Annotation Requirements"
    Grid detection requires precise keypoint annotations:

    - **Corners**: Exactly 4 points in clockwise order (TL, TR, BR, BL)
    - **Grid**: Exactly 117 points in top-left to bottom-right order
    - **Precision**: Points should be within 2-3 pixels of actual intersections

## Step 1: Environment Setup

### 1.1 Navigate to Module

```bash
cd PROJ_ROOT/criobe/grid_pose_detection
```

### 1.2 Activate Pixi Environment

```bash
# For training only
pixi shell -e grid-pose

# For training + evaluation with FiftyOne
pixi shell -e grid-pose-dev
```

The environment includes:

- Python 3.9
- PyTorch 2.5.0 with CUDA 12.1
- Ultralytics YOLO with pose support
- NumPy, SciPy for template matching
- FiftyOne (dev environment)

### 1.3 Verify GPU

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Step 2: Pull Data from CVAT

### 2.1 Pull Corner Detection Dataset

```bash
cd PROJ_ROOT/criobe/data_engineering
pixi shell

python create_fiftyone_dataset.py \
    --cvat-project-name "criobe_corner_annotation" \
    --dataset-name "criobe_corners_fo"

# Or for Banggai
python create_fiftyone_dataset.py \
    --cvat-project-name "banggai_corner_detection" \
    --dataset-name "banggai_corners_fo"
```

**Verify corner dataset:**

```bash
fiftyone app launch criobe_corners_fo
```

Check:

- Each image has exactly 4 keypoints
- Keypoints form a quadrilateral (quadrat corners)
- Train/val/test splits are assigned

### 2.2 Pull Grid Detection Dataset

```bash
cd PROJ_ROOT/criobe/data_engineering
pixi shell

python create_fiftyone_dataset.py \
    --cvat-project-name "criobe_grid_annotation" \
    --dataset-name "criobe_grid_fo"
```

**Verify grid dataset:**

```bash
fiftyone app launch criobe_grid_fo
```

Check:

- Each image has exactly 117 keypoints
- Keypoints form a regular 9×13 grid
- Points are ordered correctly (top-left to bottom-right)

## Step 3: Prepare YOLO Keypoint Format

### 3.1 Prepare GridCorners Data

```bash
cd PROJ_ROOT/criobe/grid_pose_detection
pixi shell -e grid-pose-dev

python src/prepare_data.py \
    --task gridcorners \
    --dataset criobe_corners_fo \
    --output-dir data/prepared_for_training/gridcorners
```

**What this does:**

1. Loads FiftyOne dataset with skeleton annotations
2. Converts 4-point skeletons to YOLO keypoint format
3. Creates COCO-style keypoint labels with visibility flags
4. Generates `dataset.yaml` for YOLO training
5. Creates keypoint template for template matching

**Expected output:**

```
INFO: Loading FiftyOne dataset: criobe_corners_fo
INFO: Found 200 samples (train: 140, val: 40, test: 20)
INFO: Converting 4-point skeletons to YOLO format...
INFO: Processing train split: 140 samples
INFO: Processing val split: 40 samples
INFO: Processing test split: 20 samples
INFO: Created keypoint template: assets/kp_template_corners.npy
INFO: Dataset prepared at: data/prepared_for_training/gridcorners
```

**Output structure:**

```
data/prepared_for_training/gridcorners/
├── dataset.yaml
├── train/
│   ├── images/
│   └── labels/    # YOLO keypoint format
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 3.2 Prepare GridPose Data

```bash
python src/prepare_data.py \
    --task gridpose \
    --dataset criobe_grid_fo \
    --output-dir data/prepared_for_training/gridpose
```

**Output:**

```
INFO: Loading FiftyOne dataset: criobe_grid_fo
INFO: Found 180 samples (train: 126, val: 36, test: 18)
INFO: Converting 117-point skeletons to YOLO format...
INFO: Created keypoint template: assets/kp_template_gridpose.npy
INFO: Dataset prepared at: data/prepared_for_training/gridpose
```

### 3.3 Understand YOLO Keypoint Format

**Label file format** (`.txt`):

```
class_id x_center y_center width height kp1_x kp1_y kp1_vis kp2_x kp2_y kp2_vis ... kpN_x kpN_y kpN_vis
```

**Example for GridCorners (4 keypoints):**

```
0 0.500 0.500 0.980 0.980 0.123 0.145 2 0.876 0.143 2 0.879 0.857 2 0.121 0.859 2
```

- `class_id`: Always 0 (single class "corner" or "grid_point")
- `x_center, y_center, width, height`: Bounding box (normalized 0-1)
- `kpX_x, kpX_y`: Keypoint coordinates (normalized 0-1)
- `kpX_vis`: Visibility flag (0=not labeled, 1=occluded, 2=visible)

### 3.4 Inspect Keypoint Template

Templates ensure correct keypoint ordering via Hungarian matching:

```bash
python -c "
import numpy as np
template = np.load('assets/kp_template_corners.npy')
print('Corner template shape:', template.shape)  # (4, 2)
print('Template coordinates (normalized):')
print(template)
"
```

**GridCorners template** (approximate):

```
[[0.05, 0.05],   # Top-left
 [0.95, 0.05],   # Top-right
 [0.95, 0.95],   # Bottom-right
 [0.05, 0.95]]   # Bottom-left
```

**GridPose template** (117 points, 9×13 grid):

```python
# Regular grid from (0.05, 0.05) to (0.95, 0.95)
# Spacing: ~0.1125 horizontal, ~0.075 vertical
```

## Step 4: Configure Training

### 4.1 GridCorners Training Config

Edit `experiments/train_cfg_gridcorners.yaml`:

```yaml
# Model
model: yolo11n-pose.pt  # Nano model (fast, sufficient for 4 points)

# Dataset
data: data/prepared_for_training/gridcorners/dataset.yaml

# Hyperparameters
epochs: 100
batch: 16              # Adjust for GPU
imgsz: 1920            # Match quadrat image size
kpt_shape: [4, 3]      # 4 keypoints, 3 values each (x, y, visibility)

# Optimizer
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.937
weight_decay: 0.0005

# Data augmentation (conservative for keypoints)
degrees: 5.0           # Small rotation (corners should stay in frame)
translate: 0.05        # Small translation
scale: 0.2             # Modest scaling
shear: 0.0             # No shear (distorts corners)
perspective: 0.0001    # Minimal perspective
flipud: 0.0            # Don't flip vertically (breaks corner order)
fliplr: 0.0            # Don't flip horizontally (breaks corner order)
mosaic: 0.0            # Don't use mosaic (confuses corner detection)

# Keypoint-specific
pose: 1.0              # Keypoint loss weight
kobj: 2.0              # Keypoint objectness weight

# Advanced
amp: true              # Mixed precision
patience: 20           # Early stopping patience
save_period: 10        # Save checkpoint every 10 epochs
```

!!! warning "Augmentation for Keypoint Detection"
    Keypoint detection requires **conservative augmentation**:

    - No horizontal/vertical flips (breaks point ordering)
    - Limited rotation/translation (points must stay visible)
    - No mosaic (creates impossible geometries)

### 4.2 GridPose Training Config

Edit `experiments/train_cfg_gridpose.yaml`:

```yaml
model: yolo11n-pose.pt  # Can use yolo11s-pose for more accuracy

data: data/prepared_for_training/gridpose/dataset.yaml

epochs: 100
batch: 8               # Larger model, reduce batch size
imgsz: 1920
kpt_shape: [117, 3]    # 117 keypoints!

optimizer: AdamW
lr0: 0.001
lrf: 0.01

# Very conservative augmentation (117 points must stay ordered)
degrees: 2.0
translate: 0.02
scale: 0.1
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.0
mosaic: 0.0

pose: 1.0
kobj: 2.0

amp: true
patience: 25
```

!!! tip "Model Size Selection"
    - **GridCorners (4 points)**: yolo11n-pose is sufficient (~2M params)
    - **GridPose (117 points)**: yolo11s-pose or yolo11m-pose recommended (~9-22M params)

## Step 5: Train GridCorners Model

### 5.1 Start Training

```bash
pixi run -e grid-pose yolo pose train \
    cfg=experiments/train_cfg_gridcorners.yaml \
    project=runs/pose \
    name=gridcorners_yolo11n
```

**Or with explicit parameters:**

```bash
pixi run -e grid-pose yolo pose train \
    model=yolo11n-pose.pt \
    data=data/prepared_for_training/gridcorners/dataset.yaml \
    epochs=100 \
    imgsz=1920 \
    batch=16 \
    kpt_shape="[4, 3]" \
    project=runs/pose \
    name=gridcorners_yolo11n
```

### 5.2 Monitor Training

```
Epoch    GPU_mem   box_loss   pose_loss   kobj_loss   cls_loss  Instances       Size
  1/100      4.2G      0.845      0.234       0.123      0.567          1       1920

         Class     Images  Instances    Box(P    R    mAP50    mAP50-95)    Pose(P    R    mAP50    mAP50-95)
           all         40         40    0.923    0.875    0.912    0.723    0.891    0.834    0.878    0.645
```

**Key metrics:**

- **pose_loss**: Keypoint localization loss (should decrease to ~0.05-0.1)
- **kobj_loss**: Keypoint objectness loss
- **Pose(mAP50)**: Mean Average Precision for keypoints (aim for >0.85)

### 5.3 Training Duration

- **GridCorners (yolo11n-pose, 200 images)**: ~2-3 hours on RTX 4090
- **Each epoch**: ~1-2 minutes

## Step 6: Train GridPose Model

### 6.1 Start Training

```bash
pixi run -e grid-pose yolo pose train \
    cfg=experiments/train_cfg_gridpose.yaml \
    project=runs/pose \
    name=gridpose_yolo11n
```

### 6.2 Monitor Training

GridPose training is more challenging (117 points vs 4):

```
Epoch    GPU_mem   box_loss   pose_loss   kobj_loss   cls_loss  Instances       Size
  1/100      6.8G      0.912      0.456       0.234      0.678          1       1920

         Class     Images  Instances    Pose(P    R    mAP50    mAP50-95)
           all         36         36    0.812    0.756    0.789    0.512
```

**Aim for:**

- **Pose(mAP50)**: >0.75 (acceptable for grid detection)
- **pose_loss**: <0.1 (indicates good localization)

### 6.3 Training Duration

- **GridPose (yolo11n-pose, 180 images, 117 keypoints)**: ~4-6 hours on RTX 4090
- **Each epoch**: ~3-4 minutes

## Step 7: Evaluate Models

### 7.1 GridCorners Validation

```bash
pixi run -e grid-pose yolo pose val \
    model=runs/pose/gridcorners_yolo11n/weights/best.pt \
    data=data/prepared_for_training/gridcorners/dataset.yaml \
    split=test \
    save_json=true
```

**Output:**

```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)     Pose(P          R      mAP50  mAP50-95)
                   all         20         20      0.945      0.900      0.935      0.756      0.923      0.875      0.912      0.698
```

### 7.2 GridPose Validation

```bash
pixi run -e grid-pose yolo pose val \
    model=runs/pose/gridpose_yolo11n/weights/best.pt \
    data=data/prepared_for_training/gridpose/dataset.yaml \
    split=test \
    save_json=true
```

### 7.3 Template Matching Evaluation

The critical test is whether detected points can be correctly ordered using template matching:

```bash
pixi run -e grid-pose-dev python src/evaluate_with_template.py \
    --model runs/pose/gridpose_yolo11n/weights/best.pt \
    --test-dir data/prepared_for_training/gridpose/test/images \
    --template assets/kp_template_gridpose.npy \
    --output-dir results/gridpose_eval
```

**Metrics:**

- **Matching success rate**: Percentage of images where all points matched correctly (aim for >95%)
- **Mean Euclidean distance**: Average pixel error per keypoint (aim for <3 pixels)
- **Max point error**: Worst-case error (should be <10 pixels)

**Example output:**

```
Template Matching Evaluation Results:
======================================
Total images: 18
Successful matches: 18 (100.0%)
Failed matches: 0 (0.0%)

Distance Metrics:
-----------------
Mean Euclidean distance: 2.34 pixels
Median Euclidean distance: 2.12 pixels
Std Euclidean distance: 0.87 pixels
Max point error: 6.78 pixels
Min point error: 0.45 pixels

Per-Point Statistics:
---------------------
Point 0 (top-left corner): mean=1.89px, max=4.23px
Point 1: mean=2.12px, max=5.67px
...
Point 116 (bottom-right corner): mean=2.45px, max=6.78px
```

## Step 8: Inference and Testing

### 8.1 Single Image Prediction (GridCorners)

```bash
pixi run -e grid-pose python src/gridpose_inference.py predict_corners \
    --test-dir data/test_samples/1-raw_jpg/ \
    --model-path runs/pose/gridcorners_yolo11n/weights/best.pt \
    --template-path assets/kp_template_corners.npy \
    --output-dir results/corner_predictions \
    --visualize
```

**Output:**

- Detected corner coordinates
- Corner visualization overlaid on images
- Ordered points (TL, TR, BR, BL)

### 8.2 Batch Prediction (GridPose)

```bash
pixi run -e grid-pose python src/gridpose_inference.py predict_grid \
    --test-dir data/test_samples/3-image_warping/ \
    --model-path runs/pose/gridpose_yolo11n/weights/best.pt \
    --template-path assets/kp_template_gridpose.npy \
    --output-dir results/grid_predictions \
    --save-coco-json
```

**Output:**

- 117 grid points per image
- COCO keypoint format JSON
- Visualizations with grid overlay

### 8.3 Export to YOLO TXT Format

For integration with warping and grid removal:

```bash
pixi run -e grid-pose python src/gridpose_inference.py predict_as_yolo_txt \
    --test-dir data/test_samples/1-raw_jpg/ \
    --model-path runs/pose/gridcorners_yolo11n/weights/best.pt \
    --template-path assets/kp_template_corners.npy \
    --output-dir results/corners_yolo_format
```

Creates `.txt` files compatible with YOLO format for downstream processing.

## Step 9: Deploy to Nuclio

### 9.1 Deploy GridCorners Function

```bash
cd PROJ_ROOT/criobe/grid_pose_detection/deploy/pth-yolo-gridcorners

# Copy trained model
cp ../../runs/pose/gridcorners_yolo11n/weights/best.pt model_weights.pt

# Package and deploy
./deploy_as_zip.sh

nuctl deploy --project-name cvat \
    --path ./nuclio \
    --platform local \
    --verbose
```

### 9.2 Deploy GridPose Function

```bash
cd PROJ_ROOT/criobe/grid_pose_detection/deploy/pth-yolo-gridpose

# Copy trained model
cp ../../runs/pose/gridpose_yolo11n/weights/best.pt model_weights.pt

# Package and deploy
./deploy_as_zip.sh

nuctl deploy --project-name cvat \
    --path ./nuclio \
    --platform local \
    --verbose
```

### 9.3 Test Deployed Functions

**Test GridCorners:**

```bash
curl -X POST http://localhost:8001 \
    -H "Content-Type: application/json" \
    -d @test_payload_corners.json
```

**Test GridPose:**

```bash
curl -X POST http://localhost:8002 \
    -H "Content-Type: application/json" \
    -d @test_payload_grid.json
```

### 9.4 Integrate with CVAT

**Configure GridCorners webhook:**

1. Navigate to corner detection project in CVAT
2. **Actions** → **Webhooks** → **Create webhook**
3. Configure:
    - **URL**: `http://bridge:8000/detect-model-webhook?model_name=pth-yolo-gridcorners&conv_mask_to_poly=false`
    - **Events**: "When a job state is changed to 'in progress'"

**Configure GridPose webhook:**

1. Navigate to grid detection project in CVAT
2. **Actions** → **Webhooks** → **Create webhook**
3. Configure:
    - **URL**: `http://bridge:8000/detect-model-webhook?model_name=pth-yolo-gridpose&conv_mask_to_poly=false`
    - **Events**: "When a job state is changed to 'in progress'"

For complete deployment, see [Model Deployment Guide](model-deployment.md).

## Troubleshooting

??? question "Model detects wrong number of keypoints"
    **GridCorners detecting ≠4 points:**

    - **Too many**: Model may detect non-corner points (rocks, fish, etc.)
        - Solution: Increase confidence threshold, improve training data quality
    - **Too few**: Model missing some corners
        - Solution: More training data, especially for occluded/difficult corners

    **GridPose detecting ≠117 points:**

    - Template matching should handle missing/extra points
    - Review predictions with `--visualize` flag
    - Check if grid is actually visible in all images

??? question "Template matching fails frequently"
    **Causes:**

    - Keypoints detected in wrong order
    - Significant perspective distortion
    - Non-regular grid in images

    **Solutions:**

    1. Retrain with more diverse data
    2. Update template to match your specific grid:
       ```python
       # Create custom template
       import numpy as np
       template = create_grid_template(rows=9, cols=13, spacing_x=0.1125, spacing_y=0.075)
       np.save('assets/kp_template_custom.npy', template)
       ```
    3. Adjust matching threshold in inference script

??? question "High keypoint localization error"
    **Check:**

    - Image resolution sufficient (1920x1920 recommended)
    - Annotations are precise (review in CVAT)
    - Augmentation not too aggressive
    - Sufficient training epochs

    **Improve:**

    1. Use larger model: yolo11s-pose or yolo11m-pose
    2. Train longer: 150-200 epochs
    3. Fine-tune learning rate: try lr0=0.0005
    4. Add more training data with difficult cases

??? question "Model works on training data but fails on new images"
    **Overfitting symptoms:**

    - High training accuracy, low validation accuracy
    - Works on specific sites but fails on others

    **Solutions:**

    - Collect more diverse training data (different sites, conditions)
    - Increase augmentation slightly
    - Use smaller model to reduce overfitting
    - Implement early stopping

## Advanced Topics

### Custom Grid Sizes

For different grid patterns (e.g., 7×7 = 49 points):

1. **Create custom template:**

   ```python
   import numpy as np

   def create_grid_template(rows, cols, margin=0.05):
       points = []
       x_spacing = (1.0 - 2*margin) / (cols - 1)
       y_spacing = (1.0 - 2*margin) / (rows - 1)

       for row in range(rows):
           for col in range(cols):
               x = margin + col * x_spacing
               y = margin + row * y_spacing
               points.append([x, y])

       return np.array(points, dtype=np.float32)

   template_7x7 = create_grid_template(rows=7, cols=7)
   np.save('assets/kp_template_7x7.npy', template_7x7)
   ```

2. **Update config:**

   ```yaml
   kpt_shape: [49, 3]  # 7x7 grid
   ```

### Multi-Task Learning

Train a single model for both corners and grid detection:

```python
# Requires custom dataset with variable keypoint counts
# Not recommended for production (separate models are clearer)
```

### Keypoint Heatmap Refinement

For sub-pixel accuracy, use heatmap-based refinement:

```python
from grid_pose_detection.refinement import refine_keypoints_heatmap

refined_points = refine_keypoints_heatmap(
    image,
    rough_keypoints,
    window_size=15
)
```

## Next Steps

Congratulations! You've trained grid detection models. Next:

- **Train grid removal**: [Grid Removal Guide](grid-removal.md)
- **Deploy all models**: [Model Deployment Guide](model-deployment.md)
- **Complete pipeline**: [Three-Stage CRIOBE Setup](../data-preparation/3-three-stage-criobe.md)

## Reference

### Module Documentation

- [grid_pose_detection/README.md](https://github.com/taiamiti/criobe/grid_pose_detection/README.md)
- [grid_pose_detection/src/](https://github.com/taiamiti/criobe/grid_pose_detection/src/)

### External Resources

- [YOLO Pose Documentation](https://docs.ultralytics.com/tasks/pose/)
- [Hungarian Algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm)
- [COCO Keypoint Format](https://cocodataset.org/#format-data)

---

**Related Guides**: [Grid Removal](grid-removal.md) · [Model Deployment](model-deployment.md) · [Back to Overview](index.md)
