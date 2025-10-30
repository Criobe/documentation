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

- **Demo dataset testing**: ~5 minutes (test with 5 samples first to verify setup)
- **Data preparation**: ~1 hour (ML datasets for training)
- **GridCorners training**: 4-6 hours
- **GridPose training**: 4-6 hours
- **Evaluation**: ~1 hour
- **Deployment**: ~30 minutes

!!! tip "Start with Demo Datasets"
    Before training on full ML datasets, test your setup with demo datasets (5 samples) to verify your environment is correctly configured. This saves time if there are setup issues.

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

python create_fiftyone_dataset.py "criobe_corner_annotation"

# Or for Banggai
python create_fiftyone_dataset.py "banggai_corner_detection"
```

!!! note "FiftyOne Dataset Naming"
    The script uses Fire CLI with a single positional argument (project name). The FiftyOne dataset will be created with the same name as the CVAT project.

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

python create_fiftyone_dataset.py "criobe_grid_annotation"
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

python src/prepare_data.py gridcorners
```

!!! note "Hardcoded Configuration"
    The `prepare_data.py` script uses Fire CLI with task name as the only argument (`gridcorners` or `gridpose`). Dataset names and output paths are hardcoded in the `config.py` file.

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
python src/prepare_data.py gridpose
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

## Step 4: Train Models

!!! info "Training Implementation"
    The grid_pose_detection module uses a specialized training script (`reproduce_exp_pose_as_det_with_mapping.py`) that treats keypoint detection as a detection problem with template matching. Training parameters are hardcoded in the script rather than using external config files.

### 4.1 Train GridCorners Model

```bash
cd PROJ_ROOT/criobe/grid_pose_detection
pixi shell -e grid-pose

python src/reproduce_exp_pose_as_det_with_mapping.py gridcorners
```

**What this does:**

1. Trains YOLO11n detection model (not pose model) to detect grid corners
2. Uses template matching to refine detected points to exact 4-corner configuration
3. Outputs trained model to `runs/gridcorners/detect/train14/weights/best.pt`
4. Creates corner template at `assets/kp_template_corners.npy`

**Training parameters** (hardcoded in script):
- Model: yolo11n.pt (detection, not pose)
- Epochs: 70
- Batch size: 16
- Image size: 1920
- Data augmentation: Conservative (no flips, limited rotation)

### 4.2 Train GridPose Model

```bash
python src/reproduce_exp_pose_as_det_with_mapping.py gridpose
```

**What this does:**

1. Trains YOLO11n detection model to detect all 117 grid intersection points
2. Uses template matching to ensure correct point ordering
3. Outputs trained model to `runs/gridpose/detect/train6/weights/best.pt`
4. Creates grid template at `assets/kp_template_gridpose.npy`

**Training parameters** (hardcoded in script):
- Model: yolo11n.pt
- Epochs: 70
- Batch size: 8
- Image size: 1920

!!! tip "Customizing Training"
    To modify training parameters (epochs, batch size, etc.), edit the `reproduce_exp_pose_as_det_with_mapping.py` script directly. Look for the `run_gridcorners_experiment()` and `run_gridpose_experiment()` functions.

### 4.3 Monitor Training

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

### 4.4 Training Duration

**Expected training times:**
- **GridCorners**: ~2-3 hours on RTX 4090 (70 epochs)
- **GridPose**: ~4-6 hours on RTX 4090 (70 epochs)

**Note:** Training completes automatically and includes evaluation on validation set.

## Step 5: Evaluate Models

!!! note "Automatic Evaluation"
    The training script (`reproduce_exp_pose_as_det_with_mapping.py`) automatically evaluates models using FiftyOne integration. Results are stored in the FiftyOne dataset and can be viewed interactively.

### 5.1 View Results in FiftyOne

```bash
pixi run -e grid-pose-dev fiftyone app launch criobe_corner_annotation
# or
pixi run -e grid-pose-dev fiftyone app launch criobe_grid_annotation
```

Use FiftyOne to:
- Inspect prediction quality visually
- Filter by confidence scores
- Identify problem cases
- Compare predictions vs ground truth

### 5.2 Key Metrics

The training script evaluates both detection accuracy and template matching success:

**Detection Metrics:**
- **box_loss**: Bounding box localization (should converge to <0.5)
- **cls_loss**: Classification loss
- **Detection mAP**: Detection accuracy before template matching

**Template Matching Metrics:**
- **Matching success rate**: Percentage of images where all points matched to template (aim for >95%)
- **Mean point error**: Average pixel distance from ground truth (aim for <3 pixels)
- **Max point error**: Worst-case pixel error (should be <10 pixels)

These metrics are computed automatically during training and stored in FiftyOne for interactive exploration.

## Step 6: Inference on New Images

### 6.1 GridCorners Inference

```bash
pixi run -e grid-pose python src/gridpose_inference.py predict_as_yolo_txt \
    --test_dir=data/test_samples/1-raw_jpg/ \
    --model_path=runs/gridcorners/detect/train14/weights/best.pt \
    --template_path=assets/kp_template_corners.npy \
    --output_dir=results/corner_predictions \
    --label_name=quadrat_corner \
    --max_cost=0.3
```

**Output:**

- YOLO TXT format keypoint files
- Template-matched corner coordinates (4 ordered points)
- Optional debug visualizations

### 6.2 GridPose Inference

```bash
pixi run -e grid-pose python src/gridpose_inference.py predict_as_yolo_txt \
    --test_dir=data/test_samples/3-image_warping/ \
    --model_path=runs/gridpose/detect/train6/weights/best.pt \
    --template_path=assets/kp_template_gridpose.npy \
    --output_dir=results/grid_predictions \
    --label_name=grid \
    --max_cost=0.12
```

**Output:**

- YOLO TXT format keypoint files (117 ordered points)
- Template-matched grid coordinates
- Optional debug visualizations

### 6.3 Parameters

- `--test_dir`: Input directory with images
- `--model_path`: Trained detection model
- `--template_path`: Keypoint template for matching
- `--output_dir`: Where to save predictions
- `--label_name`: Label for COCO export (`quadrat_corner` or `grid`)
- `--max_cost`: Maximum matching cost threshold (lower = stricter matching)
- `--debug`: Optional debug output directory for visualizations
- `--conf`: Detection confidence threshold (default: 0.1)

### 6.4 Export to COCO Format

For integration with warping and grid removal:

```bash
pixi run -e grid-pose python src/gridpose_inference.py predict_as_yolo_txt \
    --test-dir data/test_samples/1-raw_jpg/ \
    --model-path runs/pose/gridcorners_yolo11n/weights/best.pt \
    --template-path assets/kp_template_corners.npy \
    --output-dir results/corners_yolo_format
```

Creates `.txt` files compatible with YOLO format for downstream processing.

## Step 7: Deploy to Nuclio

### 7.1 Deploy GridCorners Function

```bash
cd PROJ_ROOT/criobe/grid_pose_detection

# Copy trained model to deployment directory
cp runs/gridcorners/detect/train14/weights/best.pt deploy/gridcorners/nuclio/model_weights.pt

# Package and deploy
./deploy_gridcorners_as_zip.sh

nuctl deploy --project-name cvat \
    --path ./deploy/gridcorners/nuclio \
    --platform local \
    --verbose
```

### 7.2 Deploy GridPose Function

```bash
cd PROJ_ROOT/criobe/grid_pose_detection

# Copy trained model to deployment directory
cp runs/gridpose/detect/train6/weights/best.pt deploy/gridpose/nuclio/model_weights.pt

# Package and deploy
./deploy_gridpose_as_zip.sh

nuctl deploy --project-name cvat \
    --path ./deploy/gridpose/nuclio \
    --platform local \
    --verbose
```

### 7.3 Test Deployed Functions

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

### 7.4 Integrate with CVAT

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
