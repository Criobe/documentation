# DINOv2 + MMSeg Segmentation Training

Train high-accuracy semantic segmentation models using DINOv2 foundation models with MMSegmentation framework and CoralSCoP refinement.

## Introduction

This guide covers training **DINOv2-based semantic segmentation models** for coral classification. The DINOv2 + SegFormer combination achieves state-of-the-art accuracy (~49.5% mIoU) on CRIOBE coral datasets, making it ideal for scientific research and applications requiring maximum precision.

### When to Use DINOv2 + MMSeg

**Best for:**

- Low data regime - finetuning with foundation models gives good performance with low data
- Semantic segmentation (class per pixel)
- Two-stage workflows (semantic → instance refinement)
- Applications where speed is secondary to accuracy

**Consider YOLO instead if:**

- Need real-time inference (<10s per image)
- Enough data for fine-tuning
- Faster iteration during development

### What You'll Learn

- Convert polyline annotations to semantic segmentation masks
- Configure DINOv2 + SegFormer training
- Train with multi-scale strategy for better accuracy
- Evaluate semantic segmentation performance
- Two-stage inference: semantic segmentation + CoralSCoP instance refinement
- Deploy DINOv2 models to Nuclio

### Expected Outcomes

- Semantic segmentation model achieving ~49.5% mIoU
- Per-class IoU metrics for all coral genera
- Instance-refined predictions via CoralSCoP + SAM
- Deployment-ready model weights

## Prerequisites

Ensure you have:

- [x] Completed one of the [Data Preparation Guides](../data-preparation/index.md)
- [x] FiftyOne dataset with polyline annotations
- [x] CUDA-capable GPU with 16GB+ VRAM (24GB recommended)
- [x] Pixi installed and configured
- [x] At least 150GB free disk space

!!! warning "GPU Memory Requirements"
    DINOv2 training requires significant VRAM:

    - **Minimum**: 16GB (batch size 2-4, image size 1280)
    - **Recommended**: 24GB (batch size 8, image size 1920)
    - **Optimal**: 40GB+ (batch size 16, multi-scale training)

!!! info "Alternative: Password-Protected Datasets"
    If you have obtained access credentials, you can skip the data preparation workflow by using pre-annotated CVAT project backups:

    - **CRIOBE Archive** (`criobe.7z`): Contains criobe_finegrained_annotated (345 images, 16 genera)
    - **Banggai Archive** (`banggai.7z`): Contains banggai_extended_annotated (126 images, 10 genera)

    These archives contain complete CVAT projects that can be restored directly to your CVAT instance, then pulled to FiftyOne for training.

    **To obtain access**: Email gilles.siu@criobe.pf with your name, institution, and research purpose (academic research only).

    **Setup instructions**: See [Developer Data Preparation Guide](../../setup/installation/for-developers/3-data-preparation.md#step-2-access-ml-datasets-from-cvat-backups) for download and restoration steps.

## Step 1: Environment Setup

### 1.1 Navigate to Module

```bash
cd PROJ_ROOT/criobe/DINOv2_mmseg
```

### 1.2 Activate Pixi Environment

```bash
pixi shell -e dinov2-mmseg
```

The environment includes:

- Python 3.9
- PyTorch 2.0.0 with CUDA 11.7
- MMSegmentation 1.2.2
- DINOv2 models
- FiftyOne for dataset management
- All required dependencies

!!! info "CUDA Version Note"
    This module uses CUDA 11.7 (different from other modules using 12.x). Ensure your GPU drivers support CUDA 11.7+.

### 1.3 Verify Environment

```bash
python -c "import torch; import mmseg; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MMSeg: {mmseg.__version__}')"
```

Expected output:
```
PyTorch: 2.0.0+cu117
CUDA: True
MMSeg: 1.2.2
```

## Step 2: Pull and Prepare Data

### 2.1 Pull from CVAT (if needed)

If you haven't already created the FiftyOne dataset:

```bash
cd PROJ_ROOT/criobe/data_engineering
pixi shell

python create_fiftyone_dataset.py "criobe_finegrained_annotated"
```

### 2.2 Convert to MMSegmentation Format

```bash
cd PROJ_ROOT/criobe/DINOv2_mmseg
pixi shell -e dinov2-mmseg

python prepare_data.py criobe_finegrained_annotated
```

**What this script does:**

1. Loads FiftyOne dataset with polyline annotations
2. Converts polylines to semantic segmentation masks (PNG images)
3. Creates train/val/test splits based on FiftyOne tags
4. Generates MMSeg-compatible directory structure
5. Creates class mapping and color palette

!!! note "Configuration"
    Output directory, taxonomy, and image size are configured via environment variables in the `.env` file, not CLI arguments. The script uses Fire for CLI parsing with a single positional argument (dataset name).

**Expected output:**

```
INFO: Loading FiftyOne dataset: criobe_finegrained_annotated
INFO: Found 450 samples (train: 315, val: 90, test: 45)
INFO: Converting polylines to semantic masks...
INFO: Processing train split: 315 samples
INFO: Created 315 semantic masks (1920x1920)
INFO: Processing val split: 90 samples
INFO: Created 90 semantic masks
INFO: Processing test split: 45 samples
INFO: Created 45 semantic masks
INFO: Dataset prepared at: data/prepared_for_training/criobe_finegrained
```

### 2.3 Verify Output Structure

```bash
tree data/prepared_for_training/criobe_finegrained -L 2
```

Expected structure:

```
data/prepared_for_training/criobe_finegrained/
├── train/
│   ├── data/               # Training images (JPG)
│   └── labels/             # Semantic masks (PNG, single-channel)
├── val/
│   ├── data/               # Validation images (JPG)
│   └── labels/             # Semantic masks (PNG, single-channel)
├── test/
│   ├── data/               # Test images (JPG)
│   └── labels/             # Semantic masks (PNG, single-channel)
├── classes.txt             # Class names (one per line)
├── palette.txt             # RGB color palette for visualization
└── dataset_info.py         # MMSeg dataset configuration
```

### 2.4 Inspect Semantic Masks

```bash
# Check mask values (should be class indices 0-15)
python -c "
import cv2
import numpy as np
mask = cv2.imread('data/prepared_for_training/criobe_finegrained/train/labels/Mangareva_2017_01.png', 0)
print(f'Mask shape: {mask.shape}')
print(f'Unique classes: {np.unique(mask)}')
print(f'Class distribution: {np.bincount(mask.flatten())}')
"
```

Expected:
```
Mask shape: (1920, 1920)
Unique classes: [ 0  1  2  5 10 11 13 14]  # Varies by image
Class distribution: [2845632  156743   23456  ...] # Pixel counts per class
```

## Step 3: Configure Training

MMSegmentation uses hierarchical Python configuration files. Pre-configured training configs are in `DINOv2_mmseg/configs/`.

**Config structure:**

- **Base config** (`dinov2_vitb14_*_ms_config_base.py`): Model, dataset, training schedule, data augmentation
- **Head config** (`dinov2_vitb14_*_ms_config_segformer.py`): Decoder head (extends base config)

**View example configs:**

- [dinov2_vitb14_banggai_ms_config_base.py](https://github.com/criobe-pf/DINOv2_mmseg/blob/main/configs/dinov2_vitb14_banggai_ms_config_base.py)
- [dinov2_vitb14_banggai_ms_config_segformer.py](https://github.com/criobe-pf/DINOv2_mmseg/blob/main/configs/dinov2_vitb14_banggai_ms_config_segformer.py)

**Key parameters to adjust:**

**In base config:**

- `data_root`: Path to prepared dataset (e.g., `./data/coral_annotation_dinov2_mmseg/criobe_finegrained_annotated`)
- `crop_size`: Training crop size (default: 518×518, larger = more memory but better accuracy)
- `batch_size`: Samples per GPU (default: 5, reduce if OOM)
- `num_workers`: Data loading workers (default: 8 train, 4 val)
- `max_epochs`: Training epochs (default: 150)
- `val_interval`: Validation frequency (default: every 20 epochs)
- `lr`: Learning rate (default: 0.001)
- `freeze_vit`: Freeze DINOv2 backbone (default: `True`, set `False` to fine-tune)

**In head config:**

- `num_classes`: Number of coral genera (16 for criobe_finegrained, 11 for banggai_extended)

!!! tip "Batch Size Tuning for VRAM"
    Adjust `batch_size` in the base config based on your GPU memory:

    - **16GB VRAM**: `batch_size=2`, `crop_size=(518, 518)`
    - **24GB VRAM**: `batch_size=4-5`, `crop_size=(518, 518)` or larger
    - **40GB+ VRAM**: `batch_size=8`, `crop_size=(1024, 1024)`

## Step 4: Train DINOv2 + SegFormer

### 4.1 Start Training

```bash
pixi run -e dinov2-mmseg python train.py \
    configs/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
    --work-dir work_dirs/criobe_finegrained_dinov2_segformer
```

**Arguments:**

- `--work-dir`: Output directory for checkpoints and logs
- `--resume`: Resume training from checkpoint (optional)
- `--amp`: Enable Automatic Mixed Precision for faster training (optional)

### 4.2 Monitor Training

Training progress is logged to terminal and TensorBoard:

```
Epoch [1][10/79]  lr: 1.000e-04, eta: 23:45:12, time: 2.345, data_time: 0.234,
loss: 2.456, decode.loss_ce: 2.456, decode.acc_seg: 45.67
Epoch [1][20/79]  lr: 1.000e-04, eta: 23:42:30, time: 2.312, data_time: 0.221,
loss: 2.123, decode.loss_ce: 2.123, decode.acc_seg: 48.91
...

Epoch [10] mIoU: 0.3234, mAcc: 0.5678, aAcc: 0.8234
```

**Key metrics:**

- **loss**: Total training loss (should decrease)
- **decode.loss_ce**: Cross-entropy loss
- **decode.acc_seg**: Segmentation pixel accuracy
- **mIoU**: Mean Intersection over Union (primary metric, aim for >0.45)
- **mAcc**: Mean pixel accuracy per class
- **aAcc**: Overall pixel accuracy

### 4.3 TensorBoard Visualization

```bash
# In separate terminal
pixi run -e dinov2-mmseg tensorboard --logdir work_dirs/criobe_finegrained_dinov2_segformer
```

Open `http://localhost:6006` to view:

- Training and validation loss curves
- mIoU progression over epochs
- Per-class IoU metrics
- Learning rate schedule

## Step 5: Evaluate Semantic Segmentation

### 5.1 Validation Set Evaluation

```bash
pixi run -e dinov2-mmseg python test.py \
    configs/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
    work_dirs/criobe_finegrained_dinov2_segformer/best_mIoU_epoch_140.pth \
    --show-dir work_dirs/criobe_finegrained_dinov2_segformer/val_results
```

!!! note "Evaluation Metrics"
    Evaluation metrics (mIoU, mAcc, etc.) are configured in the config file's `evaluation` section, not via CLI arguments. The test script automatically computes all metrics specified in the config.

**Output:**

```
+----------------+-------+-------+
|     Class      |  IoU  |  Acc  |
+----------------+-------+-------+
| Acanthastrea   | 0.512 | 0.687 |
| Acropora       | 0.623 | 0.789 |
| Astreopora     | 0.487 | 0.612 |
| Atrea          | 0.456 | 0.598 |
| Fungia         | 0.534 | 0.654 |
| Goniastrea     | 0.478 | 0.601 |
| Leptastrea     | 0.445 | 0.589 |
| Merulinidae    | 0.423 | 0.567 |
| Millepora      | 0.501 | 0.645 |
| Montastrea     | 0.412 | 0.553 |
| Montipora      | 0.556 | 0.712 |
| Other          | 0.389 | 0.512 |
| Pavona/Leptoseris | 0.478 | 0.623 |
| Pocillopora    | 0.534 | 0.689 |
| Porites        | 0.598 | 0.745 |
| Psammocora     | 0.434 | 0.578 |
+----------------+-------+-------+
| Mean           | 0.4953| 0.6347|
+----------------+-------+-------+
```

### 5.2 Test Set Evaluation

```bash
pixi run -e dinov2-mmseg python test.py \
    configs/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
    work_dirs/criobe_finegrained_dinov2_segformer/best_mIoU_epoch_140.pth \
    --show-dir work_dirs/criobe_finegrained_dinov2_segformer/test_results
```

!!! tip "Testing on Test Split"
    By default, test.py evaluates on the validation split. To evaluate on the test split, modify the `data.test` section in your config file to point to your test dataset.

### 5.3 Confusion Matrix Analysis

Generate confusion matrix using MMSegmentation's analysis tools:

```bash
pixi run -e dinov2-mmseg mim run mmseg confusion_matrix \
    configs/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
    work_dirs/criobe_finegrained_dinov2_segformer/best_mIoU_epoch_140.pth \
    --show \
    --save-dir work_dirs/criobe_finegrained_dinov2_segformer/confusion
```

Generates confusion matrix showing which classes are most often confused.

## Step 6: Two-Stage Inference (Semantic → Instance)

DINOv2 produces semantic masks (pixel-level classification). For instance segmentation, use the two-stage pipeline with CoralSCoP refinement.

### 6.1 Download CoralSCoP SAM Weights

```bash
cd PROJ_ROOT/criobe/DINOv2_mmseg

# Create assets directory if needed
mkdir -p assets/pretrained_models

# Download CoralSCoP-adapted SAM weights
wget https://zenodo.org/record/XXXX/files/vit_b_coralscop.pth \
    -O assets/pretrained_models/vit_b_coralscop.pth
```

!!! info "CoralSCoP SAM Model"
    CoralSCoP is a fine-tuned Segment Anything Model (SAM) specifically adapted for coral segmentation. It refines semantic segmentation masks into precise instance-level predictions.

### 6.2 Run Two-Stage Inference

```bash
pixi run -e dinov2-mmseg python inference_with_coralscop.py \
    --input-dir data/test_samples/5-grid_removal/ \
    --output-dir results/two_stage_inference/ \
    --segformer-config configs/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
    --segformer-weights work_dirs/criobe_finegrained_dinov2_segformer/best_mIoU_epoch_140.pth \
    --sam-checkpoint assets/pretrained_models/vit_b_coralscop.pth \
    --device cuda:0 \
    --min-threshold 0.05 \
    --unsure-threshold 0.2
```

!!! info "Classification Thresholds"
    - `--min-threshold`: Minimum confidence for a pixel to be classified (default: 0.05)
    - `--unsure-threshold`: Threshold for uncertain classifications passed to SAM (default: 0.2)
    - Output format is controlled automatically - semantic masks are saved in folder mode

**Pipeline stages:**

1. **Semantic segmentation**: DINOv2 + SegFormer produces class probability map
2. **Instance proposal**: CoralSCoP generates instance-level proposals
3. **Refinement**: SAM refines boundaries for each instance
4. **Post-processing**: Non-maximum suppression, filtering

**Output structure:**

```
results/two_stage_inference/
├── visualizations/          # Overlay images with predictions
│   ├── MooreaE2B_2020_05.jpg
│   └── ...
├── masks/                   # Instance masks (PNG)
│   ├── MooreaE2B_2020_05/
│   │   ├── instance_001.png
│   │   ├── instance_002.png
│   │   └── ...
│   └── ...
├── predictions.json         # COCO-format annotations
└── inference_log.txt        # Processing log
```

### 6.3 Inference Time

Typical inference times per 1920x1920 image (RTX 4090):

- **DINOv2 + SegFormer**: ~8-12s
- **CoralSCoP + SAM**: ~5-8s per image
- **Total two-stage**: ~15-25s per image

## Step 7: Export and Deploy

### 7.1 Model Export (Optional)

MMSeg models are saved as PyTorch `.pth` files, which can be used directly for deployment.

!!! note "TorchScript Conversion"
    TorchScript conversion is optional and not commonly needed for this deployment. The Nuclio function uses the `.pth` checkpoint directly with the inference_with_coralscop.py script.

    If you need TorchScript for other deployments, you can use MMSegmentation's conversion tools:
    ```bash
    pixi run -e dinov2-mmseg mim run mmseg pytorch2torchscript \
        configs/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
        --checkpoint work_dirs/criobe_finegrained_dinov2_segformer/best_mIoU_epoch_140.pth \
        --output-file work_dirs/criobe_finegrained_dinov2_segformer/model.pt \
        --shape 1920 1920
    ```

### 7.2 Package Function with Trained Models

Use the **parameterized deployment script** - no manual copying needed:

```bash
cd PROJ_ROOT/criobe/DINOv2_mmseg

# Package function with all required files
./deploy_as_zip.sh coralscopsegformer \
    work_dirs/dinov2_vitb14_coralsegv4_ms_config_segformer/best_mIoU_epoch_140.pth \
    work_dirs/dinov2_vitb14_coralsegv4_ms_config_segformer/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
    assets/pretrained_models/vit_b_coralscop.pth
```

**Script arguments:**
```bash
./deploy_as_zip.sh MODEL_NAME SEGFORMER_WEIGHTS SEGFORMER_CONFIG SAM_WEIGHTS
```

- `MODEL_NAME`: `coralscopsegformer`
- `SEGFORMER_WEIGHTS`: Path to Segformer model weights (.pth file)
- `SEGFORMER_CONFIG`: Path to Segformer config file (.py file)
- `SAM_WEIGHTS`: Path to CoralSCoP/SAM weights (.pth file)

**What happens:**

- Script validates all inputs (model name, weights, config existence)
- Copies Segformer weights as `best_segformer.pth` (standardized name)
- Copies Segformer config as `best_segformer_config.py` (standardized name)
- Copies SAM weights as `vit_b_coralscop.pth` (standardized name)
- Copies source modules: `inferencer.py`, `dataset/`, `models/`, `segment_anything/`
- Creates `nuclio.zip` ready for deployment

### 7.3 Deploy Function to Nuclio

After packaging, deploy using one of these options:

=== "Option 1: CVAT Centralized (Production)"

    ```bash
    # Extract to CVAT's serverless directory
    unzip nuclio.zip -d /path/to/cvat/serverless/pytorch/mmseg/coralscopsegformer/

    # Deploy from CVAT directory
    cd /path/to/cvat
    nuctl deploy --project-name cvat \
        --path ./serverless/pytorch/mmseg/coralscopsegformer/nuclio/ \
        --platform local \
        --verbose
    ```

=== "Option 2: Local Bundle (Development)"

    ```bash
    # Extract to local nuclio_bundles directory
    mkdir -p nuclio_bundles/coralscopsegformer
    unzip nuclio.zip -d nuclio_bundles/coralscopsegformer/

    # Deploy directly from local bundle
    nuctl deploy --project-name cvat \
        --path ./nuclio_bundles/coralscopsegformer/nuclio/ \
        --platform local \
        --verbose
    ```

!!! tip "Deployment Options"
    - **Option 1** is useful when CVAT manages all serverless functions centrally
    - **Option 2** is more flexible for development and testing

### 7.4 Verify Deployment

```bash
# Check function status
nuctl get functions --platform local | grep coralscopsegformer

# View function logs
nuctl get function pth-mmseg-coralscopsegformer --platform local
```

### 7.5 Test Deployed Function

```bash
# Test with sample image
curl -X POST http://localhost:8011 \
    -H "Content-Type: application/json" \
    -d @test_payload.json
```

**Expected response:** JSON array with detected coral polylines and species labels.

### 7.6 Integrate with CVAT

1. Navigate to your coral segmentation project in CVAT
2. **Actions** → **Webhooks** → **Create webhook**
3. Configure:
    - **Target URL**: `http://bridge:8000/detect-model-webhook?model_name=pth-mmseg-coralscopsegformer&conv_mask_to_poly=true`
    - **Events**: Check "When a job state is changed to 'in progress'"
4. Click **Submit**

Now when you open annotation jobs, the two-stage DINOv2 + CoralSCoP model will run automatically!

For detailed deployment instructions, see [Model Deployment Guide](model-deployment.md).

## Troubleshooting

??? question "CUDA out of memory during training"
    **Solutions:**

    1. Reduce batch size: `samples_per_gpu=2` or `1`
    2. Reduce crop size: `crop_size=(1280, 1280)` or `(960, 960)`
    3. Disable multi-scale training (use single scale)
    4. Enable gradient accumulation:
       ```python
       optimizer_config = dict(
           type='GradientCumulativeOptimizerHook',
           cumulative_iters=4  # Accumulate over 4 iterations
       )
       ```
    5. Use gradient checkpointing (saves memory, slower):
       ```python
       model = dict(
           backbone=dict(
               use_checkpoint=True
           )
       )
       ```

??? question "Training loss plateaus or doesn't decrease"
    **Check:**

    1. Learning rate: Try `lr=0.00001` (lower) or `lr=0.001` (higher)
    2. Dataset normalization: Ensure mean/std match DINOv2 pretraining
    3. Frozen layers: Try unfreezing more backbone layers
    4. Data quality: Review masks in FiftyOne app

    **Debug:**
    ```bash
    # Visualize training samples using MMSeg tools
    pixi run -e dinov2-mmseg mim run mmseg browse_dataset \
        configs/dinov2_vitb14_coralsegv4_ms_config_segformer.py \
        --show-interval 1
    ```

??? question "Low mIoU on validation set"
    **Possible causes:**

    1. **Insufficient training**: Try 200-240 epochs
    2. **Class imbalance**: Weight loss by class frequency
       ```python
       loss_decode=dict(
           type='CrossEntropyLoss',
           use_sigmoid=False,
           class_weight=[1.0, 2.0, 1.5, ...]  # Weight rare classes higher
       )
       ```
    3. **Annotation errors**: Review dataset for incorrect masks
    4. **Backbone frozen**: Unfreeze more layers for fine-tuning

??? question "CoralSCoP produces too many/too few instances"
    **Tune thresholds:**

    ```python
    instances = sam_model.segment_instances(
        image_np,
        semantic_mask,
        score_threshold=0.3,  # Lower for more instances
        nms_threshold=0.5,    # Higher for more overlap tolerance
        min_area=100          # Minimum instance size (pixels)
    )
    ```

## Advanced Topics

### Multi-Scale Testing

For better accuracy, use multi-scale testing:

```python
test_cfg = dict(
    mode='slide',
    crop_size=(1280, 1280),
    stride=(853, 853)  # 2/3 overlap
)
```

### Hierarchical Taxonomy Training

Train on coarse taxonomy, then fine-tune on fine-grained:

```bash
# Step 1: Train on main families (7 classes)
python tools/train.py configs/dinov2_main_families.py

# Step 2: Fine-tune on finegrained (16 classes)
python tools/train.py configs/dinov2_finegrained.py \
    --load-from work_dirs/dinov2_main_families/best_mIoU_epoch_100.pth
```

### Ensemble with YOLO

Combine semantic (DINOv2) and instance (YOLO) predictions:

```python
from ensemble_utils import merge_predictions

yolo_preds = load_yolo_predictions('yolo_results.json')
mmseg_preds = load_mmseg_predictions('mmseg_results.json')

merged = merge_predictions(
    yolo_preds,
    mmseg_preds,
    strategy='weighted',  # or 'voting', 'nms'
    yolo_weight=0.4,
    mmseg_weight=0.6
)
```

## Next Steps

Congratulations! You've trained a high-accuracy DINOv2 semantic segmentation model. Next:

- **Compare with YOLO**: [YOLO Segmentation Guide](yolo-segmentation.md)
- **Deploy to production**: [Model Deployment Guide](model-deployment.md)
- **Train grid detection**: [Grid Detection Guide](grid-detection.md)

## Reference

### Module Documentation

- [DINOv2_mmseg/README.md](https://github.com/taiamiti/criobe/DINOv2_mmseg/README.md)
- [DINOv2_mmseg/configs/](https://github.com/taiamiti/criobe/DINOv2_mmseg/configs/)

### External Resources

- [MMSegmentation Documentation](https://mmsegmentation.readthedocs.io/)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [SegFormer Paper](https://arxiv.org/abs/2105.15203)
- [CoralSCoP Paper](https://arxiv.org/abs/XXXX.XXXXX)

---

**Related Guides**: [YOLO Training](yolo-segmentation.md) · [Model Deployment](model-deployment.md) · [Back to Overview](index.md)
