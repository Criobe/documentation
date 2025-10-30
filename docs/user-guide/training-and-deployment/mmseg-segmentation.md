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

python create_fiftyone_dataset.py \
    --cvat-project-name "criobe_finegrained_annotated" \
    --dataset-name "criobe_finegrained_fo"
```

### 2.2 Convert to MMSegmentation Format

```bash
cd PROJ_ROOT/criobe/DINOv2_mmseg
pixi shell -e dinov2-mmseg

python prepare_data.py criobe_finegrained_fo
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
INFO: Loading FiftyOne dataset: criobe_finegrained_fo
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
├── images/
│   ├── train/              # Training images (JPG)
│   ├── val/                # Validation images
│   └── test/               # Test images
├── annotations/
│   ├── train/              # Semantic masks (PNG, single-channel)
│   ├── val/
│   └── test/
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
mask = cv2.imread('data/prepared_for_training/criobe_finegrained/annotations/train/sample_001.png', 0)
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

### 3.1 Understanding MMSeg Configs

MMSegmentation uses hierarchical configuration files:

```
configs/
├── _base_/
│   ├── models/           # Model architectures
│   ├── datasets/         # Dataset configs
│   ├── schedules/        # Training schedules
│   └── default_runtime.py
└── dinov2_vitb14_coralsegv4_ms_config_segformer.py  # Main config
```

### 3.2 Edit Main Training Config

```bash
nano configs/dinov2_vitb14_coralsegv4_ms_config_segformer.py
```

Key sections to configure:

```python
_base_ = [
    'mmseg::_base_/models/segformer_mit-b0.py',
    'mmseg::_base_/datasets/ade20k.py',
    'mmseg::_base_/default_runtime.py',
    'mmseg::_base_/schedules/schedule_160k.py'
]

# Model configuration
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='mmseg.VisionTransformer',
        img_size=(1920, 1920),
        patch_size=14,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=[2, 5, 8, 11],  # Multi-level features
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        with_cls_token=True,
        frozen_stages=-1,  # -1 = train all, 0+ = freeze first N stages
        interpolate_mode='bicubic',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='facebook/dinov2-base',  # DINOv2 pretrained weights
            prefix='backbone.'
        )
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[768, 768, 768, 768],  # Match backbone out_indices
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=16,  # Adjust for your taxonomy
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')  # 'whole' or 'slide' for large images
)

# Dataset configuration
dataset_type = 'CustomDataset'
data_root = 'data/prepared_for_training/criobe_finegrained'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],  # ImageNet normalization
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1920, 1920), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),  # Color augmentation
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(1920, 1920), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

# Multi-scale training for better accuracy
train_pipeline_multiscale = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        ratio_range=(0.5, 2.0),  # Random scale
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=(1280, 1280)),  # Random crop
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(1280, 1280), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

# Training schedule
optimizer = dict(
    type='AdamW',
    lr=0.0001,  # Initial learning rate
    betas=(0.9, 0.999),
    weight_decay=0.01
)

optimizer_config = dict(
    type='GradientCumulativeOptimizerHook',
    cumulative_iters=2  # Gradient accumulation for large images
)

lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-6,
    by_epoch=False
)

runner = dict(type='EpochBasedRunner', max_epochs=160)
checkpoint_config = dict(by_epoch=True, interval=10)  # Save every 10 epochs
evaluation = dict(interval=10, metric='mIoU', pre_eval=True, save_best='mIoU')

# Data loaders
data = dict(
    samples_per_gpu=4,  # Batch size per GPU (adjust for VRAM)
    workers_per_gpu=4,  # Data loading workers
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='annotations/train',
        pipeline=train_pipeline_multiscale  # Use multi-scale
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='annotations/val',
        pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='annotations/test',
        pipeline=test_pipeline
    )
)
```

!!! tip "Batch Size Tuning for VRAM"
    - **16GB VRAM**: `samples_per_gpu=2`, `crop_size=(1280, 1280)`
    - **24GB VRAM**: `samples_per_gpu=4`, `crop_size=(1920, 1920)`
    - **40GB+ VRAM**: `samples_per_gpu=8`, full multi-scale

### 3.3 Optional: Freeze Backbone Layers

For faster training or limited data:

```python
backbone=dict(
    ...
    frozen_stages=6,  # Freeze first 6 layers, train last 6
    ...
)
```

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

### 4.4 Training Duration

Typical training times (160 epochs, 450 images, RTX 4090):

- **DINOv2-base + SegFormer**: ~24-30 hours
- **Each epoch**: ~8-10 minutes

**Checkpointing:** Best model (highest mIoU) automatically saved to `work_dirs/*/best_mIoU_*.pth`

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
