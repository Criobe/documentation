# Training Configuration

Configure training experiments, hyperparameters, and monitoring for QUADRATSEG model development.

!!! info "For Developers"
    This guide configures **training experiments** with proper hyperparameters, data augmentation, and monitoring for training coral segmentation and grid detection models.

**Time Required**: 30-45 minutes
**Prerequisites**: [Datasets prepared](../../installation/for-developers/3-data-preparation.md), [GPU configured](../../installation/for-developers/4-gpu-configuration.md)

## Configuration Overview

Each module uses different configuration systems:

| Module | Config System | Location | Format |
|--------|--------------|----------|--------|
| **coral_seg_yolo** | YOLO YAML | `experiments/*.yaml` | YAML |
| **DINOv2_mmseg** | MMSegmentation | `configs/*.py` | Python |
| **grid_pose_detection** | YOLO YAML | `experiments/*.yaml` | YAML |
| **grid_inpainting** | Python args | Command line | Args |

## Step 1: Understand Configuration Structure

### YOLO Configuration (coral_seg_yolo, grid_pose_detection)

YOLO uses YAML files with hierarchical structure:

```yaml
# experiments/train_config.yaml
# Dataset configuration
data: data/prepared_yolo/criobe_finegrained/dataset.yaml

# Model configuration
model: yolo11m-seg.yaml
task: segment

# Training hyperparameters
epochs: 100
batch: 8
imgsz: 1920
device: 0

# Optimizer
optimizer: AdamW
lr0: 0.001
lrf: 0.01
momentum: 0.9
weight_decay: 0.0005

# Data augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
fliplr: 0.5
mosaic: 1.0
```

### MMSegmentation Configuration (DINOv2_mmseg)

MMSegmentation uses Python config files:

```python
# configs/train_config.py
_base_ = [
    './_base_/models/dinov2_segformer.py',
    './_base_/datasets/criobe_finegrained.py',
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_150e.py'
]

# Model configuration
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='DINOv2',
        model_name='dinov2_vitb14',
        pretrained='assets/pretrained_models/dinov2_vitb14_pretrain.pth'
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[96, 192, 384, 768],
        num_classes=18
    )
)

# Training configuration
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
lr_config = dict(policy='poly', power=0.9, min_lr=1e-6)

# Runtime configuration
runner = dict(type='EpochBasedRunner', max_epochs=150)
checkpoint_config = dict(interval=10)
```

## Step 2: Configure coral_seg_yolo Training

### Create Training Configuration

```bash
cd ~/Projects/coral-segmentation/coral_seg_yolo

# Create experiment directory
mkdir -p experiments

# Create configuration file
nano experiments/criobe_finegrained_train.yaml
```

### Complete YOLO Configuration Example

```yaml
# experiments/criobe_finegrained_train.yaml
# =============================================================================
# Experiment Configuration: CRIOBE Finegrained Coral Segmentation
# =============================================================================

# Dataset
data: data/prepared_yolo/criobe_finegrained/dataset.yaml

# Model
model: yolo11m-seg.yaml  # n, s, m, l, x
task: segment

# Training Settings
epochs: 100
batch: 8                 # Adjust based on GPU memory
imgsz: 1920             # Image size (square)
device: 0               # GPU ID or [0,1] for multi-GPU
workers: 4              # DataLoader workers

# Mixed Precision Training
amp: true               # Automatic Mixed Precision (faster, less memory)

# Optimizer
optimizer: AdamW
lr0: 0.001              # Initial learning rate
lrf: 0.01               # Final learning rate (lr0 * lrf)
momentum: 0.937
weight_decay: 0.0005

# Scheduler
warmup_epochs: 3.0
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Loss Weights
box: 7.5                # Box loss weight
cls: 0.5                # Classification loss weight
dfl: 1.5                # DFL loss weight (distribution focal loss)

# Data Augmentation
hsv_h: 0.015            # HSV hue augmentation
hsv_s: 0.7              # HSV saturation
hsv_v: 0.4              # HSV value
degrees: 0.0            # Rotation (disabled for underwater)
translate: 0.1          # Translation
scale: 0.5              # Scale
shear: 0.0              # Shear (disabled)
perspective: 0.0        # Perspective (disabled)
flipud: 0.0             # Vertical flip (disabled)
fliplr: 0.5             # Horizontal flip
mosaic: 1.0             # Mosaic augmentation
mixup: 0.0              # Mixup (disabled)
copy_paste: 0.0         # Copy-paste (disabled for segmentation)

# Validation Settings
val: true
save_json: true         # Save COCO JSON for evaluation
plots: true             # Save training plots
save_period: 10         # Save checkpoint every N epochs

# Output
project: runs/train
name: criobe_finegrained_yolo11m
exist_ok: false         # Don't overwrite existing experiment
```

### Training Configuration Variants

**Fast Training (Testing)**:
```yaml
# experiments/criobe_finegrained_fast.yaml
epochs: 10
batch: 16
imgsz: 1280             # Smaller images
model: yolo11n-seg.yaml # Smaller model
mosaic: 0.0             # Disable expensive augmentations
```

**High Accuracy (Production)**:
```yaml
# experiments/criobe_finegrained_high_acc.yaml
epochs: 200
batch: 4
imgsz: 1920
model: yolo11l-seg.yaml  # Larger model
optimizer: SGD
lr0: 0.01
```

### Start Training

```bash
cd ~/Projects/coral-segmentation/coral_seg_yolo

# Train with configuration file
pixi run python src/training/train.py \
    --config experiments/criobe_finegrained_train.yaml

# Or use YOLO CLI
pixi run yolo segment train \
    data=data/prepared_yolo/criobe_finegrained/dataset.yaml \
    model=yolo11m-seg.yaml \
    epochs=100 \
    imgsz=1920 \
    batch=8
```

## Step 3: Configure DINOv2_mmseg Training

### Create Base Configurations

MMSegmentation uses modular configs split into multiple files:

```bash
cd ~/Projects/coral-segmentation/DINOv2_mmseg

# Create config directory structure
mkdir -p configs/_base_/{models,datasets,schedules,default_runtime}
```

### Model Configuration

```python
# configs/_base_/models/dinov2_segformer.py
# Model architecture configuration

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DINOv2',
        model_name='dinov2_vitb14',
        img_size=(1960, 1960),
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],
        init_cfg=dict(
            type='Pretrained',
            checkpoint='assets/pretrained_models/dinov2_vitb14_pretrain.pth'
        )
    ),
    decode_head=dict(
        type='SegformerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=18,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=18,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)
```

### Dataset Configuration

```python
# configs/_base_/datasets/criobe_finegrained.py
# Dataset and data pipeline configuration

dataset_type = 'CoralDataset'
data_root = 'data/prepared_mmseg/criobe_finegrained'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

crop_size = (1960, 1960)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1960, 1960), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1960, 1960),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train',
        ann_dir='labels/train',
        pipeline=train_pipeline
    ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=val_pipeline
    ),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/test',
        ann_dir='labels/test',
        pipeline=val_pipeline
    )
)
```

### Schedule Configuration

```python
# configs/_base_/schedules/schedule_150e.py
# Optimizer and learning rate schedule

optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'norm': dict(decay_mult=0.)
        }
    )
)

optimizer_config = dict(grad_clip=None)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=0.9,
    min_lr=1e-6,
    by_epoch=False
)

runner = dict(type='EpochBasedRunner', max_epochs=150)
```

### Complete Training Configuration

```python
# configs/dinov2_vitb14_criobe_finegrained.py
# Main training configuration

_base_ = [
    './_base_/models/dinov2_segformer.py',
    './_base_/datasets/criobe_finegrained.py',
    './_base_/schedules/schedule_150e.py',
    './_base_/default_runtime.py'
]

# Model overrides
model = dict(
    decode_head=dict(num_classes=18),
    auxiliary_head=dict(num_classes=18)
)

# Training settings
data = dict(
    samples_per_gpu=4,  # Batch size per GPU
    workers_per_gpu=4
)

# Checkpoint and logging
checkpoint_config = dict(by_epoch=True, interval=10, max_keep_ckpts=5)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook')
    ]
)

# Evaluation
evaluation = dict(
    interval=10,
    metric='mIoU',
    pre_eval=True,
    save_best='mIoU'
)

# Runtime
work_dir = 'work_dirs/dinov2_vitb14_criobe_finegrained'
gpu_ids = [0]
```

### Start Training

```bash
cd ~/Projects/coral-segmentation/DINOv2_mmseg

# Train with config file
pixi run python tools/train.py \
    configs/dinov2_vitb14_criobe_finegrained.py \
    --work-dir work_dirs/dinov2_vitb14_criobe_finegrained \
    --gpu-ids 0

# Multi-GPU training
pixi run python -m torch.distributed.launch \
    --nproc_per_node=2 \
    tools/train.py \
    configs/dinov2_vitb14_criobe_finegrained.py \
    --launcher pytorch
```

## Step 4: Configure grid_pose_detection Training

### Grid Corners Configuration

```yaml
# experiments/gridcorners_train.yaml
# 4-point corner detection

data: data/prepared_yolo/grid_corners/dataset.yaml

model: yolo11n-pose.yaml  # Lightweight for 4 keypoints
task: pose

epochs: 100
batch: 16               # Larger batch for small model
imgsz: 1920
device: 0
workers: 4

amp: true

optimizer: AdamW
lr0: 0.002
lrf: 0.01

# Keypoint-specific settings
pose: 1.0              # Keypoint loss weight
kobj: 2.0              # Keypoint objectness weight

# Augmentation (less aggressive for keypoints)
hsv_h: 0.01
hsv_s: 0.4
hsv_v: 0.3
degrees: 0.0           # No rotation for grid detection
translate: 0.05        # Minimal translation
scale: 0.3
fliplr: 0.5
mosaic: 1.0

project: runs/train
name: gridcorners_yolo11n
```

### Grid Pose Configuration

```yaml
# experiments/gridpose_train.yaml
# 117-point grid intersection detection

data: data/prepared_yolo/grid_pose/dataset.yaml

model: yolo11n-pose.yaml
task: pose

epochs: 150            # More epochs for complex keypoints
batch: 8               # Smaller batch for 117 keypoints
imgsz: 1920
device: 0
workers: 4

amp: true

optimizer: AdamW
lr0: 0.001
lrf: 0.01

# Keypoint-specific settings
pose: 1.5              # Higher weight for many keypoints
kobj: 2.5

# Augmentation
hsv_h: 0.01
hsv_s: 0.3
hsv_v: 0.2
degrees: 0.0
translate: 0.02        # Very minimal translation
scale: 0.2
fliplr: 0.5
mosaic: 1.0

project: runs/train
name: gridpose_yolo11n
```

## Step 5: Key Hyperparameters Reference

### Learning Rate

| Parameter | Purpose | Typical Range | Effect |
|-----------|---------|---------------|--------|
| `lr0` | Initial learning rate | 0.0001 - 0.01 | Too high: unstable training, Too low: slow convergence |
| `lrf` | Final LR multiplier | 0.01 - 0.1 | Controls LR decay |
| `warmup_epochs` | Warmup period | 1 - 5 | Gradual LR increase at start |

**Tuning Tips**:
- Start with `lr0=0.001` for most tasks
- Increase for smaller models, decrease for larger models
- Use warmup for large batch sizes
- Monitor loss: if oscillating, reduce LR

### Batch Size

| GPU Memory | YOLO-n | YOLO-m | YOLO-l | DINOv2 |
|-----------|--------|--------|--------|--------|
| 8GB | 16 | 8 | 4 | 2 |
| 12GB | 24 | 12 | 8 | 4 |
| 24GB | 48 | 24 | 16 | 8 |

**Considerations**:
- Larger batch: more stable gradients, faster training
- Smaller batch: more noise, better generalization
- Use gradient accumulation for effective larger batch

### Data Augmentation

| Parameter | Purpose | Range | Coral Specific |
|-----------|---------|-------|----------------|
| `hsv_h` | Hue variation | 0.0 - 0.05 | Low (underwater color consistent) |
| `hsv_s` | Saturation | 0.0 - 0.9 | High (water clarity varies) |
| `hsv_v` | Brightness | 0.0 - 0.9 | High (depth/lighting varies) |
| `degrees` | Rotation | 0 - 180 | 0 (quadrats are oriented) |
| `translate` | Translation | 0.0 - 0.3 | Low (grid must stay visible) |
| `scale` | Scale | 0.0 - 0.9 | Moderate |
| `fliplr` | Horizontal flip | 0.0 - 1.0 | 0.5 (natural variation) |
| `mosaic` | Mosaic aug | 0.0 - 1.0 | 1.0 (helps with small corals) |

**Coral-Specific Guidelines**:
```yaml
# Conservative (when grid must be preserved)
degrees: 0.0
translate: 0.05
scale: 0.3

# Aggressive (for coral segmentation)
hsv_s: 0.7
hsv_v: 0.4
fliplr: 0.5
mosaic: 1.0
```

## Step 6: Training Monitoring

### Tensorboard

```bash
# Start tensorboard
pixi run tensorboard --logdir runs/train

# Or for MMSegmentation
pixi run tensorboard --logdir work_dirs

# Access at http://localhost:6006
```

**Metrics to Monitor**:
- **Loss curves**: Should decrease smoothly
- **mAP/mIoU**: Should increase
- **Learning rate**: Check schedule is correct
- **GPU utilization**: Should be >80%

### Logging

**YOLO logs** (console output):
```
Epoch  GPU_mem  box_loss  cls_loss  dfl_loss  Instances  Size
  1/100    5.2G     1.234     0.567     1.123        145  1920
  2/100    5.2G     1.198     0.543     1.089        152  1920
...
```

**MMSegmentation logs** (work_dirs/*/log.txt):
```
2024-01-15 10:30:00,123 - INFO - Epoch [1][50/500]  lr: 1.000e-04, loss: 2.3456
2024-01-15 10:35:00,456 - INFO - Epoch [1][100/500] lr: 1.000e-04, loss: 2.1234
```

### Checkpoints

**YOLO checkpoints**:
```
runs/train/<experiment_name>/
├── weights/
│   ├── last.pt          # Latest checkpoint
│   ├── best.pt          # Best mAP checkpoint
│   └── epoch_*.pt       # Periodic checkpoints
├── results.csv          # Training metrics
└── args.yaml            # Training arguments
```

**MMSegmentation checkpoints**:
```
work_dirs/<experiment_name>/
├── latest.pth           # Latest checkpoint
├── best_mIoU_epoch_*.pth  # Best checkpoint
├── epoch_*.pth          # Periodic checkpoints
└── *.log                # Training logs
```

## Step 7: Resume Training

### Resume YOLO Training

```bash
cd ~/Projects/coral-segmentation/coral_seg_yolo

# Resume from last checkpoint
pixi run yolo segment train \
    resume=runs/train/criobe_finegrained_yolo11m/weights/last.pt

# Or specify config and checkpoint
pixi run python src/training/train.py \
    --config experiments/criobe_finegrained_train.yaml \
    --resume runs/train/criobe_finegrained_yolo11m/weights/last.pt
```

### Resume MMSegmentation Training

```bash
cd ~/Projects/coral-segmentation/DINOv2_mmseg

# Resume from last checkpoint
pixi run python tools/train.py \
    configs/dinov2_vitb14_criobe_finegrained.py \
    --resume-from work_dirs/dinov2_vitb14_criobe_finegrained/latest.pth

# Or auto-resume
pixi run python tools/train.py \
    configs/dinov2_vitb14_criobe_finegrained.py \
    --auto-resume
```

## Step 8: Multi-GPU Training

### YOLO Multi-GPU

```yaml
# experiments/criobe_finegrained_multigpu.yaml
device: [0, 1]          # Use GPUs 0 and 1
batch: 16               # Total batch size (8 per GPU)
```

```bash
# Train with multiple GPUs
pixi run yolo segment train \
    data=data/prepared_yolo/criobe_finegrained/dataset.yaml \
    model=yolo11m-seg.yaml \
    device=[0,1] \
    batch=16
```

### MMSegmentation Multi-GPU

```bash
cd ~/Projects/coral-segmentation/DINOv2_mmseg

# Distributed training
pixi run python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=29500 \
    tools/train.py \
    configs/dinov2_vitb14_criobe_finegrained.py \
    --launcher pytorch

# Or using torchrun
pixi run torchrun \
    --nproc_per_node=2 \
    tools/train.py \
    configs/dinov2_vitb14_criobe_finegrained.py \
    --launcher pytorch
```

## Common Configuration Patterns

### Pattern 1: Quick Debugging

```yaml
# Fast iteration for testing changes
epochs: 5
batch: 16
imgsz: 640              # Much smaller
val: false              # Skip validation
plots: false
model: yolo11n-seg.yaml # Smallest model
```

### Pattern 2: Baseline Training

```yaml
# Standard configuration for comparison
epochs: 100
batch: 8
imgsz: 1920
model: yolo11m-seg.yaml
optimizer: AdamW
lr0: 0.001
# Default augmentations
```

### Pattern 3: High Accuracy

```yaml
# Maximum accuracy (slower)
epochs: 200
batch: 4
imgsz: 1920
model: yolo11x-seg.yaml  # Largest model
amp: false               # Full precision
optimizer: SGD
lr0: 0.01
momentum: 0.937
# Conservative augmentations
degrees: 0.0
translate: 0.05
```

### Pattern 4: Transfer Learning

```yaml
# Fine-tune from pre-trained model
model: path/to/pretrained.pt
freeze: 10               # Freeze first 10 layers
epochs: 50               # Fewer epochs
lr0: 0.0001              # Lower learning rate
```

## Best Practices

### Configuration Management

1. **Use descriptive names**: `criobe_finegrained_yolo11m_high_lr.yaml`
2. **Version configs**: Include date or version number
3. **Document changes**: Add comments explaining modifications
4. **Track experiments**: Log which configs were used
5. **Keep baselines**: Save successful configurations

### Hyperparameter Tuning

1. **Start with baseline**: Use default values first
2. **Change one thing**: Isolate variable effects
3. **Monitor validation**: Prevent overfitting
4. **Use early stopping**: Stop when val loss plateaus
5. **Log everything**: Track all experiments

### Training Workflow

1. **Quick test**: Train 5 epochs on small subset
2. **Baseline**: Full training with default settings
3. **Iterate**: Modify one parameter at a time
4. **Compare**: Use tensorboard to compare runs
5. **Select best**: Based on validation metrics

## Troubleshooting

### Training Loss Not Decreasing

**Symptoms**: Loss stays high or increases

**Solutions**:
```yaml
# Reduce learning rate
lr0: 0.0001  # Down from 0.001

# Increase warmup
warmup_epochs: 5.0

# Check data
# Verify dataset.yaml paths are correct
# Check annotations are valid

# Reduce batch size
batch: 4  # May help with stability

# Disable augmentation temporarily
mosaic: 0.0
mixup: 0.0
```

### Overfitting

**Symptoms**: Train loss decreases but val loss increases

**Solutions**:
```yaml
# Increase augmentation
hsv_s: 0.9
hsv_v: 0.6
mosaic: 1.0
mixup: 0.15

# Add regularization
weight_decay: 0.001  # Up from 0.0005
dropout: 0.1

# Use smaller model
model: yolo11s-seg.yaml  # Down from m

# Get more data
# Add more training images or use data from other sources
```

### Out of Memory

**Symptoms**: `CUDA out of memory` error

**Solutions**:
```yaml
# Reduce batch size
batch: 4  # Down from 8

# Reduce image size
imgsz: 1280  # Down from 1920

# Enable AMP
amp: true

# Use smaller model
model: yolo11n-seg.yaml

# Gradient accumulation
# See module documentation
```

### Slow Training

**Symptoms**: Training takes too long

**Solutions**:
```yaml
# Increase batch size (if memory allows)
batch: 16

# Increase workers
workers: 8

# Enable AMP
amp: true

# Reduce image size (if acceptable)
imgsz: 1280

# Check GPU utilization
# Run: nvidia-smi
# Should be >80%
```

### Poor Validation Performance

**Symptoms**: Low mAP/mIoU on validation set

**Solutions**:
```yaml
# Train longer
epochs: 200

# Increase model size
model: yolo11l-seg.yaml

# Tune confidence threshold
conf: 0.25  # For inference

# Check for class imbalance
# Review dataset statistics

# Add more training data
# Especially for underrepresented classes
```

## Next Steps

!!! success "Training Configured!"
    Your training experiments are configured and ready to run!

**What's next**:

1. **Start Training** - Run your first training experiment
2. **Monitor Progress** - Use tensorboard to track training
3. **Evaluate Models** - Test on validation set
4. **Iterate** - Tune hyperparameters based on results

## Quick Reference

### Essential Config Parameters

**YOLO**:
```yaml
data: dataset.yaml
model: yolo11m-seg.yaml
epochs: 100
batch: 8
imgsz: 1920
device: 0
lr0: 0.001
```

**MMSegmentation**:
```python
_base_ = ['model.py', 'dataset.py', 'schedule.py']
optimizer = dict(type='AdamW', lr=0.0001)
runner = dict(max_epochs=150)
data = dict(samples_per_gpu=4)
```

### Training Commands

```bash
# YOLO
pixi run yolo segment train data=dataset.yaml model=yolo11m-seg.yaml

# MMSegmentation
pixi run python tools/train.py configs/config.py

# Resume
pixi run yolo segment train resume=weights/last.pt
pixi run python tools/train.py config.py --resume-from latest.pth
```

### Monitoring Commands

```bash
# Tensorboard
pixi run tensorboard --logdir runs/train
pixi run tensorboard --logdir work_dirs

# Watch training
tail -f runs/train/<experiment>/results.csv
tail -f work_dirs/<experiment>/log.txt

# GPU monitoring
watch -n 1 nvidia-smi
```

### Configuration Checklist

- [ ] Dataset paths correct in config
- [ ] Model architecture selected
- [ ] Learning rate configured
- [ ] Batch size set for GPU memory
- [ ] Augmentation parameters tuned for coral images
- [ ] Checkpoint saving configured
- [ ] Tensorboard logging enabled
- [ ] Validation interval set
- [ ] Output directory specified

---

**Questions?** See [Getting Help](../../../community/index.md) or start training and monitor results!
