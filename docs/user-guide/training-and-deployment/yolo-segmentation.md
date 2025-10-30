# YOLO Segmentation Training

Train fast and accurate YOLOv11 instance segmentation models for coral detection and classification.

## Introduction

This guide walks you through training YOLOv11 segmentation models on your annotated coral datasets. YOLO provides **end-to-end instance segmentation** with excellent speed-accuracy tradeoff, making it ideal for production deployments and real-time coral monitoring.

### When to Use YOLO Segmentation

**Best for:**

- Production coral monitoring systems
- Real-time or near-real-time processing requirements
- Fast iteration during model development
- Enough data to finetune a model | Fixed location with low variability in coral genus distibution

**Consider MMSeg instead if:**

- Low annotated data regime : generalization to new sites is crucial
- Speed is not a primary concern
- Two-stage workflow is acceptable

### What You'll Learn

- Pull annotated data from CVAT using FiftyOne
- Convert polyline annotations to YOLO segmentation format
- Configure training parameters for different model sizes
- Train YOLOv11 segmentation models with GPU acceleration
- Evaluate model performance with FiftyOne integration
- Export and deploy models to Nuclio

### Expected Outcomes

- Trained YOLO segmentation model achieving ~0.65-0.75 mAP@0.5
- Model inference time ~7-10 seconds per 1920x1920 image
- Deployment-ready model weights
- Comprehensive evaluation metrics

!!! tip "Start with Demo Datasets"
    Before training on full ML datasets, test your setup with demo datasets (5 samples) to verify your environment is correctly configured. This saves time if there are setup issues.

## Prerequisites

Ensure you have:

- [x] Completed one of the [Data Preparation Guides](../data-preparation/index.md)
- [x] FiftyOne dataset with polyline annotations (e.g., `criobe_finegrained_fo` or `banggai_segmentation_fo`)
- [x] CUDA-capable GPU with 8GB+ VRAM
- [x] Pixi installed and configured
- [x] At least 100GB free disk space

!!! info "Supported Datasets"
    This guide works with any FiftyOne dataset containing:

    - Images (JPG or PNG)
    - Polyline annotations with genus-level labels
    - Train/val/test splits

    Compatible with all datasets from Data Preparation Guides A, B, and C.

## Step 1: Environment Setup

### 1.1 Navigate to Module

```bash
cd PROJ_ROOT/criobe/coral_seg_yolo
```

### 1.2 Activate Pixi Environment

```bash
# For training only
pixi shell -e coral-seg-yolo

# For training + evaluation with FiftyOne (recommended)
pixi shell -e coral-seg-yolo-dev
```

The `coral-seg-yolo-dev` environment includes:

- Python 3.9
- PyTorch 2.5.0 with CUDA 12.1
- Ultralytics YOLO
- FiftyOne for evaluation and visualization
- All required dependencies

### 1.3 Verify GPU Access

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
```

## Step 2: Pull Data from CVAT

If you haven't already pulled your annotated dataset from CVAT, do so now:

```bash
cd PROJ_ROOT/criobe/data_engineering

# Activate data engineering environment
pixi shell

# Pull your segmentation project
python create_fiftyone_dataset.py \
    --cvat-project-name "criobe_finegrained_annotated" \
    --dataset-name "criobe_finegrained_fo"

# Or for Banggai dataset
python create_fiftyone_dataset.py \
    --cvat-project-name "banggai_coral_segmentation" \
    --dataset-name "banggai_segmentation_fo"
```

**Verify dataset in FiftyOne:**

```bash
fiftyone app launch criobe_finegrained_fo
```

Check:

- Images are loaded correctly
- Polyline annotations are present
- Train/val/test splits are assigned
- Label distribution looks reasonable

## Step 3: Prepare YOLO Format Data

### 3.1 Convert FiftyOne Dataset to YOLO Format

```bash
cd PROJ_ROOT/criobe/coral_seg_yolo
pixi shell -e coral-seg-yolo-dev

python src/prepare_data.py criobe_finegrained_fo
```

**What this script does:**

1. Loads FiftyOne dataset
2. Converts polyline annotations to YOLO segmentation format (normalized coordinates)
3. Copies images to train/val/test directories
4. Generates YOLO `.txt` label files
5. Creates `dataset.yaml` configuration file

**Expected output:**

```
INFO: Loading FiftyOne dataset: criobe_finegrained_fo
INFO: Found 450 samples (train: 315, val: 90, test: 45)
INFO: Converting polylines to YOLO segmentation format...
INFO: Processing train split: 315 samples
INFO: Processing val split: 90 samples
INFO: Processing test split: 45 samples
INFO: Created dataset.yaml with 16 classes
INFO: Dataset prepared at: data/coral_annotation_yolo_seg/criobe_finegrained_fo
```

### 3.2 Verify Output Structure

```bash
tree data/coral_annotation_yolo_seg/criobe_finegrained_fo -L 2
```

Expected structure:

```
data/coral_annotation_yolo_seg/criobe_finegrained_fo/
├── dataset.yaml          # YOLO dataset configuration
├── train/
│   ├── images/           # Training images
│   └── labels/           # YOLO segmentation labels (.txt)
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 3.3 Inspect dataset.yaml

```bash
cat data/coral_annotation_yolo_seg/criobe_finegrained_fo/dataset.yaml
```

Should contain:

```yaml
path: PROJ_ROOT/criobe/coral_seg_yolo/data/coral_annotation_yolo_seg/criobe_finegrained_fo
train: train/images
val: val/images
test: test/images

nc: 16  # number of classes
names:
  0: Acanthastrea
  1: Acropora
  2: Astreopora
  3: Atrea
  4: Fungia
  5: Goniastrea
  6: Leptastrea
  7: Merulinidae
  8: Millepora
  9: Montastrea
  10: Montipora
  11: Other
  12: Pavona/Leptoseris
  13: Pocillopora
  14: Porites
  15: Psammocora
```

!!! warning "Absolute Paths Required"
    The `path` field must be an absolute path. The script automatically generates this.

### 3.4 Verify Label Files

```bash
# Check a sample label file
head -3 data/coral_annotation_yolo_seg/criobe_finegrained_fo/train/labels/sample_001.txt
```

YOLO segmentation format:
```
1 0.523 0.341 0.534 0.352 0.547 0.349 0.556 0.338 ...
3 0.672 0.584 0.681 0.593 0.695 0.591 0.704 0.579 ...
1 0.234 0.712 0.245 0.723 0.258 0.721 0.267 0.709 ...
```

Format: `class_id x1 y1 x2 y2 x3 y3 ... xn yn` (normalized 0-1 coordinates)

## Step 4: Configure Training

### 4.1 Choose Model Size

YOLOv11 offers multiple model sizes with speed-accuracy tradeoffs:

| Model | Parameters | VRAM Usage | Training Time | Inference Time | Accuracy |
|-------|------------|------------|---------------|----------------|----------|
| yolo11n-seg | 2.9M | ~4GB | ~4 hours | ~4s/image | Low |
| yolo11s-seg | 9.4M | ~6GB | ~6 hours | ~5s/image | Medium |
| yolo11m-seg | 22.4M | ~10GB | ~8 hours | ~7s/image | Good ⭐ |
| yolo11l-seg | 27.6M | ~12GB | ~10 hours | ~9s/image | Better |
| yolo11x-seg | 62.7M | ~16GB | ~12 hours | ~12s/image | Best |

**Recommended:** Start with **yolo11m-seg** for best balance.

### 4.2 Edit Training Configuration

Copy and customize the training config:

```bash
cp experiments/train_cfg_yolo_criobe.yaml experiments/my_training.yaml
nano experiments/my_training.yaml
```

**Key parameters to configure:**

```yaml
# Model checkpoint
model: yolo11m-seg.pt  # Change to desired size

# Dataset
data: data/coral_annotation_yolo_seg/criobe_finegrained_fo/dataset.yaml

# Training hyperparameters
epochs: 100            # Number of training epochs
batch: 16              # Batch size (adjust for your GPU VRAM)
imgsz: 1920            # Input image size (matches quadrat images)

# Optimizer
optimizer: AdamW       # Adam with weight decay
lr0: 0.001            # Initial learning rate
lrf: 0.01             # Final learning rate (lr0 * lrf)
momentum: 0.937
weight_decay: 0.0005

# Data augmentation
hsv_h: 0.015          # Hue augmentation
hsv_s: 0.7            # Saturation augmentation
hsv_v: 0.4            # Value augmentation
degrees: 10.0         # Rotation augmentation (±degrees)
translate: 0.1        # Translation augmentation
scale: 0.5            # Scale augmentation
shear: 0.0            # Shear augmentation
perspective: 0.0001   # Perspective augmentation
flipud: 0.5           # Vertical flip probability
fliplr: 0.5           # Horizontal flip probability
mosaic: 1.0           # Mosaic augmentation probability
mixup: 0.0            # Mixup augmentation probability

# Advanced
close_mosaic: 10      # Disable mosaic last N epochs (for stability)
amp: true             # Automatic Mixed Precision (faster training)
fraction: 1.0         # Use 100% of data (reduce for testing)
patience: 30          # Early stopping patience (epochs)
save: true            # Save checkpoints
save_period: 10       # Save checkpoint every N epochs
val: true             # Validate during training
plots: true           # Generate training plots
```

!!! tip "Batch Size Tuning"
    If you get CUDA out of memory errors:

    - Reduce `batch` to 8, 4, or even 2
    - Reduce `imgsz` to 1280 or 640
    - Use `yolo11n-seg` or `yolo11s-seg` instead

    For larger GPUs (24GB+):

    - Increase `batch` to 32 or 64 for faster training

### 4.3 Create Project Directory

Training outputs will be saved to a run directory:

```bash
# Runs will be saved automatically to:
# runs/segment/train/
# runs/segment/train2/
# runs/segment/train3/
# etc.
```

## Step 5: Train Model

### 5.1 Start Training

```bash
pixi run -e coral-seg-yolo yolo task=segment mode=train \
    cfg=experiments/my_training.yaml \
    project=runs/segment \
    name=criobe_finegrained_yolo11m
```

**Alternative with explicit model:**

```bash
pixi run -e coral-seg-yolo yolo segment train \
    model=yolo11m-seg.pt \
    data=data/coral_annotation_yolo_seg/criobe_finegrained_fo/dataset.yaml \
    epochs=100 \
    imgsz=1920 \
    batch=16 \
    project=runs/segment \
    name=criobe_finegrained_yolo11m
```

!!! info "First Run Downloads Pretrained Weights"
    The first time you run training, YOLO will automatically download the pretrained weights (~50MB for yolo11m-seg). This is normal and only happens once.

### 5.2 Monitor Training

Training progress is displayed in the terminal:

```
Epoch    GPU_mem   box_loss   seg_loss   cls_loss   dfl_loss  Instances       Size
  1/100      8.2G      1.234      1.567      2.345      1.123         89       1920

         Class     Images  Instances      Box(P          R      mAP50  mAP50-95)      Mask(P          R      mAP50  mAP50-95)
           all        90        756      0.612      0.548      0.573      0.345      0.603      0.541      0.561      0.298
    Acropora         90        134      0.724      0.689      0.712      0.445      0.715      0.681      0.704      0.398
      Porites        90        156      0.681      0.634      0.663      0.389      0.673      0.628      0.655      0.341
...
```

**Key metrics to watch:**

- **box_loss**: Decreasing = model learning bounding box localization
- **seg_loss**: Decreasing = model learning segmentation masks
- **cls_loss**: Decreasing = model learning classification
- **mAP50**: Mean Average Precision at IoU 0.5 (primary metric)
- **Mask(mAP50)**: Segmentation mask mAP (should reach 0.65-0.75 for coral data)

### 5.3 Use TensorBoard for Visualization

```bash
# In a separate terminal
pixi run -e coral-seg-yolo tensorboard --logdir runs/segment
```

Open browser at `http://localhost:6006` to view:

- Training and validation loss curves
- mAP metrics over time
- Learning rate schedule
- Sample predictions

### 5.4 Training Duration

Typical training times (yolo11m-seg, 450 images, RTX 4090):

- **100 epochs**: ~8 hours
- **Each epoch**: ~4-5 minutes

**Early stopping:** Training may stop early if validation mAP doesn't improve for `patience` epochs (default 30).

## Step 6: Evaluate Model

### 6.1 Automatic Validation

YOLO automatically validates during training. Results are saved to:

```
runs/segment/criobe_finegrained_yolo11m/
├── weights/
│   ├── best.pt          # Best checkpoint (highest mAP)
│   └── last.pt          # Last epoch checkpoint
├── results.csv          # Training metrics per epoch
├── results.png          # Plots of metrics
├── confusion_matrix.png # Class confusion matrix
├── val_batch0_pred.jpg  # Sample predictions
└── ...
```

### 6.2 Test Set Evaluation

Evaluate the best model on the test set:

```bash
pixi run -e coral-seg-yolo yolo segment val \
    model=runs/segment/criobe_finegrained_yolo11m/weights/best.pt \
    data=data/coral_annotation_yolo_seg/criobe_finegrained_fo/dataset.yaml \
    split=test \
    imgsz=1920 \
    save_json=true
```

**Output:**

```
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)      Mask(P          R      mAP50  mAP50-95)
                   all         45        324      0.658      0.612      0.647      0.389      0.651      0.608      0.643      0.361
            Acanthastrea          45         18      0.672      0.611      0.655      0.402      0.669      0.606      0.651      0.379
              Acropora         45         56      0.789      0.750      0.781      0.512      0.783      0.744      0.776      0.468
             Astreopora         45         22      0.634      0.591      0.618      0.351      0.628      0.584      0.612      0.328
...
```

### 6.3 FiftyOne Evaluation

For detailed error analysis, use FiftyOne integration:

```bash
pixi run -e coral-seg-yolo-dev python src/fiftyone_evals.py \
    --dataset-name criobe_finegrained_fo \
    --model-path runs/segment/criobe_finegrained_yolo11m/weights/best.pt \
    --pred-field-name "predictions_yolo11m"
```

**What this does:**

1. Loads your FiftyOne dataset
2. Runs model inference on all images
3. Stores predictions as a new field in the dataset
4. Computes detailed evaluation metrics
5. Enables interactive error analysis

**Launch FiftyOne app:**

```bash
fiftyone app launch criobe_finegrained_fo
```

**Interactive analysis:**

- Filter by prediction confidence
- Compare ground truth vs predictions side-by-side
- Identify false positives and false negatives
- Analyze per-class performance
- Find difficult examples for additional annotation

### 6.4 Key Metrics Interpretation

**Box Metrics:**

- **Precision**: High precision = few false positives (trustworthy predictions)
- **Recall**: High recall = few false negatives (model finds most corals)
- **mAP@0.5**: Primary metric (aim for >0.65 for coral data)
- **mAP@0.5:0.95**: Stricter metric across multiple IoU thresholds

**Mask Metrics:**

- Similar to box metrics but for segmentation masks
- Typically slightly lower than box metrics
- More important for precise boundary delineation

**Per-Class Performance:**

- Some genera will perform better than others
- Common corals (Acropora, Porites) usually have higher mAP
- Rare genera may have lower scores due to limited training samples

!!! tip "Improving Performance"
    If metrics are lower than expected:

    - **More data**: Annotate additional images, especially for underperforming classes
    - **Data quality**: Review and correct annotation errors
    - **Longer training**: Increase epochs to 150-200
    - **Larger model**: Try yolo11l-seg or yolo11x-seg
    - **Tune hyperparameters**: Adjust learning rate, augmentation strength

## Step 7: Inference on New Images

### 7.1 Single Image Prediction

```bash
pixi run -e coral-seg-yolo yolo segment predict \
    model=runs/segment/criobe_finegrained_yolo11m/weights/best.pt \
    source=data/test_samples/5-grid_removal/MooreaE2B_2020_05.jpg \
    imgsz=1920 \
    conf=0.25 \
    save=true
```

Results saved to `runs/segment/predict/`

### 7.2 Batch Inference

```bash
pixi run -e coral-seg-yolo yolo segment predict \
    model=runs/segment/criobe_finegrained_yolo11m/weights/best.pt \
    source=data/test_samples/5-grid_removal/ \
    imgsz=1920 \
    conf=0.25 \
    save=true \
    save_txt=true
```

### 7.3 Advanced Inference Script

For production inference with overlapping patches and refinement:

```bash
pixi run -e coral-seg-yolo-dev python src/inference_demo.py \
    data/test_samples/5-grid_removal/ \
    results/inference_output/ \
    runs/segment/criobe_finegrained_yolo11m/weights/best.pt \
    --method overlapping \
    --patch-size 1920 \
    --overlap 0.2 \
    --conf-threshold 0.25 \
    --iou-threshold 0.5
```

**Inference methods:**

- **direct**: Simple whole-image inference (fast)
- **overlapping**: Sliding window with overlap (better for large images)
- **tiling**: Non-overlapping tiles (memory efficient)

## Step 8: Export Model

### 8.1 Export to ONNX (for deployment)

```bash
pixi run -e coral-seg-yolo yolo export \
    model=runs/segment/criobe_finegrained_yolo11m/weights/best.pt \
    format=onnx \
    imgsz=1920 \
    simplify=true
```

Creates `best.onnx` in the same directory.

### 8.2 Export Formats

YOLO supports multiple export formats:

```bash
# TorchScript (good for production PyTorch)
pixi run -e coral-seg-yolo yolo export model=best.pt format=torchscript

# TensorRT (fastest GPU inference, NVIDIA only)
pixi run -e coral-seg-yolo yolo export model=best.pt format=engine device=0

# CoreML (for Apple devices)
pixi run -e coral-seg-yolo yolo export model=best.pt format=coreml
```

## Step 9: Deploy to Nuclio

### 9.1 Package Function with Trained Model

Use the **parameterized deployment script** - no manual copying needed:

```bash
cd PROJ_ROOT/criobe/coral_seg_yolo

# For CRIOBE finegrained model (coralsegv4)
./deploy_as_zip.sh coralsegv4 \
    runs/yolov11_criobe_finegrained_annotated/train/weights/best.pt

# For Banggai model (coralsegbanggai)
./deploy_as_zip.sh coralsegbanggai \
    runs/yolov11_banggai_extended_annotated/train/weights/best.pt

# Or use your custom training run path
./deploy_as_zip.sh coralsegv4 \
    runs/segment/my_experiment/weights/best.pt
```

**Script arguments:**
```bash
./deploy_as_zip.sh MODEL_NAME MODEL_WEIGHTS_PATH
```

- `MODEL_NAME`: `coralsegv4` or `coralsegbanggai`
- `MODEL_WEIGHTS_PATH`: Path to trained weights (.pt file)

**What happens:**

- Script validates model name and weights existence
- Copies weights as `best.pt` (standardized name)
- Copies `src/` module for inference engine
- Creates `nuclio.zip` ready for deployment

### 9.2 Deploy Function to Nuclio

After packaging, deploy using one of these options:

=== "Option 1: CVAT Centralized (Production)"

    ```bash
    # Extract to CVAT's serverless directory
    unzip nuclio.zip -d /path/to/cvat/serverless/pytorch/yolo/coralsegv4/

    # Deploy from CVAT directory
    cd /path/to/cvat
    nuctl deploy --project-name cvat \
        --path ./serverless/pytorch/yolo/coralsegv4/nuclio/ \
        --platform local \
        --verbose
    ```

=== "Option 2: Local Bundle (Development)"

    ```bash
    # Extract to local nuclio_bundles directory
    mkdir -p nuclio_bundles/coralsegv4
    unzip nuclio.zip -d nuclio_bundles/coralsegv4/

    # Deploy directly from local bundle
    nuctl deploy --project-name cvat \
        --path ./nuclio_bundles/coralsegv4/nuclio/ \
        --platform local \
        --verbose
    ```

!!! tip "Deployment Options"
    - **Option 1** is useful when CVAT manages all serverless functions centrally
    - **Option 2** is more flexible for development and testing

### 9.3 Verify Deployment

```bash
# Check function status
nuctl get functions --platform local | grep coralsegv4

# View function logs
nuctl get function pth-yolo-coralsegv4 --platform local
```

### 9.4 Test Deployed Function

```bash
# Test with sample image
curl -X POST http://localhost:8010 \
    -H "Content-Type: application/json" \
    -d @test_payload.json
```

**Expected response:** JSON array with detected coral polylines and labels.

### 9.5 Integrate with CVAT

1. Navigate to your coral segmentation project in CVAT
2. **Actions** → **Webhooks** → **Create webhook**
3. Configure:
    - **Target URL**: `http://bridge:8000/detect-model-webhook?model_name=pth-yolo-coralsegv4&conv_mask_to_poly=true`
    - **Events**: Check "When a job state is changed to 'in progress'"
4. Click **Submit**

!!! info "Model Name in Webhook"
    Use `pth-yolo-coralsegv4` or `pth-yolo-coralsegbanggai` based on which model you deployed.

Now when you open annotation jobs, your model will run automatically!

For complete deployment instructions, see [Model Deployment Guide](model-deployment.md).

## Troubleshooting

??? question "CUDA out of memory during training"
    **Solutions:**

    1. Reduce batch size: `batch=8` or `batch=4`
    2. Reduce image size: `imgsz=1280` or `imgsz=640`
    3. Use smaller model: `yolo11n-seg.pt` or `yolo11s-seg.pt`
    4. Enable gradient accumulation:
       ```yaml
       accumulate: 2  # Accumulate gradients over 2 batches
       ```
    5. Close other GPU applications

??? question "Training loss not decreasing"
    **Check:**

    1. Learning rate too high/low: Try `lr0=0.01` or `lr0=0.0001`
    2. Dataset issues: Verify labels are correct
    3. Data augmentation too aggressive: Reduce `hsv_*`, `degrees`, etc.
    4. Insufficient training: Increase `epochs` to 150-200

    **Debug:**
    ```bash
    # Visualize training batches
    python -c "from ultralytics import YOLO; YOLO('yolo11m-seg.pt').train(data='dataset.yaml', plots=True, epochs=1)"
    # Check runs/segment/train/train_batch0.jpg
    ```

??? question "Low mAP on validation/test set"
    **Possible causes:**

    1. **Overfitting**: Training mAP much higher than validation mAP
        - Solution: Increase data augmentation, add more training data
    2. **Underfitting**: Both training and validation mAP low
        - Solution: Train longer, use larger model, reduce augmentation
    3. **Data quality issues**: Incorrect annotations
        - Solution: Review dataset in FiftyOne, correct errors
    4. **Class imbalance**: Some classes have very few samples
        - Solution: Annotate more data for rare classes, use class weighting

??? question "Model predicts nothing or very few objects"
    **Check:**

    1. Confidence threshold too high: Try `conf=0.1` during inference
    2. Model undertrained: Check training logs, ensure loss converged
    3. Input image size mismatch: Use same `imgsz` as training
    4. Model path incorrect: Verify you're loading `best.pt`

    **Test:**
    ```bash
    # Run with very low confidence
    yolo segment predict model=best.pt source=test.jpg conf=0.01 save=true
    ```

??? question "Deployment to Nuclio fails"
    **Common issues:**

    1. Model file too large for zip packaging
        - Solution: Use shared volume mount instead of embedding
    2. Missing dependencies in function.yaml
        - Solution: Add all required packages to `build.requirements`
    3. GPU not accessible in container
        - Solution: Verify `nvidia.com/gpu: "1"` in resources

    **Debug:**
    ```bash
    # Check Nuclio logs
    nuctl get logs pth-yolo-my-model --platform local

    # Test locally first
    python main.py  # From deploy directory
    ```

## Advanced Topics

### Multi-Taxonomy Training

Train models on different taxonomic hierarchies by creating transformed datasets at the FiftyOne level first, then exporting them for training.

!!! note "Taxonomy Transformation Workflow"
    Taxonomy transformations must be applied when creating the FiftyOne dataset, not during YOLO export. The workflow is:

    1. Create a new FiftyOne dataset with simplified taxonomy using `data_engineering/create_cvat_annotation_tasks.py`
    2. Export the transformed dataset to YOLO format using `prepare_data.py`

**Example: Extended taxonomy (10 classes):**

```bash
# Step 1: Create FiftyOne dataset with extended taxonomy (in data_engineering/)
cd PROJ_ROOT/criobe/data_engineering
python create_cvat_annotation_tasks.py \
    --source-dataset criobe_finegrained_fo \
    --target-dataset criobe_extended_fo \
    --taxonomy extended

# Step 2: Export to YOLO format (in coral_seg_yolo/)
cd PROJ_ROOT/criobe/coral_seg_yolo
python src/prepare_data.py criobe_extended_fo
```

**Example: Main families (7 classes):**

```bash
# Step 1: Create FiftyOne dataset with main families taxonomy
cd PROJ_ROOT/criobe/data_engineering
python create_cvat_annotation_tasks.py \
    --source-dataset criobe_finegrained_fo \
    --target-dataset criobe_main_families_fo \
    --taxonomy main_families

# Step 2: Export to YOLO format
cd PROJ_ROOT/criobe/coral_seg_yolo
python src/prepare_data.py criobe_main_families_fo
```

See `data_engineering/tools.py` for available taxonomy definitions (finegrained, extended, main_families, agnostic).

### Transfer Learning from Pretrained Coral Models

Use existing coral models as initialization:

```bash
yolo segment train \
    model=assets/coralsegv4_yolo11m_best.pt \  # Start from coral model
    data=data/coral_annotation_yolo_seg/my_new_site/dataset.yaml \
    epochs=50 \  # Fewer epochs needed
    freeze=10    # Freeze first 10 layers
```

## Next Steps

Congratulations! You've trained a YOLO coral segmentation model. Next:

- **Deploy for production**: [Model Deployment Guide](model-deployment.md)
- **Compare with DINOv2**: [MMSeg Segmentation Training](mmseg-segmentation.md)
- **Train grid detection**: [Grid Detection Guide](grid-detection.md)
- **Annotate more data**: Use your model for semi-automatic annotation ([Guide A](../data-preparation/1-single-stage-segmentation.md))

## Reference

### Module Documentation

- [coral_seg_yolo/README.md](https://github.com/taiamiti/criobe/coral_seg_yolo/README.md) - Complete module documentation
- [coral_seg_yolo/experiments/](https://github.com/taiamiti/criobe/coral_seg_yolo/experiments/) - Training configuration examples

### External Resources

- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [YOLO Segmentation Guide](https://docs.ultralytics.com/tasks/segment/)
- [Training Tips](https://docs.ultralytics.com/guides/model-training-tips/)

---

**Related Guides**: [MMSeg Training](mmseg-segmentation.md) · [Model Deployment](model-deployment.md) · [Back to Overview](index.md)
