# Grid Removal with LaMa Inpainting

Set up and optionally fine-tune SimpleLama inpainting models for grid overlay removal from quadrat images.

## Introduction

This guide covers the `grid_inpainting` module, which uses **SimpleLama** (a lightweight variant of LaMa) to remove grid overlays from underwater quadrat images. Unlike other modules, **grid removal typically uses pre-trained models** without additional training.

### When to Use This Guide

**Use pre-trained SimpleLama (no training needed) when:**

- Working with standard PVC grid overlays
- Grid lines are thin (1-3mm in images)
- Grid color is white, yellow, or light-colored
- Working with CRIOBE or similar datasets

**Consider fine-tuning when:**

- Grid material is unusual (thick ropes, dark colors, etc.)
- Grid creates strong shadows
- Working with very different underwater conditions
- Pre-trained model produces visible artifacts

### What You'll Learn

- Download and use pre-trained SimpleLama model
- Generate grid masks from keypoint annotations
- Run grid removal on datasets
- Evaluate inpainting quality
- Optional: Fine-tune LaMa on custom grid patterns
- Deploy grid removal function to Nuclio

### Expected Outcomes

- Clean coral images with grid completely removed
- No visible artifacts or blur
- Preserved coral texture and color
- Processing time ~5-8 seconds per image

### Time Required

- **Setup with pre-trained model**: ~30 minutes
- **Testing and evaluation**: ~1 hour
- **Optional fine-tuning**: ~6-12 hours
- **Deployment**: ~30 minutes

## Prerequisites

Ensure you have:

- [x] Completed [Grid Detection Training](grid-detection.md) or have grid keypoint annotations
- [x] Warped quadrat images with visible grid overlays
- [x] Grid keypoint annotations (117 points in COCO keypoint format)
- [x] CUDA-capable GPU with 8GB+ VRAM
- [x] Pixi installed and configured
- [x] At least 50GB free disk space

## Step 1: Environment Setup

### 1.1 Navigate to Module

```bash
cd PROJ_ROOT/criobe/grid_inpainting
```

### 1.2 Activate Pixi Environment

```bash
# Full environment with training capabilities
pixi shell -e grid-inpainting

# Or minimal deployment environment
pixi shell -e grid-inpainting-deploy
```

The `grid-inpainting` environment includes:

- Python 3.9
- PyTorch with CUDA 12.1
- SimpleLama implementation
- OpenCV, PIL for image processing
- CUDA-optimized inpainting

### 1.3 Verify GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

## Step 2: Download Pre-trained Model

### 2.1 Download SimpleLama Weights

The pre-trained SimpleLama model (big-lama) is sufficient for most grid removal tasks:

```bash
# Download from Hugging Face or project repository
./download_model.sh
```

Or manually:

```bash
mkdir -p assets/pretrained_models

# Download big-lama checkpoint
wget https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt \
    -O assets/pretrained_models/big-lama.pt
```

**Model size:** ~50MB

### 2.2 Verify Model

```bash
python -c "
import torch
checkpoint = torch.load('assets/pretrained_models/big-lama.pt', map_location='cpu')
print('Model loaded successfully')
print(f'Keys: {list(checkpoint.keys())}')
"
```

!!! info "SimpleLama vs Original LaMa"
    **SimpleLama** is a streamlined version of the original LaMa (Large Mask Inpainting) model:

    - Smaller model size (~50MB vs 200MB)
    - Faster inference (~5s vs 10s per image)
    - Optimized for small, structured masks (like grids)
    - Slightly lower quality on very large masks

    For grid removal, SimpleLama is recommended.

## Step 3: Prepare Data

### 3.1 Organize Input Data

Grid removal requires two inputs:

1. **Images with grid overlays** (warped quadrat images)
2. **Grid keypoint annotations** (COCO keypoint JSON)

**Expected directory structure:**

```
data/
├── images_with_grid/
│   ├── MooreaE2B_2020_05.jpg
│   ├── MooreaE2B_2020_06.jpg
│   └── ...
└── grid_annotations/
    └── person_keypoints_default.json  # COCO format with 117 keypoints per image
```

### 3.2 Download Test Samples (Optional)

```bash
./download_test_samples.sh
```

This downloads example images for testing:

```
data/test_samples/
├── 3-image_warping/           # Warped images with grids
└── 4-grid_pose_export/        # Grid keypoint annotations
    └── person_keypoints_default.json
```

### 3.3 Verify Grid Annotations

Check that keypoint annotations are correct:

```bash
python src/visualize_grid_keypoints.py \
    --images-dir data/test_samples/3-image_warping/ \
    --annotations data/test_samples/4-grid_pose_export/person_keypoints_default.json \
    --output-dir results/grid_visualization
```

Verify:

- All 117 keypoints are present per image
- Keypoints align with grid intersections
- No missing or misplaced points

## Step 4: Run Grid Removal

### 4.1 Basic Grid Removal

Remove grids from a dataset using pre-trained SimpleLama:

```bash
pixi run -e grid-inpainting python grid_rem_with_kp.py remove_grid_from_coco_dataset \
    --data-path data/test_samples/3-image_warping/ \
    --labels-path data/test_samples/4-grid_pose_export/person_keypoints_default.json \
    --output-dir results/grid_removed/ \
    --model-path assets/pretrained_models/big-lama.pt \
    --device cuda:0 \
    --grid-line-width 8 \
    --save-masks
```

**Parameters:**

- `--data-path`: Directory containing images with grids
- `--labels-path`: COCO keypoint JSON file
- `--output-dir`: Where to save clean images
- `--model-path`: SimpleLama checkpoint
- `--device`: CUDA device (cuda:0) or cpu
- `--grid-line-width`: Width of grid lines in pixels (adjust to match your grid, default 8)
- `--save-masks`: Also save generated grid masks for debugging

**What this does:**

1. Loads images and grid keypoint annotations
2. Generates binary masks from keypoints (grid lines)
3. Runs SimpleLama inpainting to fill masked regions
4. Saves clean images without grids

**Expected output:**

```
INFO: Loading SimpleLama model from assets/pretrained_models/big-lama.pt
INFO: Model loaded successfully (CUDA)
INFO: Processing 25 images...
INFO: [1/25] MooreaE2B_2020_05.jpg
INFO:   - Generated grid mask from 117 keypoints
INFO:   - Running inpainting (CUDA)...
INFO:   - Inpainting completed in 5.3s
INFO:   - Saved to results/grid_removed/MooreaE2B_2020_05.jpg
...
INFO: Processed 25 images in 142.5s (5.7s per image average)
```

### 4.2 Adjust Grid Line Width

If grid lines are not fully removed or too much area is inpainted:

**Grid still visible:**

```bash
# Increase line width
python grid_rem_with_kp.py ... --grid-line-width 12
```

**Over-inpainting (coral texture lost):**

```bash
# Decrease line width
python grid_rem_with_kp.py ... --grid-line-width 5
```

### 4.3 Batch Processing

For large datasets:

```bash
pixi run -e grid-inpainting python grid_rem_with_kp.py remove_grid_batch \
    --data-dirs data/moorea_2020/ data/moorea_2021/ data/tikehau_2023/ \
    --output-dir results/all_sites_clean/ \
    --model-path assets/pretrained_models/big-lama.pt \
    --batch-size 4 \
    --num-workers 2
```

**Batch processing benefits:**

- Process multiple images in parallel
- Progress tracking with tqdm
- Automatic error handling and logging

## Step 5: Evaluate Grid Removal Quality

### 5.1 Visual Inspection

Open results in image viewer and check:

- [ ] Grid lines completely removed
- [ ] No visible artifacts or blur
- [ ] Coral textures preserved
- [ ] Color consistency maintained
- [ ] No edge effects or black regions

### 5.2 Compare Before/After

Create side-by-side comparisons:

```bash
python src/create_comparison.py \
    --original-dir data/test_samples/3-image_warping/ \
    --cleaned-dir results/grid_removed/ \
    --output-dir results/comparisons/ \
    --layout horizontal
```

### 5.3 Quantitative Evaluation (If Ground Truth Available)

If you have ground-truth clean images (without grids):

```bash
python src/evaluate_inpainting.py \
    --ground-truth-dir data/ground_truth_clean/ \
    --predicted-dir results/grid_removed/ \
    --metrics psnr ssim lpips
```

**Metrics:**

- **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better (>30 dB is good)
- **SSIM (Structural Similarity Index)**: Closer to 1.0 is better (>0.90 is good)
- **LPIPS (Learned Perceptual Image Patch Similarity)**: Lower is better (<0.1 is good)

### 5.4 Check for Common Artifacts

**Common issues to watch for:**

- **Blurring**: Inpainted regions appear softer than surrounding coral
    - Solution: Use smaller grid-line-width
- **Color shift**: Inpainted areas have different color tone
    - Solution: Fine-tune model on your specific images
- **Pattern repetition**: Inpainting creates unrealistic repeated patterns
    - Solution: Usually minor, acceptable for most use cases
- **Edge effects**: Artifacts at grid-background boundaries
    - Solution: Adjust mask dilation/erosion in preprocessing

## Step 6: Optional - Fine-tune LaMa

Fine-tuning is only needed if pre-trained model produces poor results on your specific grid type.

### 6.1 Prepare Training Data

For fine-tuning, you need image pairs:

- **Input**: Images with grids
- **Target**: Corresponding clean images (ground truth)

**Creating synthetic training data:**

If you don't have ground truth, you can:

1. Manually inpaint a subset of images (20-50) using Photoshop/GIMP
2. Use the pre-trained model's best outputs as pseudo-ground truth
3. Acquire clean images from same sites without grids

**Data structure:**

```
data/fine_tuning/
├── train/
│   ├── with_grid/
│   │   ├── img_001.jpg
│   │   └── ...
│   └── clean/
│       ├── img_001.jpg
│       └── ...
└── val/
    ├── with_grid/
    └── clean/
```

### 6.2 Generate Masks for Training

```bash
python src/prepare_training_masks.py \
    --images-with-grid data/fine_tuning/train/with_grid/ \
    --keypoints-json data/fine_tuning/train/keypoints.json \
    --output-dir data/fine_tuning/train/masks/ \
    --grid-line-width 8
```

### 6.3 Configure Fine-tuning

Edit `configs/train_config.yaml`:

```yaml
data:
    train_dir: data/fine_tuning/train
    val_dir: data/fine_tuning/val
    img_size: 1920

model:
    checkpoint: assets/pretrained_models/big-lama.pt  # Start from pre-trained
    architecture: simple_lama

training:
    epochs: 50                    # Fewer epochs for fine-tuning
    batch_size: 2                 # Small batch for large images
    learning_rate: 0.0001         # Lower LR for fine-tuning
    weight_decay: 0.0001

    optimizer: AdamW
    scheduler: CosineAnnealingLR

losses:
    l1_weight: 1.0               # Pixel-level loss
    perceptual_weight: 0.1       # VGG perceptual loss
    adversarial_weight: 0.01     # GAN loss

augmentation:
    random_crop: true
    crop_size: 512               # Train on patches
    random_flip: true
    color_jitter: 0.1
```

### 6.4 Start Fine-tuning

```bash
pixi run -e grid-inpainting python tools/train_lama.py \
    --config configs/train_config.yaml \
    --work-dir work_dirs/fine_tuned_lama \
    --gpu 0
```

**Monitor training:**

```bash
tensorboard --logdir work_dirs/fine_tuned_lama
```

### 6.5 Fine-tuning Duration

- **50 epochs, 100 image pairs**: ~6-8 hours on RTX 4090
- **Each epoch**: ~6-8 minutes

### 6.6 Evaluate Fine-tuned Model

```bash
pixi run -e grid-inpainting python grid_rem_with_kp.py remove_grid_from_coco_dataset \
    --data-path data/test_samples/3-image_warping/ \
    --labels-path data/test_samples/4-grid_pose_export/person_keypoints_default.json \
    --output-dir results/grid_removed_finetuned/ \
    --model-path work_dirs/fine_tuned_lama/best_checkpoint.pt \
    --device cuda:0
```

Compare outputs with pre-trained model to see if fine-tuning improved results.

## Step 7: Deploy to Nuclio

### 7.1 Prepare Deployment

```bash
cd PROJ_ROOT/criobe/grid_inpainting/deploy

# Copy model weights
cp ../assets/pretrained_models/big-lama.pt pth-lama-nuclio/model_weights.pt

# Or if you fine-tuned:
# cp ../work_dirs/fine_tuned_lama/best_checkpoint.pt pth-lama-nuclio/model_weights.pt
```

### 7.2 Review Deployment Function

The Nuclio function (`deploy/pth-lama-nuclio/main.py`) should:

1. Load SimpleLama model on startup
2. Receive grid keypoints from webhook
3. Generate grid mask from keypoints
4. Run inpainting
5. Return clean image

**Key parts of `main.py`:**

```python
import torch
from lama_inpaint import SimpleLama
import numpy as np
from PIL import Image

# Load model once at initialization
model = SimpleLama(checkpoint='model_weights.pt', device='cuda:0')

def handler(context, event):
    data = event.body

    # Decode image and keypoints
    image = decode_image(data['image'])
    keypoints = np.array(data['keypoints'])  # 117 x 2 array

    # Generate mask from keypoints
    mask = generate_grid_mask(image.shape, keypoints, line_width=8)

    # Run inpainting
    result = model.inpaint(image, mask)

    # Encode and return
    return encode_image(result)
```

### 7.3 Deploy Function

```bash
cd PROJ_ROOT/criobe/grid_inpainting/deploy

./deploy_as_zip.sh

nuctl deploy --project-name cvat \
    --path ./pth-lama-nuclio \
    --platform local \
    --verbose
```

### 7.4 Test Deployed Function

```bash
# Prepare test payload
cat > test_payload.json <<EOF
{
  "image": "<base64_encoded_image>",
  "keypoints": [[x1,y1], [x2,y2], ..., [x117,y117]]
}
EOF

# Test function
curl -X POST http://localhost:8003 \
    -H "Content-Type: application/json" \
    -d @test_payload.json \
    -o test_output.jpg
```

### 7.5 Integrate with CVAT Pipeline

The grid removal function is called automatically by the Bridge service when grid detection tasks are completed.

**No manual webhook configuration needed** - the bridge handles this internally via the `remove-grid-and-create-new-task-webhook` endpoint.

For complete pipeline integration, see [Three-Stage CRIOBE Setup](../data-preparation/3-three-stage-criobe.md).

## Troubleshooting

??? question "Grid lines still visible after inpainting"
    **Solutions:**

    1. Increase grid line width:
       ```bash
       --grid-line-width 12  # or 15
       ```
    2. Check keypoint accuracy - if points are off by >5 pixels, mask won't cover grid
    3. Grid may be thicker or higher contrast than expected - consider fine-tuning
    4. Try dilating the mask:
       ```python
       # In generate_grid_mask function
       import cv2
       mask = cv2.dilate(mask, kernel=np.ones((3,3)), iterations=1)
       ```

??? question "Inpainting produces blurry results"
    **Causes:**

    - Model filling too large regions
    - Grid line width too large
    - Model not optimized for your image characteristics

    **Solutions:**

    1. Reduce grid line width to minimum needed
    2. Fine-tune model on your specific dataset
    3. Post-process with sharpening:
       ```python
       from PIL import ImageFilter
       result_image = result_image.filter(ImageFilter.SHARPEN)
       ```

??? question "Color shift in inpainted regions"
    **Causes:**

    - Model trained on different underwater conditions
    - Lighting differences in your dataset

    **Solutions:**

    1. Fine-tune model on your specific images
    2. Color-correct images before grid removal:
       ```python
       # Normalize underwater white balance
       image = apply_underwater_color_correction(image)
       ```
    3. Accept minor color differences (often not noticeable in final annotations)

??? question "Very slow inference (>15s per image)"
    **Check:**

    1. Using GPU: `--device cuda:0` (not `cpu`)
    2. CUDA is properly installed and accessible
    3. Image size - very large images (>3000px) are slower

    **Optimize:**

    - Process in batches: `--batch-size 4`
    - Use mixed precision: `--amp true`
    - Resize images before inpainting (if acceptable):
       ```bash
       --resize 1920
       ```

??? question "Deployment function runs out of memory"
    **Solutions:**

    1. Reduce batch size in Nuclio function
    2. Process images sequentially
    3. Increase Nuclio function memory limits:
       ```yaml
       resources:
         limits:
           memory: 16Gi  # Increase from default 8Gi
       ```

## Advanced Topics

### Custom Mask Generation

For non-standard grids (irregular, curved, etc.):

```python
def generate_custom_mask(image_shape, keypoints, pattern='bezier'):
    """
    Generate mask with smooth curves instead of straight lines
    """
    from scipy.interpolate import make_interp_spline

    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    # Fit spline through keypoints for smooth curves
    for i in range(9):  # 9 horizontal lines
        row_kps = keypoints[i*13:(i+1)*13]
        spline = make_interp_spline(row_kps[:, 0], row_kps[:, 1], k=3)
        # Draw spline as grid line
        ...

    return mask
```

### Multi-Stage Inpainting

For very thick grids, use progressive inpainting:

```python
# First pass: Inpaint grid with larger margin
mask_stage1 = generate_grid_mask(image, keypoints, line_width=12)
result_stage1 = model.inpaint(image, mask_stage1)

# Second pass: Refine with narrower margin
mask_stage2 = generate_grid_mask(result_stage1, keypoints, line_width=6)
result_final = model.inpaint(result_stage1, mask_stage2)
```

### Integration with Image Enhancement

Combine grid removal with underwater image enhancement:

```python
from underwater_enhancement import enhance_underwater_image

# Enhance before grid removal
enhanced = enhance_underwater_image(original_image)

# Remove grid
clean = remove_grid(enhanced, keypoints)

# Optional: Enhance again after grid removal
final = enhance_underwater_image(clean, mild=True)
```

## Next Steps

Congratulations! You've set up grid removal for your pipeline. Next:

- **Complete the pipeline**: [Three-Stage CRIOBE Setup](../data-preparation/3-three-stage-criobe.md)
- **Deploy all models**: [Model Deployment Guide](model-deployment.md)
- **Train coral segmentation**: [YOLO Segmentation](yolo-segmentation.md) or [MMSeg Segmentation](mmseg-segmentation.md)

## Reference

### Module Documentation

- [grid_inpainting/README.md](https://github.com/taiamiti/criobe/grid_inpainting/README.md)
- [grid_inpainting/src/](https://github.com/taiamiti/criobe/grid_inpainting/src/)

### External Resources

- [LaMa Paper](https://arxiv.org/abs/2109.07161) - Original LaMa inpainting
- [SimpleLama GitHub](https://github.com/enesmsahin/simple-lama-inpainting)
- [Image Inpainting Survey](https://arxiv.org/abs/2001.00212)

---

**Related Guides**: [Grid Detection](grid-detection.md) · [Model Deployment](model-deployment.md) · [Back to Overview](index.md)
