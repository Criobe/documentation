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

!!! info "Environment Variables Required"
    Before proceeding, ensure your `.env` file is configured with data paths and processing settings. See the [Environment Variables Guide](../../setup/configuration/for-developers/1-environment-variables.md) for the complete `grid_inpainting` configuration.

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

Check that keypoint annotations are correct using FiftyOne:

```bash
pixi run -e grid-inpainting python -c "
import fiftyone as fo
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    label_types=['keypoints'],
    data_path='data/test_samples/3-image_warping/',
    labels_path='data/test_samples/4-grid_pose_export/person_keypoints_default.json'
)
session = fo.launch_app(dataset)
session.wait()
"
```

Verify in the FiftyOne app:

- All 117 keypoints are present per image
- Keypoints align with grid intersections
- No missing or misplaced points

## Step 4: Run Grid Removal

### 4.1 Basic Grid Removal

Remove grids from a dataset using pre-trained SimpleLama:

```bash
pixi run -e grid-inpainting python grid_rem_with_kp.py remove_grid_from_coco_dataset \
    data/test_samples/3-image_warping/ \
    data/test_samples/4-grid_pose_export/person_keypoints_default.json \
    results/grid_removed/
```

**Parameters:**

- First argument: Directory containing images with grids (data path)
- Second argument: COCO keypoint JSON file (labels path)
- Third argument: Where to save clean images (output directory)

!!! note "Configuration Note"
    Model path, device, grid line width, and batch size are hardcoded in the script. To customize:

    - Model: Hardcoded to `models/big-lama.pt`
    - Grid line width: Hardcoded to 30px in `src/grid_remover.py:54`
    - Batch size: Hardcoded to 50 in `grid_rem_with_kp.py:59`
    - Device: Automatically uses CUDA if available

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

If grid lines are not fully removed or too much area is inpainted, you'll need to modify the hardcoded grid line width in the source code:

!!! tip "Adjusting Grid Line Width"
    Grid line width is hardcoded to 30px in `src/grid_remover.py:54`. To adjust:

    **If grid is still visible** (under-inpainting):
    - Edit `src/grid_remover.py` and increase the line width value (e.g., to 35 or 40)

    **If too much coral texture is removed** (over-inpainting):
    - Edit `src/grid_remover.py` and decrease the line width value (e.g., to 25 or 20)

    After modifying the source, re-run the grid removal command.

## Step 5: Evaluate Grid Removal Quality

### 5.1 Visual Inspection

Open results in image viewer and check:

- [ ] Grid lines completely removed
- [ ] No visible artifacts or blur
- [ ] Coral textures preserved
- [ ] Color consistency maintained
- [ ] No edge effects or black regions

### 5.2 Compare Before/After

Use an image viewer or photo editing software to create side-by-side comparisons of the original and grid-removed images. This helps visualize the quality of grid removal.

Alternatively, use FiftyOne to compare results:

```bash
pixi run -e grid-inpainting python -c "
import fiftyone as fo
from pathlib import Path
# Compare before/after by loading both as separate datasets
# or use external tools like ImageMagick for montages
"
```

### 5.3 Check for Common Artifacts

**Common issues to watch for:**

- **Blurring**: Inpainted regions appear softer than surrounding coral
    - Solution: Adjust grid line width in source code (see Step 4.2)
- **Color shift**: Inpainted areas have different color tone
    - Usually minor and acceptable for coral segmentation tasks
- **Pattern repetition**: Inpainting creates unrealistic repeated patterns
    - Usually minor, acceptable for most use cases
- **Edge effects**: Artifacts at grid-background boundaries
    - Typically minimal with SimpleLama model

!!! note "Quantitative Evaluation"
    If you have ground-truth clean images without grids, you can evaluate grid removal quality using standard image quality metrics (PSNR, SSIM, LPIPS) with external tools like scikit-image or PyTorch metrics.

!!! info "Fine-tuning Not Currently Supported"
    This module is deployment-focused and does not include training infrastructure. The pre-trained SimpleLama model works well for most grid removal tasks. If you need custom training, you would need to set up a separate training environment using the original LaMa repository.

## Step 6: Deploy to Nuclio

### 6.1 Prepare Deployment

```bash
cd PROJ_ROOT/criobe/grid_inpainting

# Model is automatically downloaded by Nuclio function.yaml
# No manual model copying needed
```

!!! note "Automatic Model Download"
    The grid inpainting Nuclio function automatically downloads the SimpleLama model from Google Cloud Storage during deployment. You don't need to manually copy model weights.

### 6.2 Review Deployment Function

The Nuclio function (`deploy/main.py`) should:

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

### 6.3 Package and Deploy Function

```bash
cd PROJ_ROOT/criobe/grid_inpainting

# Package function (model downloads automatically via function.yaml)
./deploy_as_zip.sh
```

After packaging, deploy using one of these options:

=== "Option 1: CVAT Centralized (Production)"

    ```bash
    # Extract to CVAT's serverless directory
    unzip nuclio.zip -d /path/to/cvat/serverless/pytorch/lama/

    # Deploy from CVAT directory
    cd /path/to/cvat
    nuctl deploy --project-name cvat \
        --path ./serverless/pytorch/lama/nuclio/ \
        --platform local \
        --verbose
    ```

=== "Option 2: Local Bundle (Development)"

    ```bash
    # Extract to local nuclio_bundles directory
    mkdir -p nuclio_bundles/lama
    unzip nuclio.zip -d nuclio_bundles/lama/

    # Deploy directly from local bundle
    nuctl deploy --project-name cvat \
        --path ./nuclio_bundles/lama/nuclio/ \
        --platform local \
        --verbose
    ```

!!! tip "Deployment Options"
    - **Option 1** is useful when CVAT manages all serverless functions centrally
    - **Option 2** is more flexible for development and testing

### 6.4 Test Deployed Function

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

### 6.5 Integrate with CVAT Pipeline

The grid removal function is called automatically by the Bridge service when grid detection tasks are completed.

**No manual webhook configuration needed** - the bridge handles this internally via the `remove-grid-and-create-new-task-webhook` endpoint.

For complete pipeline integration, see [Three-Stage CRIOBE Setup](../data-preparation/3-three-stage-criobe.md).

## Troubleshooting

??? question "Grid lines still visible after inpainting"
    **Solutions:**

    1. Increase grid line width by editing `src/grid_remover.py` line 54 (default: 30px → try 35 or 40px)
    2. Check keypoint accuracy - if points are off by >5 pixels, mask won't cover grid
    3. Grid may be thicker or higher contrast than expected
    4. Try dilating the mask by modifying `src/grid_remover.py`:
       ```python
       # In generate_grid_mask function
       import cv2
       mask = cv2.dilate(mask, kernel=np.ones((3,3)), iterations=1)
       ```

??? question "Inpainting produces blurry results"
    **Causes:**

    - Model filling too large regions
    - Grid line width too large (edit `src/grid_remover.py` to reduce)

    **Solutions:**

    1. Reduce grid line width in source code to minimum needed
    2. Post-process with sharpening:
       ```python
       from PIL import ImageFilter
       result_image = result_image.filter(ImageFilter.SHARPEN)
       ```

??? question "Color shift in inpainted regions"
    **Causes:**

    - Model trained on different underwater conditions
    - Lighting differences in your dataset

    **Solutions:**

    1. Color-correct images before grid removal using external tools
    2. Accept minor color differences (often not noticeable in final annotations)
    3. The pre-trained SimpleLama model generally handles underwater images well

??? question "Very slow inference (>15s per image)"
    **Check:**

    1. GPU is being used (automatically selected if CUDA is available)
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
