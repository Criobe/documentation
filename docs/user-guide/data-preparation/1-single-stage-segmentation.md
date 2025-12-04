# Guide A: Single-Stage Segmentation Setup

Learn how to create an annotated coral segmentation dataset when your images are already pre-processed (cropped and clean).

## Introduction

This guide shows you how to set up **the simplest annotation workflow** for coral segmentation. Use this approach when you have clean coral images that are already cropped to the region of interest and don't require grid removal.

### Use Case

**When to use this guide:**

- Images are already cropped to show only the coral quadrat area
- No grid overlay needs to be removed
- Images have consistent resolution and perspective
- Ready for direct annotation

**Examples of suitable images:**

- Photos taken with underwater cameras at fixed heights
- Pre-processed images from previous pipeline stages
- Images from the Banggai dataset after warping

### What You'll Learn

- Create a CVAT project with coral genus-level labels
- Configure annotation settings for polyline segmentation
- Deploy a YOLO model for semi-automatic pre-annotation
- Set up the model detection webhook
- Perform manual correction of model predictions
- Export annotations to FiftyOne for training

### Expected Outcome

At the end of this guide, you will have:

- A CVAT project with labeled coral images
- High-quality polygon annotations for 16 coral genera (CVAT polygons → FiftyOne polylines)
- A FiftyOne dataset ready for YOLO or MMSeg training
- Understanding of semi-automatic annotation workflows

### Time Required

- **Initial setup**: ~30 minutes
- **Annotation time**: Varies by dataset size
    - Manual annotation: ~10-15 minutes per image
    - Semi-automatic (pre-annotation + correction): ~3-5 minutes per image

## Prerequisites

Before starting, ensure you have:

- [x] CVAT instance running at `http://localhost:8080` (or your configured URL)
- [x] Admin access to CVAT
- [x] Nuclio serverless platform deployed (for semi-automatic annotation)
- [x] Clean coral images ready for annotation
- [x] Basic understanding of coral morphology (recommended)

!!! tip "Optional: Bridge Service"
    The Bridge service is not required for this single-stage workflow. It's only needed for multi-stage pipelines (Guides B and C).

## Step 1: Create CVAT Project for Segmentation

### 1.1 Create New Project

1. Log in to CVAT at `http://localhost:8080`
2. Click **Projects** in the top navigation
3. Click the **+** button to create a new project
4. Enter project details:
    - **Name**: `my_coral_segmentation` (or your preferred name)
    - **Description**: "Coral genus-level segmentation for finegrained taxonomy (16 genera)"

### 1.2 Configure Label Schema

In the project creation dialog, you need to configure the labels. CVAT provides a **Raw** mode for JSON input.

1. Click the **Raw** tab in the label configuration section
2. **Delete any existing JSON** in the text area
3. **Copy and paste** the following JSON exactly:

```json
[
  {
    "name": "Acanthastrea",
    "color": "#ff0000",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Acropora",
    "color": "#00ff00",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Astreopora",
    "color": "#0000ff",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Atrea",
    "color": "#ffff00",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Fungia",
    "color": "#ff00ff",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Goniastrea",
    "color": "#00ffff",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Leptastrea",
    "color": "#ff8000",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Merulinidae",
    "color": "#8000ff",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Millepora",
    "color": "#00ff80",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Montastrea",
    "color": "#ff0080",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Montipora",
    "color": "#80ff00",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Other",
    "color": "#808080",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Pavona/Leptoseris",
    "color": "#ff8080",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Pocillopora",
    "color": "#8080ff",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Porites",
    "color": "#80ff80",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  },
  {
    "name": "Psammocora",
    "color": "#ff80ff",
    "type": "polygon",
    "attributes": [
      {
        "name": "confidence",
        "input_type": "number",
        "mutable": true,
        "values": [
          "0",
          "100",
          "1"
        ],
        "default_value": "100"
      }
    ]
  }
]
```

!!! warning "Important Label Configuration Notes"
    - **Annotation Type**: Must be `"polyline"` for coral instance segmentation
    - **Label Names**: Must match exactly (case-sensitive)
    - **Color Codes**: Help distinguish genera visually; you can customize colors if needed
    - **Pavona/Leptoseris**: These two genera are merged because they're difficult to distinguish in quadrat images

4. Click **Continue** to create the project

### 1.3 Verify Label Configuration

After creating the project:

1. Open the project page
2. Click the **Labels** tab
3. Verify you see all 16 coral genera listed
4. Each label should show its assigned color

## Step 2: Upload Images to CVAT

### 2.1 Create Annotation Tasks

For better dataset organization, create separate tasks for different subsets (train/val/test):

1. Navigate to your project (`my_coral_segmentation`)
2. Click **Create a new task**
3. Configure the task:
    - **Name**: `train_batch_01` (use consistent naming)
    - **Subset**: Select **Train** from dropdown
    - **Project**: Should auto-select your project

### 2.2 Upload Images

1. Click **Select files** or drag and drop images
2. Select your coral images (supports JPG, PNG formats)
3. Recommended batch size: 20-50 images per task for easier management

!!! tip "Image Organization Best Practices"
    - Group images by location, time period, or similarity
    - Use descriptive task names: `train_batch_01`, `val_moorea_2023`, etc.
    - Create separate tasks for train/val/test subsets
    - Aim for 70% train, 20% validation, 10% test split

### 2.3 Configure Task Settings

Before submitting:

- **Image quality**: Keep at default (70%) unless images are very large
- **Overlap**: Set to `0` (not needed for segmentation)
- **Segment size**: Set to `1` (one job per task) for small batches
- **Start frame**: Leave at `0`

Click **Submit** to create the task.

### 2.4 Repeat for Validation and Test Sets

Create additional tasks following the same process:

- **Validation set**: Name tasks `val_batch_01`, `val_batch_02`, etc.
- **Test set**: Name tasks `test_batch_01`, etc.

## Step 3: Manual Annotation Workflow

Before setting up semi-automatic annotation, it's helpful to understand the manual annotation process.

### 3.1 Open Annotation Interface

1. Navigate to your task (e.g., `train_batch_01`)
2. Click the **Job #1** link to open the annotation interface
3. Familiarize yourself with the interface layout

### 3.2 Annotation Workflow

**Basic annotation steps:**

1. **Select label**: Click the desired genus from the right sidebar (or use keyboard shortcut `1-9`, `0`, etc.)
2. **Draw polyline**: Click around the coral colony boundary
3. **Close polyline**: Press `N` or double-click to finish
4. **Repeat**: Continue for all visible coral colonies
5. **Save**: Press `Ctrl+S` frequently to save progress

### 3.3 Quality Control Checklist

For each image, ensure:

- [x] All visible coral colonies are annotated
- [x] Polylines tightly follow coral boundaries
- [x] Overlapping corals are separated into individual instances
- [x] Genus labels are correct
- [x] Uncertain colonies are labeled as "Other"
- [x] Background (sand, rock) is not annotated

### 3.4 Common Mistakes to Avoid

!!! warning "Annotation Errors to Watch For"
    - **Merging colonies**: Don't draw one polyline around multiple separate colonies
    - **Including shadows**: Only annotate the coral tissue, not shadows
    - **Incorrect genus**: Use "Other" if uncertain rather than guessing
    - **Open polylines**: Always close the shape (press `N` or double-click)
    - **Too few points**: Use enough vertices to capture the colony boundary accurately

### 3.5 Keyboard Shortcuts

Speed up annotation with these shortcuts:

| Shortcut | Action |
|----------|--------|
| `N` | Finish current polyline |
| `Esc` | Cancel current annotation |
| `Ctrl+S` | Save work |
| `F` | Go to next frame (image) |
| `D` | Go to previous frame |
| `Del` | Delete selected shape |
| `1-9, 0` | Quick label selection |
| `Ctrl+Z` | Undo |
| `Ctrl+Shift+Z` | Redo |

!!! tip "Efficient Annotation Tips"
    - Use keyboard shortcuts to switch labels without clicking
    - Zoom in (`Ctrl+Wheel`) for precise boundaries on complex shapes
    - Annotate by genus (do all Acropora first, then all Porites, etc.)
    - Take breaks every 30 minutes to maintain accuracy

## Step 4: Semi-Automatic Annotation

Semi-automatic annotation dramatically speeds up the annotation process by using a pre-trained model to generate initial predictions that you then review and correct.

### 4.1 Deploy YOLO Segmentation Model

The YOLO coral segmentation model must be deployed as a Nuclio serverless function.

**If you've already deployed Nuclio models** during setup, the model should be available. Verify:

1. Open Nuclio dashboard at `http://localhost:8070`
2. Navigate to the `cvat` project
3. Look for function named `pth-yolo-coralsegv4` or similar

**If not deployed**, follow these steps:

```bash
# Navigate to the YOLO segmentation module
cd PROJ_ROOT/criobe/coral_seg_yolo

# Deploy the model (see deployment guide for details)
cd deploy
./deploy_as_zip.sh

# Deploy to Nuclio
nuctl deploy --project-name cvat \
    --path ./pth-yolo-coralsegv4-nuclio \
    --platform local \
    --verbose
```

Verify the deployment:

```bash
# Check function status
nuctl get functions --platform local

# Test the function
curl -X POST http://localhost:8000 \
    -H "Content-Type: application/json" \
    -d @test_payload.json
```

### 4.2 Configure Model in CVAT

1. Log in to CVAT as admin
2. Go to **Models** page (in top navigation)
3. Verify the deployed model appears in the list
4. Note the model name (e.g., `pth-yolo-coralsegv4`)

### 4.3 Set Up Detection Webhook

To run automatic pre-annotation when you open a job:

1. Navigate to your CVAT project
2. Click **Actions** → **Webhooks**
3. Click **Create webhook**
4. Configure webhook:
    - **Target URL**: `http://bridge:8000/detect-model-webhook?model_name=pth-yolo-coralsegv4&conv_mask_to_poly=true`
    - **Description**: "Auto-detect corals on job open"
    - **Events**: Check **"When a job state is changed to 'in progress'"**
    - **Content type**: `application/json`
5. Click **Submit**

!!! info "Webhook URL Parameters"
    - `model_name=pth-yolo-coralsegv4`: Specifies which Nuclio model to use
    - `conv_mask_to_poly=true`: Converts YOLO mask predictions to CVAT polylines

    If using a different model (e.g., Banggai model), change the model name:
    `model_name=pth-yolo-coralsegbanggai`

### 4.4 Run Automatic Pre-Annotation

Now when you open an annotation job, the model will automatically run:

1. Navigate to a task with unannotated images
2. Click **Job #1** to open it
3. Change job state to **"in progress"** to trigger the automatic detection
4. Wait 10-30 seconds (depending on image complexity)
5. **Refresh the page** (`F5`)
6. You should see coral polylines automatically added!

!!! tip "Monitoring Pre-Annotation"
    - Check the CVAT notification bell for webhook execution status
    - If nothing appears after 60 seconds, check:
        - Nuclio function logs: `nuctl get logs pth-yolo-coralsegv4`
        - Bridge service logs: `docker logs bridge`
        - Webhook configuration is correct

### 4.5 Manual Correction Workflow

After pre-annotation, review and correct the predictions:

**Quality assurance steps:**

1. **Check for missed corals**
    - Scan the entire image
    - Add any colonies the model missed (especially small or partially occluded ones)

2. **Correct boundaries**
    - Click on a polyline to select it
    - Drag control points to adjust boundaries
    - Add points by clicking on the line edge

3. **Fix misclassifications**
    - Select the polyline
    - Change the label using the right sidebar or keyboard shortcuts

4. **Remove false positives**
    - Delete polylines drawn on non-coral areas (rocks, sand, fish)
    - Press `Del` after selecting unwanted shapes

5. **Split merged colonies**
    - If the model merged multiple colonies into one polyline:
        - Delete the merged shape
        - Manually draw separate polylines for each colony

6. **Save frequently**
    - Press `Ctrl+S` after correcting each image

!!! warning "Model Limitations"
    Pre-trained models are helpful but not perfect. Expect to correct:

    - **Missed small colonies** (< 2cm diameter)
    - **Genus misclassifications** (especially similar genera like Pavona/Leptoseris)
    - **False positives** on complex backgrounds
    - **Boundary inaccuracies** on irregular colony shapes

    Always review every image carefully!

### 4.6 Batch Processing Tips

For large datasets:

1. **Process in batches**: Create tasks with 20-50 images each
2. **Run pre-annotation on all tasks**: Open each job briefly to trigger webhooks
3. **Return to correct**: After all pre-annotations complete, systematically correct each task
4. **Track progress**: Mark tasks as "completed" only after thorough review

## Step 5: Export and Prepare Data

Once annotation is complete, export the dataset for training.

### 5.1 Mark Tasks as Completed

Before exporting, mark all annotated tasks as complete:

1. Navigate to your project
2. For each task, click **Actions** → **Finish**
3. Verify the task status shows **Completed**

### 5.2 Pull Data Using FiftyOne

Use the `data_engineering` module to pull annotations from CVAT:

```bash
# Navigate to data engineering module
cd PROJ_ROOT/criobe/data_engineering

# Activate the Pixi environment
pixi shell

# Pull project annotations into FiftyOne
python create_fiftyone_dataset.py \
    --cvat-project-name "my_coral_segmentation" \
    --dataset-name "my_coral_segmentation_fo"
```

**What this does:**

- Connects to CVAT using credentials from `.env`
- Downloads all images and polygon annotations (converted to FiftyOne polylines)
- Creates a persistent FiftyOne dataset
- Stores data in `data/media/my_coral_segmentation/`

**Expected output:**

```
INFO: Connecting to CVAT at http://localhost:8080
INFO: Found project 'my_coral_segmentation' (ID: 5)
INFO: Downloading 120 images...
INFO: Processing annotations...
INFO: Created FiftyOne dataset 'my_coral_segmentation_fo' with 120 samples
INFO: Dataset stored in PROJ_ROOT/criobe/data_engineering/data/media/my_coral_segmentation
```

### 5.3 Verify Dataset in FiftyOne

Launch the FiftyOne app to verify your dataset:

```bash
# From within the pixi shell
fiftyone app launch my_coral_segmentation_fo
```

This opens a browser at `http://localhost:5151` showing:

- All images with annotations overlaid
- Label counts and distribution
- Interactive filtering and search

**Verification checklist:**

- [x] All images are present
- [x] Polylines are correctly loaded
- [x] Label distribution looks reasonable
- [x] No missing or corrupted annotations

### 5.4 Convert to YOLO Format

To train a YOLO model, convert the FiftyOne dataset:

```bash
# Navigate to YOLO module
cd PROJ_ROOT/criobe/coral_seg_yolo

# Activate environment
pixi shell -e coral-seg-yolo-dev

# Convert dataset
python src/prepare_data.py \
    --dataset-name my_coral_segmentation_fo \
    --output-dir data/prepared_for_training/my_dataset
```

**Output structure:**

```
data/prepared_for_training/my_dataset/
├── dataset.yaml          # YOLO dataset config
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

### 5.5 Convert to MMSegmentation Format

To train a DINOv2/MMSeg model:

```bash
# Navigate to DINOv2 module
cd PROJ_ROOT/criobe/DINOv2_mmseg

# Activate environment
pixi shell -e dinov2-mmseg

# Convert dataset
python prepare_data.py \
    --dataset-name my_coral_segmentation_fo \
    --output-dir data/prepared_for_training/my_dataset
```

**Output structure:**

```
data/prepared_for_training/my_dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── train/            # Semantic segmentation masks
│   ├── val/
│   └── test/
└── config.py             # MMSeg dataset config
```

### 5.6 Verification Commands

Verify the exported data:

**YOLO format:**

```bash
# Check dataset.yaml
cat data/prepared_for_training/my_dataset/dataset.yaml

# Count images and labels
find data/prepared_for_training/my_dataset/train/images -type f | wc -l
find data/prepared_for_training/my_dataset/train/labels -type f | wc -l
# Should show equal counts

# Verify label format
head -5 data/prepared_for_training/my_dataset/train/labels/sample_001.txt
```

**MMSeg format:**

```bash
# Count images and masks
find data/prepared_for_training/my_dataset/images/train -type f | wc -l
find data/prepared_for_training/my_dataset/annotations/train -type f | wc -l
# Should show equal counts

# Check mask values (should be 0-15 for 16 classes)
python -c "import cv2; import numpy as np; mask = cv2.imread('data/prepared_for_training/my_dataset/annotations/train/sample_001.png', 0); print(f'Unique values: {np.unique(mask)}')"
```

## Troubleshooting

### Common Issues

??? question "Model webhook doesn't trigger"
    **Possible causes:**

    - Bridge service not running: `docker ps | grep bridge`
    - Webhook URL incorrect: Verify it points to `http://bridge:8000`
    - Nuclio function not deployed: Check `http://localhost:8070`

    **Solutions:**

    1. Check bridge logs: `docker logs bridge`
    2. Test webhook manually:
        ```bash
        curl -X POST http://localhost:8000/detect-model-webhook?model_name=pth-yolo-coralsegv4&conv_mask_to_poly=true
        ```
    3. Verify network connectivity between CVAT and Bridge

??? question "Pre-annotations are low quality"
    **Possible causes:**

    - Model not trained on similar images
    - Image quality issues (blur, lighting, turbidity)
    - Different coral species distribution

    **Solutions:**

    - Consider the pre-annotations as a starting point
    - Increase manual correction effort
    - Train a custom model on your dataset (see [Training Guides](../training-and-deployment/index.md))

??? question "FiftyOne dataset pull fails"
    **Possible causes:**

    - CVAT credentials incorrect in `.env`
    - CVAT project name mismatch
    - Network connectivity issues

    **Solutions:**

    1. Check `.env` file:
        ```bash
        cat PROJ_ROOT/criobe/data_engineering/.env
        # Verify FIFTYONE_CVAT_URL, USERNAME, PASSWORD
        ```
    2. Test CVAT API manually:
        ```python
        from cvat_sdk import make_client
        client = make_client(host="http://localhost:8080", credentials=("admin", "password"))
        projects = client.projects.list()
        print([p.name for p in projects])
        ```

??? question "Exported labels are empty or incorrect"
    **Possible causes:**

    - Tasks not marked as completed
    - Annotations not saved in CVAT
    - Label field name mismatch

    **Solutions:**

    - Verify task status in CVAT project page
    - Re-open jobs and ensure annotations are visible
    - Check FiftyOne dataset in app: `fiftyone app launch <dataset-name>`

## Next Steps

Congratulations! You've completed the single-stage segmentation setup. You now have:

- ✅ A CVAT project with coral annotations
- ✅ A FiftyOne dataset for analysis
- ✅ Training-ready data in YOLO and/or MMSeg format

### Continue Learning

- **Train a YOLO model**: [YOLO Segmentation Training Guide](../training-and-deployment/yolo-segmentation.md)
- **Train a DINOv2 model**: [MMSeg Segmentation Training Guide](../training-and-deployment/mmseg-segmentation.md)
- **Learn multi-stage workflows**: [Two-Stage Banggai Setup](2-two-stage-banggai.md)

### Reference Materials

- [CVAT Label Templates](../reference/cvat-label-templates.md) - All label configurations
- [Coral Taxonomy Guide](../reference/cvat-label-templates.md#coral-taxonomies) - Genus identification help
- [FiftyOne Documentation](https://docs.voxel51.com/) - Dataset management

---

**Next Guide**: [Two-Stage Banggai Setup](2-two-stage-banggai.md) · [Back to Overview](index.md)
