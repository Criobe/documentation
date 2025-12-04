# Module Documentation

This page provides links to detailed technical documentation for each module in the CRIOBE coral annotation and segmentation system.

## Pipeline Modules

### Stage 1 & 2: Grid Pose Detection

**Module**: `grid_pose_detection`

Handles corner detection and grid intersection point detection for quadrat images.

**Features**:

- Quadrat corner detection (4 points)
- Grid intersection point detection (117 points for 9×13 grids)
- YOLOv8 pose estimation models
- Nuclio serverless deployment

**Documentation**: [grid_pose_detection/README.md](https://github.com/taiamiti/criobe/grid_pose_detection/README.md)

### Stage 3: Grid Inpainting

**Module**: `grid_inpainting`

Automatically removes grid overlays from quadrat images using LaMa inpainting.

**Features**:

- LaMa (Large Mask Inpainting) model
- Grid line removal from keypoint annotations
- Mask generation from grid intersection points
- GPU-accelerated processing

**Documentation**: [grid_inpainting/README.md](https://github.com/taiamiti/criobe/grid_inpainting/README.md)

### Stage 4 (YOLO): Coral Segmentation

**Module**: `coral_seg_yolo`

Fast coral instance segmentation using YOLOv8.

**Features**:

- YOLOv8 instance segmentation
- 16 coral genera classification
- Fast inference for real-time annotation
- Nuclio deployment

**Documentation**: [coral_seg_yolo/README.md](https://github.com/taiamiti/criobe/coral_seg_yolo/README.md)

### Stage 4 (MMSeg): Coral Segmentation

**Module**: `DINOv2_mmseg`

High-accuracy coral semantic segmentation using DINOv2 + MMSegmentation.

**Features**:

- DINOv2 vision transformer backbone
- MMSegmentation framework
- Maximum accuracy for scientific analysis
- Multiple taxonomy levels (16 genera, 10 extended, 7 families, binary)

**Documentation**: [DINOv2_mmseg/README.md](https://github.com/taiamiti/criobe/DINOv2_mmseg/README.md)

### Bridge Service

**Module**: `bridge`

Orchestrates webhooks and automation between CVAT and processing modules.

**Features**:

- Webhook endpoints for CVAT events
- Automatic task creation and progression
- Image transformation pipelines (warping, grid removal)
- Multi-stage workflow orchestration

**Documentation**: [bridge/README.md](https://github.com/taiamiti/criobe/bridge/README.md)

## Additional Resources

### Data Engineering

**Module**: `data_engineering`

Tools for managing datasets, pulling annotations from CVAT, and preparing training data.

**Features**:

- FiftyOne dataset creation
- CVAT annotation export
- Dataset conversion utilities
- Training data preparation

### Test Data

Download test samples and example datasets:

- Test samples: [Google Cloud Storage - test_samples.zip](https://storage.googleapis.com/data_criobe/test_samples.zip)

---

**Back to**: [User Guide Index](../index.md) · [Reference Section](cvat-label-templates.md)
