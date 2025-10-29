# Data Preparation Guides

Learn how to prepare annotated datasets for training coral segmentation models using CVAT and semi-automatic annotation workflows.

## Overview

This section contains three practical guides with increasing complexity, each focusing on real-world workflows used for the CRIOBE coral monitoring platform. All guides emphasize **semi-automatic annotation** using pre-trained models deployed as Nuclio serverless functions.

<div class="grid cards" markdown>

-   :material-numeric-1-box:{ .lg .middle } **Single-Stage Segmentation**

    ---

    **Simplest workflow** for annotating pre-processed coral images

    **Use case**: Clean images ready for annotation (no grid, already cropped)

    **Stages**: Direct coral segmentation only

    [:octicons-arrow-right-24: Start Here](1-single-stage-segmentation.md)

-   :material-numeric-2-box:{ .lg .middle } **Two-Stage Banggai Setup**

    ---

    **Medium complexity** with perspective correction

    **Use case**: Raw quadrat images without grid overlays

    **Stages**: Corner detection → Cropping/warping → Segmentation

    [:octicons-arrow-right-24: Two-Stage Guide](2-two-stage-banggai.md)

-   :material-numeric-3-box:{ .lg .middle } **Three-Stage CRIOBE Setup**

    ---

    **Full pipeline** with grid removal

    **Use case**: Raw quadrat images with grid overlays

    **Stages**: Corner detection → Grid detection → Grid removal → Segmentation

    [:octicons-arrow-right-24: Complete Pipeline](3-three-stage-criobe.md)

</div>

## Guide Comparison

Choose the guide that matches your data preprocessing needs:

| Guide | Complexity | Input Images | Pipeline Stages | CVAT Projects | Nuclio Models | Time to Setup |
|-------|------------|--------------|-----------------|---------------|---------------|---------------|
| **A: Single-Stage** | Low | Pre-processed (clean) | 1 | 1 | 1 | ~30 min |
| **B: Two-Stage Banggai** | Medium | Raw (no grid) | 2 | 2 | 2 | ~1 hour |
| **C: Three-Stage CRIOBE** | High | Raw (with grid) | 4 | 4 | 4 | ~2 hours |

## What You'll Learn

All three guides teach you how to:

- **Create CVAT projects** with proper label configurations
- **Deploy ML models** as Nuclio serverless functions
- **Set up semi-automatic annotation** using model pre-annotation + manual correction
- **Configure webhooks** for pipeline automation
- **Export and prepare data** using FiftyOne for training

## Prerequisites

Before starting any guide, ensure you have:

- [x] **CVAT instance** running (see [Installation Guide](../../setup/installation/index.md))
- [x] **Nuclio serverless platform** deployed (see [ML Models Deployment](../../setup/installation/for-end-users/2-ml-models-deployment.md))
- [x] **Bridge service** configured (required for Guides B & C)
- [x] **Basic coral taxonomy knowledge** (see [Coral Taxonomies Reference](../reference/cvat-label-templates.md))

## Key Concepts

### Semi-Automatic Annotation Workflow

All guides use a **semi-automatic annotation approach** that combines:

1. **Model Pre-annotation**: Deploy pre-trained models to automatically detect/annotate features
2. **Manual Correction**: Human annotators review and correct model predictions
3. **Quality Assurance**: Verify annotations meet quality standards before export

This approach is **much faster** than manual annotation from scratch while maintaining high quality.

### CVAT Label Configurations

Each pipeline stage requires specific label configurations in CVAT:

- **Corner Detection**: 4-point skeleton for quadrat corners
- **Grid Detection**: 117-point skeleton for grid intersection points
- **Coral Segmentation**: 16 coral genera as polygon labels (in CVAT, imported as polylines in FiftyOne)

All configurations are provided as **copy-paste ready JSON templates** in the guides.

### Webhook Automation

Guides B and C use the **Bridge service** to automate data flow between pipeline stages:

- **Task Completion Webhooks**: Automatically process images when annotation tasks are marked complete
- **Model Detection Webhooks**: Run ML models when jobs are opened

This enables **end-to-end automation** from raw image upload to final annotated dataset.

## Learning Path

!!! tip "Recommended Learning Order"
    Even if you need the full three-stage pipeline, we recommend:

    1. **Start with Guide A** to understand CVAT basics and semi-automatic annotation
    2. **Move to Guide B** to learn webhook automation
    3. **Complete Guide C** for the full preprocessing pipeline

    This progressive approach builds your understanding step-by-step.

## What's Next?

After completing data preparation, proceed to training models:

- [Training and Deployment Guides](../training-and-deployment/index.md)
- [YOLO Segmentation Training](../training-and-deployment/yolo-segmentation.md)
- [DINOv2 Segmentation Training](../training-and-deployment/mmseg-segmentation.md)

## Reference Materials

- [CVAT Label Configuration Templates](../reference/cvat-label-templates.md) - All JSON templates in one place
- [Coral Taxonomy Reference](../reference/cvat-label-templates.md#coral-taxonomies) - 16 genera classification
- [Webhook Configuration Examples](../reference/cvat-label-templates.md#webhook-configurations) - Bridge service setup

---

**Choose Your Guide**: [Single-Stage](1-single-stage-segmentation.md) · [Two-Stage](2-two-stage-banggai.md) · [Three-Stage](3-three-stage-criobe.md)
