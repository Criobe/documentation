# User Guide

Comprehensive guide for using the QUADRATSEG platform developed by CRIOBE. Whether you're annotating images, running inference, or analyzing results, this guide will help you accomplish your tasks.

## Guide Structure

This user guide is organized into practical workflow-oriented sections:

<div class="grid cards" markdown>

-   :material-database:{ .lg .middle } **Data Preparation**

    ---

    Create annotated datasets using CVAT and semi-automatic annotation workflows. Three guides with growing complexity.

    [:octicons-arrow-right-24: Prepare Data](data-preparation/index.md)

-   :material-brain:{ .lg .middle } **Training & Deployment**

    ---

    Train custom models and deploy them as Nuclio serverless functions for production use.

    [:octicons-arrow-right-24: Train Models](training-and-deployment/index.md)

-   :material-book-open-variant:{ .lg .middle } **Reference**

    ---

    CVAT label templates, webhook configurations, and taxonomy mappings for quick reference.

    [:octicons-arrow-right-24: View Reference](reference/cvat-label-templates.md)

-   :material-lightbulb:{ .lg .middle } **Concepts**

    ---

    Understand core concepts, pipeline stages, datasets, and workflows (coming soon).

    [:octicons-arrow-right-24: Learn Concepts](concepts/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step learning paths that teach you how to use the system (coming soon).

    [:octicons-arrow-right-24: Start Learning](tutorials/index.md)

-   :material-hammer-wrench:{ .lg .middle } **How-To Guides**

    ---

    Task-oriented guides for specific problems (coming soon).

    [:octicons-arrow-right-24: Find Solutions](how-to/index.md)

</div>

## Quick Reference

### Data Preparation Workflows

| Workflow | Complexity | Pipeline Stages | Guide |
|----------|------------|-----------------|-------|
| Single-Stage Segmentation | Low | Segmentation only | [Guide A](data-preparation/1-single-stage-segmentation.md) |
| Two-Stage Banggai | Medium | Corners → Segmentation | [Guide B](data-preparation/2-two-stage-banggai.md) |
| Three-Stage CRIOBE | High | Corners → Grid → Removal → Segmentation | [Guide C](data-preparation/3-three-stage-criobe.md) |

### Training & Deployment

| Module | Task | Guide |
|--------|------|-------|
| YOLO Segmentation | Fast coral instance segmentation | [YOLO Training](training-and-deployment/yolo-segmentation.md) |
| DINOv2 + MMSeg | Accurate semantic segmentation | [MMSeg Training](training-and-deployment/mmseg-segmentation.md) |
| Grid Detection | Corner & grid keypoint detection | [Grid Detection](training-and-deployment/grid-detection.md) |
| Grid Removal | Inpainting for grid removal | [Grid Removal](training-and-deployment/grid-removal.md) |
| Model Deployment | Deploy to Nuclio serverless | [Deployment Guide](training-and-deployment/model-deployment.md) |

### Reference Materials

| Resource | Description |
|----------|-------------|
| [CVAT Label Templates](reference/cvat-label-templates.md) | Copy-paste JSON configs for all pipeline stages |
| [Webhook Configurations](reference/cvat-label-templates.md#webhook-configurations) | Bridge service webhook examples |
| [Coral Taxonomies](reference/cvat-label-templates.md#taxonomy-hierarchies) | 16 genera to binary classification mappings |

## Learning Paths

Choose a learning path based on your role:

=== "Annotator / Data Manager"
    **Goal**: Create high-quality annotated datasets

    1. Start with [Single-Stage Segmentation](data-preparation/1-single-stage-segmentation.md) to learn CVAT basics
    2. Review [CVAT Label Templates](reference/cvat-label-templates.md) for all annotation types
    3. Set up [semi-automatic annotation](data-preparation/1-single-stage-segmentation.md#step-4-semi-automatic-annotation)
    4. Learn quality control and export workflows
    5. Progress to multi-stage pipelines if needed ([Guide B](data-preparation/2-two-stage-banggai.md) or [Guide C](data-preparation/3-three-stage-criobe.md))

=== "Researcher / Analyst"
    **Goal**: Use the platform for coral monitoring analysis

    1. Understand the [Data Preparation overview](data-preparation/index.md)
    2. Choose appropriate workflow for your images (Guides A/B/C)
    3. Set up complete pipeline with [webhook automation](data-preparation/2-two-stage-banggai.md#stage-2-automated-cropping-and-segmentation)
    4. Use pre-trained models for inference
    5. Explore custom model training if needed ([Training Guides](training-and-deployment/index.md))

=== "ML Engineer / Developer"
    **Goal**: Train and deploy custom coral segmentation models

    1. Complete a data preparation workflow ([Guide A](data-preparation/1-single-stage-segmentation.md) minimum)
    2. Review [Training & Deployment overview](training-and-deployment/index.md)
    3. Train models:
        - Start with [YOLO Segmentation](training-and-deployment/yolo-segmentation.md) for speed
        - Try [DINOv2 + MMSeg](training-and-deployment/mmseg-segmentation.md) for accuracy
        - Train grid models if needed ([Grid Detection](training-and-deployment/grid-detection.md))
    4. Deploy models with [Nuclio](training-and-deployment/model-deployment.md)
    5. Integrate into production pipelines

=== "System Administrator"
    **Goal**: Deploy and maintain the QUADRATSEG platform

    1. Complete [Installation guides](../setup/installation/index.md)
    2. Set up [CVAT with Nuclio](../setup/installation/for-end-users/2-ml-models-deployment.md)
    3. Deploy all [pre-trained models](training-and-deployment/model-deployment.md)
    4. Configure [webhook automation](reference/cvat-label-templates.md#webhook-configurations)
    5. Monitor and troubleshoot deployments

## Documentation Conventions

This guide uses consistent conventions:

!!! note "Informational Notes"
    Additional context or helpful information appears in blue boxes.

!!! tip "Pro Tips"
    Expert advice and best practices appear in green boxes.

!!! warning "Warnings"
    Important information that could prevent errors appears in orange boxes.

!!! danger "Critical Information"
    Actions that could cause data loss or system issues appear in red boxes.

### Code Block Conventions

```bash
# Commands to run in terminal start with $
$ pixi run python script.py

# Expected output is shown without $
INFO: Server started successfully
```

```python
# Python code examples use standard syntax highlighting
import fiftyone as fo
dataset = fo.load_dataset("criobe_finegrained_annotated")
```

### File Path Conventions

- **Absolute paths**: `/home/user/Projects/criobe/`
- **Relative paths**: `./data/test_samples/`
- **Module paths**: `<module-name>/path/to/file`

## Prerequisites

Before using this guide, ensure you have:

- [x] Completed [system installation](../setup/installation/index.md)
- [x] Configured [environment variables](../setup/configuration/environment-variables.md)
- [x] Access to CVAT instance (if using webhooks)

## Need Help?

!!! question "Can't Find What You Need?"
    - **Search**: Use the search bar at the top of this page
    - **Ask**: See [Getting Help](../community/getting-help.md)
    - **Contribute**: Help improve docs via [GitHub](https://github.com/criobe/coral-segmentation)

## What's Next?

- **Create datasets?** Start with [Data Preparation](data-preparation/index.md)
- **Train models?** Explore [Training & Deployment](training-and-deployment/index.md)
- **Need label configs?** Check [Reference Templates](reference/cvat-label-templates.md)
- **System setup?** See [Installation Guides](../setup/installation/index.md)

---

**Explore:** [Data Preparation](data-preparation/index.md) · [Training & Deployment](training-and-deployment/index.md) · [Reference](reference/cvat-label-templates.md)
