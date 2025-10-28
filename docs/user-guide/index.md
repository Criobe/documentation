# User Guide

Comprehensive guide for using the CRIOBE coral segmentation pipeline. Whether you're annotating images, running inference, or analyzing results, this guide will help you accomplish your tasks.

## Guide Structure

This user guide is organized into four sections to match different learning and working styles:

<div class="grid cards" markdown>

-   :material-lightbulb:{ .lg .middle } **Concepts**

    ---

    Understand core concepts, pipeline stages, datasets, and workflows before diving into practical tasks.

    [:octicons-arrow-right-24: Learn Concepts](concepts/index.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Step-by-step learning paths that teach you how to use the system from start to finish.

    [:octicons-arrow-right-24: Start Learning](tutorials/index.md)

-   :material-hammer-wrench:{ .lg .middle } **How-To Guides**

    ---

    Task-oriented guides that help you solve specific problems and accomplish real-world tasks.

    [:octicons-arrow-right-24: Find Solutions](how-to/index.md)

-   :material-package-variant:{ .lg .middle } **Modules**

    ---

    Detailed reference for each module in the pipeline with command examples and options.

    [:octicons-arrow-right-24: Browse Modules](modules/index.md)

</div>

## Quick Reference

### Common Tasks

| Task | Guide Type | Link |
|------|------------|------|
| Understand the pipeline | Concept | [Pipeline Overview](concepts/pipeline-overview.md) |
| Run complete workflow | Tutorial | [Complete Pipeline](tutorials/complete-pipeline.md) |
| Annotate in CVAT | How-To | [Annotate in CVAT](how-to/annotate-in-cvat.md) |
| Run inference | How-To | [Run Inference](how-to/run-inference.md) |
| Train custom model | Tutorial | [Model Training](tutorials/model-training.md) |
| Manage datasets | How-To | [Manage Datasets](how-to/manage-datasets.md) |

### By Module

| Module | Purpose | Documentation |
|--------|---------|---------------|
| Bridge | CVAT webhook automation | [Bridge Module](modules/bridge.md) |
| Data Engineering | Dataset management | [Data Engineering](modules/data-engineering.md) |
| Grid Pose Detection | Grid keypoint detection | [Grid Detection](modules/grid-pose-detection.md) |
| Grid Inpainting | Grid removal | [Grid Inpainting](modules/grid-inpainting.md) |
| YOLO Segmentation | Fast coral segmentation | [YOLO Module](modules/coral-seg-yolo.md) |
| DINOv2 Segmentation | Accurate segmentation | [DINOv2 Module](modules/dinov2-mmseg.md) |

## Learning Paths

Choose a learning path based on your role:

=== "Annotator"
    1. Read [Pipeline Overview](concepts/pipeline-overview.md)
    2. Learn [CVAT annotation best practices](how-to/annotate-in-cvat.md)
    3. Understand [coral taxonomies](concepts/taxonomies.md)
    4. Practice [exporting annotations](how-to/export-annotations.md)

=== "Researcher / Analyst"
    1. Understand [datasets and formats](concepts/datasets.md)
    2. Follow [Complete Pipeline Tutorial](tutorials/complete-pipeline.md)
    3. Learn to [run inference](how-to/run-inference.md)
    4. Explore [visualization tools](how-to/visualize-results.md)
    5. Try [training custom models](tutorials/model-training.md)

=== "ML Engineer"
    1. Review all [concepts](concepts/index.md)
    2. Complete [data preparation tutorial](tutorials/data-preparation.md)
    3. Work through [model training tutorial](tutorials/model-training.md)
    4. Learn [model evaluation](tutorials/model-evaluation.md)
    5. Explore [module documentation](modules/index.md)

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

- **New to the system?** Start with [Concepts](concepts/index.md)
- **Ready to try?** Jump to [Tutorials](tutorials/index.md)
- **Specific task?** Check [How-To Guides](how-to/index.md)
- **Technical details?** Browse [Modules](modules/index.md)

---

**Explore:** [Concepts](concepts/index.md) · [Tutorials](tutorials/index.md) · [How-To](how-to/index.md) · [Modules](modules/index.md)
