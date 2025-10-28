# Quickstart Guide

Get started with the QUADRATSEG platform—developed by CRIOBE—for coral segmentation in minutes. This section provides fast-track guides to help you see the system in action quickly.

## Available Quickstart Guides

<div class="grid cards" markdown>

-   :material-lightning-bolt:{ .lg .middle } **5-Minute Demo**

    ---

    Run a complete demo using pre-downloaded test data and see all pipeline stages in action.

    **Time Required**: 5-10 minutes

    [:octicons-arrow-right-24: Start Demo](5-minute-demo.md)

-   :material-file-document-edit:{ .lg .middle } **First Annotation**

    ---

    Create your first CVAT annotation project and run the automated pipeline.

    **Time Required**: 15-20 minutes

    [:octicons-arrow-right-24: Create Annotation](first-annotation.md)

</div>

## Prerequisites

Before starting, ensure you have:

- [ ] Python 3.9+ installed
- [ ] Pixi installed ([installation guide](https://pixi.sh/latest/#installation))
- [ ] Git installed
- [ ] NVIDIA GPU with CUDA support (recommended)

!!! warning "GPU Requirement"
    While the system can run on CPU, inference will be significantly slower. For production use, an NVIDIA GPU with at least 6GB VRAM is strongly recommended.

## What to Expect

The quickstart guides will walk you through:

1. **Environment Setup**: Installing dependencies using Pixi
2. **Model Download**: Getting pre-trained models
3. **Running Inference**: Processing sample images through the pipeline
4. **Viewing Results**: Understanding the outputs

## Next Steps

After completing the quickstart:

- **Understand the System**: Read [Pipeline Overview](../user-guide/concepts/pipeline-overview.md)
- **Full Installation**: Follow the [complete installation guide](../setup/installation/index.md)
- **Learn More**: Explore [tutorials](../user-guide/tutorials/index.md) for in-depth learning

## Quick Links

| Guide | Purpose | Difficulty |
|-------|---------|------------|
| [5-Minute Demo](5-minute-demo.md) | See the system in action | ⭐ Easy |
| [First Annotation](first-annotation.md) | Create annotation workflow | ⭐⭐ Moderate |

---

!!! tip "Getting Help"
    If you encounter issues during quickstart, see our [Getting Help](../community/getting-help.md) page or open an issue on GitHub.
