# Quickstart Guide

Get started with the QUADRATSEG platform—developed by CRIOBE—for automated coral reef monitoring and segmentation. Choose your path based on your role.

## Who Are You?

QUADRATSEG serves two distinct user types with different needs and setup paths:

<div class="grid cards" markdown>

-   :material-flask:{ .lg .middle } **Coral Researcher (End User)**

    ---

    You want to **process coral quadrat images** to get coverage metrics and species identification for analysis.

    **Your Goal**: Upload images → Get results

    **Setup Time**: 20-30 minutes

    [:octicons-arrow-right-24: Production Setup](production-setup.md)

-   :material-code-braces:{ .lg .middle } **AI Researcher / Developer**

    ---

    You want to **train models, experiment with architectures**, and iterate on the ML pipeline using new data.

    **Your Goal**: Continuous learning loop with model training

    **Setup Time**: 30-45 minutes

    [:octicons-arrow-right-24: Developer Setup](developer-setup.md)

</div>

## Setup Comparison

| Aspect | End User (Production) | Developer |
|--------|----------------------|-----------|
| **Installation** | Docker only (CVAT + Nuclio + Bridge) | Docker + Pixi environments |
| **Tools Needed** | Docker, Docker Compose | Docker, Pixi, Git, Python 3.9+ |
| **GPU Required** | Yes (for Nuclio functions) | Yes (for training + inference) |
| **Use Case** | Process images, get results | Train models, experiment, iterate |
| **Pixi Required** | ❌ No | ✅ Yes |
| **Complexity** | ⭐⭐ Moderate | ⭐⭐⭐ Advanced |

!!! info "Key Distinction: Pixi is for Development Only"
    **End users** (coral researchers) do **not need Pixi**. The production system runs entirely via Docker with models packaged in Nuclio serverless functions.

    **Developers** (AI researchers) need Pixi to train, evaluate, and experiment with models before packaging them for deployment.

## What is QUADRATSEG?

QUADRATSEG is an automated coral reef monitoring platform that processes underwater photo-quadrat images through a multi-stage ML pipeline:

1. **Grid Corner Detection** - Locate quadrat corners for perspective correction
2. **Image Warping** - Create standardized, perspective-corrected images
3. **Grid Pose Detection** - Detect all grid intersection points
4. **Grid Removal** - Inpaint grid lines to reveal clean coral imagery
5. **Coral Segmentation** - Identify and classify coral species with instance segmentation

## After Setup

Once you've completed your setup path:

**End Users**:

1. [Create your first annotation project](first-annotation.md)
2. Upload your coral quadrat images
3. Let the automated pipeline process them
4. Review and refine results in CVAT

**Developers**:

1. Explore the [pipeline architecture](../user-guide/concepts/pipeline-overview.md)
2. Follow [data preparation tutorials](../user-guide/tutorials/data-preparation.md)
3. Learn to [train custom models](../user-guide/tutorials/model-training.md)
4. Package models as [Nuclio functions](../user-guide/modules/bridge.md)

## Prerequisites by Role

### For End Users (Production Setup)

- [x] Docker and Docker Compose installed
- [x] NVIDIA GPU with CUDA support (8GB+ VRAM recommended)
- [x] NVIDIA Docker runtime installed
- [x] 20GB+ disk space for models and data

### For Developers (Development Setup)

- [x] All end-user prerequisites above
- [x] Pixi package manager ([install guide](https://pixi.sh/latest/#installation))
- [x] Python 3.9+ (managed by Pixi)
- [x] Git for version control
- [x] 50GB+ disk space for datasets, experiments, and model training

!!! warning "GPU Requirement"
    Both paths require an NVIDIA GPU with CUDA support. While CPU inference is possible, it will be 10-20x slower. For production use, a GPU with at least 8GB VRAM is strongly recommended.

## Getting Help

- **Documentation Issues**: Check [troubleshooting guides](../setup/deployment/docker-compose.md#troubleshooting)
- **Questions**: See [Getting Help](../community/getting-help.md)
- **Bug Reports**: Open an issue on [GitHub](https://github.com/criobe/coral-segmentation/issues)

---

**Ready to start?** Choose your setup guide above based on your role.
