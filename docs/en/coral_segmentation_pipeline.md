# Coral Segmentation Pipeline

This overview explains how quadrat imagery flows from capture to reviewed annotations. CRIOBE quadrats include a physical grid that must be removed before running segmentation models, so the default pipeline uses three automated stages.

## Core Components
- **CVAT projects** for corner labels (`corner_annotation`), grid QA (`grid_annotation`), and final coral review (`coral_segmentation`).
- **Bridge service** that listens to CVAT webhooks and sequences downstream jobs.
- **Nuclio functions** providing inference (`pth-yolo-gridpose`, `pth-lama`, `pth-yolo-coralsegv4`) plus optional helpers (`gridcorners`, `coralsegbanggai`, `sam`).

## Required Nuclio Functions
Bridge webhooks only succeed when the paired Nuclio functions are deployed. From the CVAT checkout, use `nuctl` (Linux/macOS) or the Nuclio dashboard (Windows) to publish:

| Function | Purpose | Command snippet |
|----------|---------|-----------------|
| `pth-yolo-gridpose` | Detects grid intersections for CRIOBE quadrats | `nuctl deploy --project-name cvat --path "./serverless/pytorch/yolo/gridpose/nuclio/" --platform local -v` |
| `pth-lama` | Removes grid lines via inpainting | `nuctl deploy --project-name cvat --path "./serverless/pytorch/lama/nuclio/" --platform local -v` |
| `pth-yolo-coralsegv4` | Segments coral cover on gridless imagery | `nuctl deploy --project-name cvat --path "./serverless/pytorch/yolo/coralsegv4/nuclio/" --file "./serverless/pytorch/yolo/coralsegv4/nuclio/function.yaml" --platform local -v` |

Optional helpers:
- `gridcorners` – sanity-check corner annotations.
- `coralsegbanggai` / `coralscopsegformer` – alternative models for specific regions.
- `sam` – semi-automatic mask refinement.

Ensure each function attaches to the `cvat_cvat` Docker network (`platform.attributes.network: cvat_cvat`). Detailed platform-specific workflows live in [Serverless setup](setup_nuclio.md).

## Three-Stage Flow for CRIOBE Data
1. **Corner annotation complete** – CVAT fires a webhook; the bridge crops and warps each quadrat using the corner coordinates.
2. **Grid detection** – Warped tiles trigger `gridpose` to localize the physical grid overlay.
3. **Grid removal** – LaMa removes grid lines so coral pixels remain unobstructed for segmentation.
4. **Coral segmentation** – `coralsegv4` infers masks and detections; CVAT creates review-ready tasks.

The first three steps convert raw images into gridless warped imagery tailored for CRIOBE’s dataset.

## Alternative Two-Stage Flow
For imagery without grids (e.g., other field campaigns), skip grid detection/removal and run:
1. Corner annotation ➜ warp imagery.
2. Coral segmentation ➜ create review tasks.

Use the `/crop-quadrat-and-detect-corals-webhook` endpoint for this streamlined path.

## CVAT Webhook Setup
Configure webhooks per project by navigating to **Actions → Setup webhooks**.

### Stage 1 – Corner ➜ Grid Detection
- **Source project**: Corner annotation project.
- **URL**: `http://bridge.gateway:8000/crop-quadrat-and-detect-grid-webhook?target_proj_id=<grid_project_id>`
- **Content type**: `application/json`
- **Events**: Enable both `task` and `job`
- **Ping**: Ensure you receive HTTP 200 before saving.

### Stage 2 – Grid ➜ Coral Segmentation
- **Source project**: Grid annotation project.
- **URL**: `http://bridge.gateway:8000/remove-grid-and-detect-corals-webhook?target_proj_id=<coral_project_id>`
- Same settings as Stage 1.

### Direct Pipeline (No Grid)
- **Source project**: Corner annotation project.
- **URL**: `http://bridge.gateway:8000/crop-quadrat-and-detect-corals-webhook?target_proj_id=<coral_project_id>`

### Manual Task Creation
- **Source project**: Any project.
- **URL**: `http://bridge.gateway:8000/crop-quadrat-and-create-new-task-webhook?target_proj_id=<target_project_id>`
- Produces warped imagery only—no models invoked.

Use CVAT’s deliveries table to verify status codes and to retry failed payloads when Nuclio or the bridge is offline.

For step-by-step bridge configuration, see the dedicated [Bridge operations guide](bridge/operations.md) and the broader [Bridge overview](bridge/index.md).

## What Happens After Review
Once tasks are segmented and approved in CVAT, export datasets to support:
1. **Model development** – Prepare annotations for training or fine-tuning segmentation models.
2. **Coral coverage analysis** – Feed reviewed masks into external applications that compute coverage metrics or long-term monitoring dashboards.

Maintain separate export presets for modeling (full-resolution masks, optional metadata) and analytics (aggregated per-quadrat outputs).

## Operational Reminders
- Deploy Nuclio functions before enabling webhooks to avoid failed deliveries.
- Keep the source quadrat project untouched for provenance; derived tasks carry metadata linking back to the parent task ID.
- Document any new endpoints or custom steps in the bridge operations guide so reviewers understand how tasks arrive.
