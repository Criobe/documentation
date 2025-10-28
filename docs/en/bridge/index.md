# CVAT Bridge Overview

The bridge service automates the flow of quadrat imagery between CVAT and Nuclio. It listens to webhooks created on CVAT projects and spins up follow-on tasks so manual reviewers only touch the steps that still need human judgment.

## What the Bridge Automates
- Warps raw quadrat frames once corner annotations are complete
- Runs grid detection and optional grid removal before model inference
- Launches new CVAT tasks that point to the cleaned imagery and fresh model output

## Pipeline Stages at a Glance
1. **Corner ➜ Grid Detection** (`/crop-quadrat-and-detect-grid-webhook`)  
   Crops quadrats, runs `gridpose`, and populates the grid annotation project you specify.
2. **Grid ➜ Coral Segmentation** (`/remove-grid-and-detect-corals-webhook`)  
   Removes grid lines with LaMa, runs `coralsegv4`, then creates review tasks in your coral project.
3. **Corner ➜ Coral (Direct)** (`/crop-quadrat-and-detect-corals-webhook`)  
   Skips grid detection when speed matters more than grid overlays.
4. **Manual Task Creation** (`/crop-quadrat-and-create-new-task-webhook`)  
   Produces warped imagery without triggering any models for ad-hoc workflows.

## Using These Docs
- Need to deploy or upgrade the service? See [Bridge Deployment](deployment.md).
- Configuring webhooks or validating daily operations? Follow [Bridge Operations](operations.md).
- Seeing blocked requests, failing Nuclio calls, or other edge cases? Start with [Troubleshooting](troubleshooting.md).

For the full infrastructure bootstrap (CVAT + Nuclio) refer to the dedicated [setup guide](../setup_cvat_with_nuclio_and_bridge.md).
