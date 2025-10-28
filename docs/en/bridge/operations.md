# Bridge Operations

Use this runbook when enabling webhooks, validating daily processing, or running spot checks on the bridge service.

## Before You Start
- Bridge container is running and reachable at `http://bridge.gateway:8000/` from CVAT containers.
- Target project IDs for each pipeline stage are known.
- Nuclio functions are healthy in the `cvat` namespace.

## Register CVAT Webhooks
From a CVAT project, go to **Actions → Setup webhooks** and configure entries that match your workflow.

### Two-Stage Processing (Recommended)
1. **Corner ➜ Grid Detection**
   - Source: Corner annotation project
   - URL: `http://bridge.gateway:8000/crop-quadrat-and-detect-grid-webhook?target_proj_id=<grid_project_id>`
2. **Grid ➜ Coral Segmentation**
   - Source: Grid annotation project
   - URL: `http://bridge.gateway:8000/remove-grid-and-detect-corals-webhook?target_proj_id=<coral_project_id>`

### Direct Corner ➜ Coral
- Source: Corner annotation project
- URL: `http://bridge.gateway:8000/crop-quadrat-and-detect-corals-webhook?target_proj_id=<coral_project_id>`

### Manual Task Creation
- Source: Any project
- URL: `http://bridge.gateway:8000/crop-quadrat-and-create-new-task-webhook?target_proj_id=<target_project_id>`
- Result: Warped imagery only—no ML inference.

### Common Settings
- **Content-Type**: `application/json`
- **Active**: Checked
- **Events**: Enable both `task` and `job`
- **Ping**: Use the button to verify HTTP 200 before saving

## Operational Checks
- **Service health**  
  `curl http://127.0.0.1:8000/health`
- **Webhook dry-run**  
  ```bash
  curl -X POST "http://127.0.0.1:8000/test-webhook?target_proj_id=123&debug=true" \
    -H "Content-Type: application/json" \
    -d '{"event": "ping"}'
  ```
- **Monitor deliveries**  
  In CVAT’s webhook modal, inspect the deliveries table for HTTP status, latency, and error payloads.

## Rolling Updates
- Use `docker compose ... up -d --build bridge` to rebuild the service while keeping CVAT running.
- After upgrades, re-run the health check and fire a manual webhook ping to confirm connectivity.
