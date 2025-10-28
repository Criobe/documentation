# Platform Setup: CVAT + Nuclio + Bridge

These instructions bootstrap the complete CRIOBE labeling environment on a single host. Work through the sections in order, then continue with the detailed Nuclio and bridge guides.

## Prerequisites
- Docker Engine or Docker Desktop installed and running.
- At least 40 GB free disk space for CVAT data, Docker images, and models.
- Git access to the [`Criobe/bridge`](https://github.com/Criobe/bridge) repository (fork of CVAT v2.29.0 with bridge integration).
- Ports `8080` (CVAT), `8070` (Nuclio dashboard), and `8000` (bridge) available locally.

## 1. Clone and Configure
1. Create a workspace folder (referred to as `PROJECT_ROOT`).
2. Clone the CVAT fork:
   ```bash
   git clone https://github.com/Criobe/bridge.git PROJECT_ROOT/cvat
   ```
3. Create `PROJECT_ROOT/cvat/bridge/.env` with credentials for the bridge service:
   ```bash
   CVAT_URL=http://cvat-server:8080
   CVAT_USER=admin
   CVAT_PWD=change_me
   WEBHOOK_SECRET=optional_shared_secret
   CACHE_DIR=/tmp/cvat_cache
   AUTO_ANN_TIMEOUT=900
   ```

## 2. Launch CVAT + Bridge Stack
From `PROJECT_ROOT/cvat`, build and start the required services:

```bash
# Build CVAT, bridge, and serverless workers
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d --build bridge cvat_server cvat_worker_webhooks

# Keep the rest of the stack running
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

Create the first administrative user:
```bash
docker exec -it cvat_server \
  bash -ic 'python3 ~/manage.py createsuperuser'
```
Then visit `http://localhost:8080/` and sign in with the credentials you set.

## 3. Add Nuclio Functions
Deploy the required serverless functions after CVAT is reachable. Follow the [Serverless setup guide](setup_nuclio.md) for platform-specific instructions. Ensure each function attaches to the `cvat_cvat` Docker network.

## 4. Configure Bridge Automation
Once Nuclio functions are available, wire CVAT projects to the bridge so tasks flow automatically. Use the [Bridge operations runbook](bridge/operations.md) to register webhooks and verify deliveries.

## 5. Daily Operation Tips
- Services auto-start when the host reboots (Linux) or Docker Desktop launches (Windows). To restart manually:
  ```bash
  docker compose -f docker-compose.yml \
    -f bridge/docker-compose.bridge.yml \
    -f components/serverless/docker-compose.serverless.yml \
    up -d
  ```
- To conserve resources when idle, stop the stack:
  ```bash
  docker compose -f docker-compose.yml \
    -f bridge/docker-compose.bridge.yml \
    -f components/serverless/docker-compose.serverless.yml \
    down
  ```

### Service Endpoints
- CVAT web UI: `http://localhost:8080/projects`
- Nuclio dashboard: `http://localhost:8070/projects`
- Bridge API (from host): `http://localhost:8000/health`

## Developer Notes
- Keep Git remotes aligned with upstream CVAT updates and reapply CRIOBE patches as needed.
- `docker compose logs -f bridge` is the quickest way to inspect automation events during integration testing.
- When modifying CVAT settings, replicate changes in both English and upcoming French documentation.
