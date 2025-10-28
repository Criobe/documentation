# Bridge Deployment

This guide walks through the configuration needed to build and run the bridge alongside CVAT and Nuclio. Complete the base platform setup from the [CVAT + Nuclio guide](../setup_cvat_with_nuclio_and_bridge.md) before continuing.

## Prerequisites
- CVAT stack running from the CRIOBE fork (v2.29.0) with Docker Compose.
- Nuclio dashboard reachable at `http://localhost:8070/`.
- The following Nuclio functions deployed on the `cvat_cvat` network: `pth-yolo-gridpose`, `pth-yolo-coralsegv4`, `pth-lama`.
- Target CVAT projects created for each pipeline stage (record their IDs for webhook parameters).

## Environment Configuration
Create `bridge/.env` with the credentials the service will use to call CVAT:

```bash
CVAT_URL=http://cvat-server:8080
CVAT_USER=admin
CVAT_PWD=your_admin_password

# Optional tuning
WEBHOOK_SECRET=your_webhook_secret
CACHE_DIR=/tmp/cvat_cache
AUTO_ANN_TIMEOUT=900
```

## Build and Run
Run the bridge sidecar alongside CVAT using the layered compose files:

```bash
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d --build bridge cvat_server cvat_worker_webhooks

docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

The first command rebuilds images; the second keeps the stack running. Compose files are evaluated in order, so later files can override service definitions from earlier ones.

## Network Access for CVAT
CVAT’s Smokescreen proxy blocks webhook targets on private subnets. Expose the bridge on the host and whitelist the gateway:

```yaml
services:
  bridge:
    command: uvicorn main:app --host 0.0.0.0 --port 8000
    ports:
      - "8000:8000"
  cvat_server:
    extra_hosts:
      - "bridge.gateway:host-gateway"
    environment:
      SMOKESCREEN_OPTS: "--allow-address=172.17.0.1"
      ALLOWED_HOSTS: "localhost,127.0.0.1,cvat-server,bridge.gateway"
  cvat_worker_webhooks:
    extra_hosts:
      - "bridge.gateway:host-gateway"
    environment:
      SMOKESCREEN_OPTS: "--allow-address=172.17.0.1"
```

Adjust the gateway IP if you are on Windows (`192.168.65.254`) or prefer to allow the entire Docker range using `--allow-range=172.17.0.0/16`.

## Nuclio Function Network
When deploying serverless functions via the Nuclio UI/CLI, explicitly set the network to match the stack:

```yaml
platform:
  attributes:
    network: cvat_cvat
```

Confirm with `docker inspect` that each function container attaches to the same network as the bridge. This ensures the bridge can invoke inference endpoints directly (e.g., `http://nuclio-nuclio-pth-yolo-gridpose:8080`).
