# Bridge Troubleshooting

Common symptoms and the fixes that keep the bridge connected to CVAT and Nuclio.

## CVAT Cannot Reach the Bridge
**Symptom:** Webhook deliveries show connection errors or time out.  
**Fix:** Ensure the bridge is exposed on the host (`ports: ["8000:8000"]`) and that CVAT containers resolve `bridge.gateway` to the Docker gateway.

```yaml
services:
  cvat_server:
    extra_hosts:
      - "bridge.gateway:host-gateway"
  cvat_worker_webhooks:
    extra_hosts:
      - "bridge.gateway:host-gateway"
```

On Linux the gateway IP is usually `172.17.0.1`; on Windows Docker Desktop use `192.168.65.254`. Validate inside the container:

```bash
getent hosts bridge.gateway
```

## Smokescreen Blocks Requests
**Symptom:** CVAT webhook pings fail with `403` or connection reset.  
**Fix:** Relax Smokescreen filtering for the gateway address (or subnet) and add the bridge host to Django `ALLOWED_HOSTS`.

```yaml
environment:
  SMOKESCREEN_OPTS: "--allow-address=172.17.0.1"
  ALLOWED_HOSTS: "localhost,127.0.0.1,cvat-server,bridge.gateway"
```

If several machines contribute, permit the whole range:

```yaml
SMOKESCREEN_OPTS: "--allow-range=172.17.0.0/16"
```

## Bridge Cannot Call CVAT
**Symptom:** Bridge logs show `DisallowedHost` when requesting `http://cvat_server:8080`.  
**Fix:** Use the service alias `http://cvat-server:8080/` and ensure `ALLOWED_HOSTS` in `cvat_server` includes `cvat-server`.

## Nuclio Requests Fail
**Symptom:** Bridge receives `Connection refused` when invoking model functions.  
**Checks:**
1. Functions run on the same Docker network (`network: cvat_cvat` in their Nuclio spec).
2. Endpoints use service names, e.g. `http://nuclio-nuclio-pth-yolo-gridpose:8080`.
3. Payload size is within Nuclio defaults; test manually:
   ```bash
   curl -s -H "Content-Type: application/json" --data @payload.json \
     http://nuclio-nuclio-pth-yolo-gridpose:8080
   ```

## Still Failing?
- Inspect bridge logs: `docker compose logs -f bridge`.
- Re-run `mkdocs build --strict` to confirm docs stay accurate after edits.
- Capture webhook delivery payloads from CVAT to reproduce locally via `test-webhook`.
