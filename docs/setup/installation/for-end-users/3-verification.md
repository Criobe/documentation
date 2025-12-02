# Installation Verification

Verify that the QUADRATSEG platform is correctly installed and ready for use.

!!! info "For End Users"
    This guide tests all components of your deployment to ensure everything works correctly before configuring projects and webhooks.

**Time Required**: 10-15 minutes
**Prerequisites**: [Docker deployment](1-docker-deployment.md) and [ML models deployment](2-ml-models-deployment.md) completed

## Verification Checklist

Use this checklist to verify your installation:

- [ ] All Docker containers running
- [ ] CVAT web UI accessible
- [ ] Nuclio dashboard accessible
- [ ] Bridge API accessible
- [ ] All 6 Nuclio functions deployed and ready
- [ ] GPU accessible from Nuclio functions
- [ ] Single inference test successful
- [ ] Webhook connectivity verified

## Step 1: Verify Docker Containers

Check that all containers are running:

```bash
cd ~/quadratseg-platform/cvat

# Check container status
docker compose ps

# All containers should show "Up" status
```

**Expected Output**:
```
NAME                        STATUS          PORTS
cvat_server                 Up              0.0.0.0:8080->8080/tcp
cvat_worker_annotation      Up
cvat_worker_webhooks        Up              # ← Critical for automation!
cvat_db                     Up              5432/tcp
cvat_redis                  Up              6379/tcp
cvat_opa                    Up              8181/tcp
nuclio_dashboard            Up              0.0.0.0:8070->8070/tcp
bridge                      Up              0.0.0.0:8000->8000/tcp
```

!!! warning "Critical Service"
    `cvat_worker_webhooks` must be running for automation to work! If it's not running, restart it:
    ```bash
    docker compose restart cvat_worker_webhooks
    ```

### Check Container Health

```bash
# Check if any containers are restarting
docker compose ps | grep -i "restarting"
# Should return nothing

# View logs for any problematic containers
docker compose logs --tail=50 cvat_server
docker compose logs --tail=50 bridge
```

## Step 2: Service Health Checks

Test each service endpoint to verify it's responding:

### CVAT Server

```bash
# Test CVAT API
curl http://localhost:8080/api/server/about

# Expected output (JSON):
# {
#   "name": "CVAT",
#   "description": "Computer Vision Annotation Tool",
#   "version": "..."
# }
```

**Access Web UI**:
- Open browser: http://localhost:8080
- Login with admin credentials
- Should see CVAT dashboard

### Nuclio Dashboard

```bash
# Test Nuclio health endpoint
curl http://localhost:8070/api/healthz

# Expected output:
# OK
```

**Access Dashboard**:
- Open browser: http://localhost:8070
- Navigate to: Projects → cvat
- Should see 6 deployed functions

### Bridge Service

```bash
# Test Bridge health endpoint
curl http://localhost:8000/health

# Expected output (JSON):
# {
#   "status": "healthy"
# }
```

**Access API Documentation**:
- Open browser: http://localhost:8000/docs
- Should see Swagger UI with available endpoints

## Step 3: Verify Nuclio Functions

Check that all ML models are deployed and ready:

```bash
# List all functions
nuctl get functions --platform local

# Expected: 6 functions, all with STATE=ready, REPLICAS=1/1
```

**Expected Output**:
```
NAMESPACE | NAME                              | PROJECT | STATE | NODE PORT | REPLICAS
nuclio    | pth-yolo-gridcorners             | cvat    | ready | 49152     | 1/1
nuclio    | pth-yolo-gridpose                | cvat    | ready | 49153     | 1/1
nuclio    | pth-lama                         | cvat    | ready | 49154     | 1/1
nuclio    | pth-yolo-coralsegv4              | cvat    | ready | 49155     | 1/1
nuclio    | pth-yolo-coralsegbanggai         | cvat    | ready | 49156     | 1/1
nuclio    | pth-mmseg-coralscopsegformer     | cvat    | ready | 49157     | 1/1
```

### Check Function Details

```bash
# Get details for grid corner detection function
nuctl get function pth-yolo-gridcorners --platform local

# Check function logs for errors
nuctl get logs pth-yolo-gridcorners --platform local --tail 20
```

!!! success "All Functions Ready"
    If all functions show `ready` state and `1/1` replicas, your ML models are correctly deployed!

## Step 4: GPU Verification

Verify that Nuclio functions have access to GPU:

```bash
# Check GPU usage
nvidia-smi

# Should show:
# - GPU temperature, utilization
# - Memory usage
# - Running processes (may include python/nuclio)
```

### Test GPU Access from Function

```bash
# Find container name for a function
docker ps | grep pth-yolo-gridcorners

# Test GPU access from within container
docker exec <container-name> nvidia-smi

# Expected: GPU information displayed
```

**Expected GPU Memory** (when idle):
- Each function reserves minimal memory when idle (~100-500MB)
- During inference, memory usage increases significantly

!!! tip "GPU Monitoring"
    Run `watch -n 1 nvidia-smi` to monitor GPU in real-time during inference tests.

## Step 5: Network Connectivity Test

Verify that CVAT can reach the Bridge service (required for webhooks):

```bash
# Test from CVAT server container to Bridge
docker exec cvat_server curl http://bridge.gateway:8000/health

# Expected output:
# {"status": "healthy"}
```

**If this fails**:
```bash
# Check if extra_hosts is configured
docker inspect cvat_server | grep -A 5 ExtraHosts

# Should show:
# "ExtraHosts": [
#     "bridge.gateway:host-gateway"
# ]
```

## Step 6: Single Inference Test

Test end-to-end inference with a sample image:

### Prepare Test Image

```bash
# Download a test image (or use your own)
wget https://storage.googleapis.com/criobe_public/test_samples/1-raw_jpg/sample.jpg -O test_image.jpg

# Or use any coral quadrat image you have
```

### Test Corner Detection

```bash
# Encode image as base64
IMAGE_B64=$(base64 -w0 test_image.jpg)

# Create payload
cat > payload.json <<EOF
{
  "image": "$IMAGE_B64"
}
EOF

# Get function port
PORT=$(nuctl get function pth-yolo-gridcorners --platform local -o json | grep -oP '"httpPort":\K\d+')

# Test inference
curl -X POST \
  -H "Content-Type: application/json" \
  -d @payload.json \
  http://localhost:$PORT

# Expected: JSON response with detected corners
```

**Expected Response** (simplified):
```json
{
  "predictions": [
    {"class": 0, "confidence": 0.95, "bbox": [x1, y1, x2, y2]},
    {"class": 0, "confidence": 0.93, "bbox": [x1, y1, x2, y2]},
    {"class": 0, "confidence": 0.91, "bbox": [x1, y1, x2, y2]},
    {"class": 0, "confidence": 0.89, "bbox": [x1, y1, x2, y2]}
  ]
}
```

### Monitor GPU During Inference

```bash
# In another terminal, monitor GPU
watch -n 1 nvidia-smi

# Run inference test
# You should see GPU memory spike during inference
```

!!! success "Inference Working"
    If you receive predictions in JSON format and GPU shows activity, inference is working correctly!

## Step 7: Webhook Connectivity Test

Test that CVAT can trigger Bridge webhooks:

### Create Test Webhook (Optional)

```bash
# Use CVAT API to test webhook
# Login and get token
TOKEN=$(curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"your_password"}' \
  | grep -oP '"key":"\K[^"]+')

# Create test project
curl -X POST http://localhost:8080/api/projects \
  -H "Authorization: Token $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name":"test_verification"}'

# Add webhook (via web UI is easier - see configuration guides)
```

**Manual Test**:
1. Open CVAT: http://localhost:8080
2. Create a test project
3. Go to: Actions → Setup webhooks
4. Add webhook: `http://bridge.gateway:8000/health`
5. Click "Ping"
6. Should show: "✅ Success (200)"

## Verification Summary

After completing all steps, you should have:

| Component | Status | Verification |
|-----------|--------|-------------|
| Docker Containers | ✅ Running | All services Up |
| CVAT Web UI | ✅ Accessible | http://localhost:8080 |
| Nuclio Dashboard | ✅ Accessible | http://localhost:8070 |
| Bridge API | ✅ Accessible | http://localhost:8000 |
| Nuclio Functions | ✅ Deployed | 6 functions ready |
| GPU Access | ✅ Available | nvidia-smi from containers |
| Inference | ✅ Working | Test predictions returned |
| Webhooks | ✅ Connected | Bridge reachable from CVAT |

!!! success "Installation Verified!"
    All components are working correctly! Your platform is ready for configuration.

## Troubleshooting

### Container Keeps Restarting

**Symptoms**: Container status shows "Restarting"

**Solutions**:
```bash
# Check container logs
docker compose logs <container-name>

# Common causes:
# - Configuration error (.env)
# - Port conflict
# - Insufficient resources

# Restart specific service
docker compose restart <container-name>

# Full restart
docker compose down
docker compose up -d
```

### Traefik or Nuclio: "Client Version Too Old" Error

**Symptoms**: Traefik or Nuclio containers fail to start with error:
```
Error response from daemon: client version 1.24 is too old. Minimum supported API version is 1.44
```

**Cause**: Using Docker 29.0.1+ without compatibility workaround.

**Solution**:
```bash
# 1. Edit Docker daemon configuration
sudo nano /etc/docker/daemon.json

# 2. Add this content:
# {
#   "min-api-version": "1.24"
# }

# 3. Restart Docker
sudo systemctl restart docker

# 4. Restart the platform
docker compose down
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

See [Docker Deployment Guide](1-docker-deployment.md) for detailed explanation.

### Service Not Responding

**Symptoms**: curl commands timeout or return "Connection refused"

**Solutions**:
```bash
# Check if service is actually running
docker compose ps <service-name>

# Check service logs
docker compose logs <service-name>

# Check port bindings
docker compose port <service-name> <port>

# Restart service
docker compose restart <service-name>
```

### Function Shows Error State

**Symptoms**: Function STATE=error or REPLICAS=0/1

**Solutions**:
```bash
# Check function logs
nuctl get logs <function-name> --platform local

# Common causes:
# - Out of GPU memory
# - Model download failed
# - Dependency error

# Delete and redeploy
nuctl delete function <function-name> --platform local
cd ~/quadratseg-platform/cvat/components/serverless/pytorch/<function-name>
nuctl deploy --project-name cvat --path . --file function.yaml --platform local -v
```

### GPU Not Accessible

**Symptoms**: nvidia-smi fails in container

**Solutions**:
```bash
# Verify NVIDIA Docker runtime installed
docker info | grep nvidia

# If not installed:
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Verify GPU accessible from host
nvidia-smi

# Restart Nuclio functions
nuctl delete project cvat --platform local
# Then redeploy all functions
```

### Inference Test Fails

**Symptoms**: curl returns errors or no predictions

**Solutions**:
```bash
# Check function logs
nuctl get logs <function-name> --platform local

# Verify GPU memory available
nvidia-smi

# Test with smaller image
convert test_image.jpg -resize 640x640 test_image_small.jpg
# Retry inference

# Check function status
nuctl get function <function-name> --platform local
```

### Webhook Test Fails

**Symptoms**: Ping shows connection error

**Solutions**:
```bash
# Verify bridge accessible from cvat_server
docker exec cvat_server curl http://bridge.gateway:8000/health

# Check extra_hosts configuration
docker inspect cvat_server | grep ExtraHosts

# Check Smokescreen configuration (firewall)
docker compose logs cvat_server | grep -i smokescreen

# Verify SMOKESCREEN_OPTS in .env:
# SMOKESCREEN_OPTS=--allow-address=172.17.0.1
```

## Performance Benchmarks

Expected performance on typical hardware (NVIDIA GTX 1070, 16GB RAM):

| Operation | Expected Time | Notes |
|-----------|--------------|-------|
| Container startup | 10-30s | All services combined |
| CVAT page load | 1-3s | First load may be slower |
| Single inference (corners) | 1-2s | Including network overhead |
| Single inference (coral seg) | 7-10s | Larger model, more complex |
| GPU memory (all functions idle) | ~4-6GB | Baseline usage |
| GPU memory (during inference) | +2-4GB | Per function actively running |

## Next Steps

!!! success "Installation Complete and Verified!"
    All components are installed, deployed, and verified. Your QUADRATSEG platform is ready!

**What's next**:

1. **[Configure CVAT Projects](../../configuration/for-end-users/1-cvat-projects.md)** - Create projects for your workflow
2. **[Configure Webhooks](../../configuration/for-end-users/2-webhooks-setup.md)** - Set up automation
3. **[Test Complete Workflow](../../configuration/for-end-users/3-workflow-testing.md)** - Run end-to-end pipeline

## Quick Reference

### Health Check Commands

```bash
# CVAT
curl http://localhost:8080/api/server/about

# Nuclio
curl http://localhost:8070/api/healthz

# Bridge
curl http://localhost:8000/health

# Functions
nuctl get functions --platform local

# GPU
nvidia-smi

# Containers
docker compose ps
```

### Restart Commands

```bash
# Restart all services
docker compose restart

# Restart specific service
docker compose restart bridge

# Full restart (down + up)
docker compose down && docker compose up -d
```

---

**Questions?** See [Getting Help](../../../community/index.md) or report issues on [GitHub](https://github.com/criobe/coral-segmentation/issues).
