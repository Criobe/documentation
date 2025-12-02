# Docker Deployment

Deploy the QUADRATSEG platform stack: CVAT + Nuclio + Bridge for automated coral reef image processing.

!!! info "For End Users"
    This guide is for **coral researchers** deploying the platform for production use. No Pixi or code access needed—everything runs in Docker containers.

**Time Required**: 30-45 minutes
**Target Users**: Coral researchers, marine biologists, reef monitoring teams

## What You'll Deploy

```mermaid
graph LR
    A[CVAT<br/>Web Interface] --> B[Bridge<br/>Automation Service]
    A --> C[PostgreSQL<br/>Database]
    A --> D[Redis<br/>Cache]
    B --> E[Nuclio<br/>Serverless Platform]
    E --> F[ML Models<br/>Deployed Later]

    style A fill:#4CAF50
    style B fill:#FF9800
    style E fill:#2196F3
```

- **CVAT**: Web-based annotation and task management platform
- **Nuclio**: Serverless platform for hosting ML models
- **Bridge**: FastAPI service that orchestrates the pipeline via webhooks
- **PostgreSQL**: Database for CVAT data
- **Redis**: Cache for CVAT workers

**ML models** will be deployed separately in the [next guide](2-ml-models-deployment.md).

## Prerequisites

Before starting, ensure you have:

### System Requirements

- [x] **Operating System**: Linux (Ubuntu 20.04+ recommended) or macOS
- [x] **Docker**: Version 20.10+ ([install guide](https://docs.docker.com/engine/install/))
- [x] **Docker Compose**: Version 2.0+ (included with Docker Desktop)
- [x] **NVIDIA GPU**: 8GB+ VRAM recommended (GTX 1070 or better)
- [x] **NVIDIA Docker Runtime**: For GPU acceleration ([install guide](https://github.com/NVIDIA/nvidia-docker))
- [x] **Disk Space**: 20GB+ free space for Docker images and data
- [x] **Memory**: 16GB+ RAM recommended
- [x] **Git**: For cloning repository

### Verify Prerequisites

```bash
# Check Docker
docker --version
# Expected: Docker version 20.10.0 or higher

# Check Docker Compose
docker compose version
# Expected: Docker Compose version v2.0.0 or higher

# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi
# Expected: GPU information displayed

# Check available disk space
df -h
# Ensure at least 20GB free in Docker's storage location
```

!!! warning "GPU Required"
    While the platform can run without GPU, inference will be extremely slow. A GPU with 8GB+ VRAM is **strongly recommended** for production use.

!!! danger "IMPORTANT: Docker 29+ Users"
    If you're using Docker version 29.0.1 or higher, you **must** apply a compatibility workaround before deployment, otherwise Traefik and Nuclio will fail to start.

    **Quick Check**:
    ```bash
    docker --version
    # If version is 29.0.0 or higher, apply the workaround below
    # Docker 27.x and earlier work without workaround
    ```

    **Workaround** (required for Docker 29+):

    1. Create or edit `/etc/docker/daemon.json`:
       ```bash
       sudo nano /etc/docker/daemon.json
       ```

    2. Add the following content:
       ```json
       {
         "min-api-version": "1.24"
       }
       ```

    3. Restart Docker:
       ```bash
       sudo systemctl restart docker
       ```

    4. Verify Docker is running:
       ```bash
       docker ps
       ```

    **Why is this needed?** CVAT 2.29.0 includes Traefik with Docker client API v1.24. Docker 29+ requires API v1.44+ by default, causing the error: "client version 1.24 is too old".

    **Tested Configurations**:

    - ✅ Ubuntu 22.04, CVAT 2.29.0, Docker 27.3.1 (no workaround needed)
    - ✅ Ubuntu 22.04, CVAT 2.29.0, Docker 29.0.1 (with workaround)

## Step 1: Clone Repository

Clone CRIOBE's CVAT fork which includes the bridge service and serverless components. **Use the criobe branch**:

```bash
# Create directory for the platform
mkdir -p ~/quadratseg-platform
cd ~/quadratseg-platform

# Clone CRIOBE's CVAT repository (criobe branch)
git clone -b criobe https://github.com/criobe/cvat.git
cd cvat

# Verify you're on the criobe branch
git branch
# Should show: * criobe

# Verify repository structure
ls -la
# Should see: bridge/, components/, docker-compose.yml, etc.
```

!!! warning "Critical: Use criobe Branch"
    The CRIOBE platform **requires** the `criobe` branch, **not** `main`. The criobe branch contains custom integrations with the bridge service and Nuclio serverless components that are essential for automated pipeline processing.

**Repository Structure** (on criobe branch):
```
cvat/  (criobe branch)
├── bridge/                      # Bridge automation service
│   ├── docker-compose.bridge.yml
│   └── .env.example
├── components/
│   └── serverless/              # Nuclio serverless functions
│       └── docker-compose.serverless.yml
├── docker-compose.yml           # Main CVAT deployment
└── ...                          # CVAT core files
```

## Step 2: Deploy Basic CVAT

Deploy CVAT using the official pre-built images for version 2.29.0:

```bash
# Deploy CVAT 2.29.0
CVAT_VERSION=v2.29.0 docker compose up -d

# Check deployment status
docker compose ps
```

**Expected Output**:
```
[+] Running 10/10
 ✔ Network cvat_cvat                  Created
 ✔ Volume cvat_cvat_data              Created
 ✔ Volume cvat_cvat_keys              Created
 ✔ Container cvat_redis               Started
 ✔ Container cvat_db                  Started
 ✔ Container cvat_opa                 Started
 ✔ Container cvat_server              Started
 ✔ Container cvat_worker_annotation   Started
 ✔ Container cvat_worker_webhooks     Started
 ...
```

**Deployment Time**: 3-5 minutes (pulling images and starting services)

## Step 3: Create Superuser

Create the admin user for CVAT. These credentials will be used to configure the Bridge service in the next step.

```bash
# Create superuser (interactive)
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

You'll be prompted to enter:
- **Username**: Choose a username (e.g., `admin`)
- **Email address**: Your email address
- **Password**: Choose a secure password
- **Password (again)**: Confirm password

**Expected Output**:
```
Username: admin
Email address: admin@example.com
Password:
Password (again):
Superuser created successfully.
```

!!! important "Remember These Credentials"
    Save the username and password you just created. You'll need them in the next step to configure the Bridge service.

## Step 4: Configure Bridge

Configure the Bridge service to connect to CVAT using the superuser credentials from Step 3:

```bash
# Navigate to bridge directory
cd bridge

# Copy example configuration
cp .env.example .env

# Edit bridge configuration
nano .env
```

**Bridge Configuration** (in `bridge/.env`):

```bash
# CVAT Connection
CVAT_URL=http://cvat_server:8080
CVAT_USERNAME=admin  # Use the username from Step 3
CVAT_PASSWORD=your_password_here  # Use the password from Step 3

# Nuclio Connection
NUCLIO_HOST=nuclio
NUCLIO_PORT=8070

# Bridge Server
BRIDGE_PORT=8000

# Cache and Temporary Files
CACHE_DIR=/tmp/cvat_cache
AUTO_ANN_TIMEOUT=900  # Timeout for auto-annotation in seconds

# Logging
LOG_LEVEL=INFO
```

!!! tip "Use Superuser Credentials"
    Ensure `CVAT_USERNAME` and `CVAT_PASSWORD` match the superuser credentials you created in Step 3.

**Return to CVAT root directory:**

```bash
cd ..
pwd
# Should show: ~/quadratseg-platform/cvat
```

## Step 5: Deploy Bridge and Nuclio

Now deploy the Bridge automation service and Nuclio serverless platform to work with your running CVAT instance:

### 5.1 Build and Deploy Bridge

The Bridge service must be built locally on first deployment. We also rebuild cvat_server and cvat_worker_webhooks to ensure proper integration:

```bash
# Build bridge and rebuild CVAT components
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d --build bridge cvat_server cvat_worker_webhooks

# Start all remaining services
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d

# Check all services are running
docker compose ps
```

**Expected Output**:
```
[+] Running 12/12
 ✔ Container nuclio_dashboard         Started
 ✔ Container bridge                   Started
 ✔ Container cvat_server              Started
 ✔ Container cvat_worker_webhooks     Started
 ...
```

**Deployment Time**: 3-5 minutes (building bridge image and restarting CVAT components)

### 5.2 Subsequent Deployments

After the first deployment, use this command (no build needed):

```bash
# Subsequent deployments: use pre-built images
docker compose \
  -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

!!! success "Image Strategy"
    - **CVAT**: Uses official pre-built images (pulled from Docker Hub)
    - **Nuclio**: Uses official pre-built images (pulled from Docker Hub)
    - **Bridge**: Built locally from source (only needs building once)

## Step 6: Verify Services

Check that all services are running correctly:

```bash
# Check container status
docker compose ps

# Expected output: All containers should show "Up" status
# Key services to verify:
# - cvat_server (port 8080)
# - nuclio_dashboard (port 8070)
# - bridge (port 8000)
# - cvat_db (PostgreSQL)
# - cvat_redis
# - cvat_worker_webhooks (important for automation!)
```

### Service Health Checks

Test each service endpoint:

```bash
# Test CVAT
curl http://localhost:8080/api/server/about
# Expected: JSON response with CVAT version

# Test Nuclio Dashboard
curl -I http://localhost:8070
# Expected: "OK"

# Test Bridge
curl http://localhost:8000/health
# Expected: {"status": "healthy"}
```

!!! tip "All Services Healthy"
    If all three commands return successful responses, your platform is deployed correctly!

## Step 7: Access CVAT

Your CVAT instance is now ready!

1. **Open Browser**: Navigate to http://localhost:8080
2. **Login**: Use the superuser credentials from Step 3
    - Username: (your chosen username)
    - Password: (your chosen password)
3. **Explore Interface**:
    - **Projects**: Create and manage annotation projects
    - **Tasks**: Upload images and create annotation tasks
    - **Jobs**: Annotate images and review results

!!! success "Deployment Complete!"
    CVAT + Nuclio + Bridge are now deployed and ready for use!

## Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **CVAT Web UI** | http://localhost:8080 | Main annotation interface |
| **Nuclio Dashboard** | http://localhost:8070 | Serverless function management |
| **Bridge API** | http://localhost:8000/docs | API documentation (Swagger) |
| **Bridge Health** | http://localhost:8000/health | Health check endpoint |

## Common Management Commands

### Start/Stop Services

```bash
# Start services
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d

# Stop services
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  down

# Stop and remove volumes (WARNING: deletes all data!)
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  down -v
```

### View Logs

```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f cvat_server
docker compose logs -f bridge
docker compose logs -f nuclio_dashboard

# View last 100 lines
docker compose logs --tail=100 cvat_server
```

### Restart Specific Service

```bash
# Restart bridge service
docker compose restart bridge

# Restart CVAT server
docker compose restart cvat_server

# Restart webhook worker (important for automation!)
docker compose restart cvat_worker_webhooks
```

## Troubleshooting

### Traefik/Nuclio "client version too old" Error

**Symptoms**: Container logs show errors like:
```
traefik | {"level":"error","msg":"Failed to retrieve information of the docker client and server host: Error response from daemon: client version 1.24 is too old. Minimum supported API version is 1.44..."}
```

or

```
nuclio | Error response from daemon: client version 1.42 is too old
```

**Cause**: Docker 29+ requires minimum API version 1.44, but CVAT 2.29.0 uses older clients.

**Solution**: Apply the Docker daemon configuration workaround (see warning box at top of this guide).

### Services Not Starting

**Symptoms**: Containers exit immediately or fail to start

**Solutions**:
```bash
# Check logs for errors
docker compose logs cvat_server
docker compose logs bridge

# Common issues:
# 1. Port conflicts (8080, 8070, 8000 already in use)
# 2. Insufficient permissions
# 3. Missing environment variables

# Check port usage
sudo netstat -tulpn | grep -E "8080|8070|8000"

# Rebuild if needed
docker compose down
docker compose up -d --build bridge cvat_server cvat_worker_webhooks
```

### Cannot Access CVAT Web UI

**Symptoms**: Browser shows "Connection refused" or "Site can't be reached"

**Solutions**:
```bash
# Verify cvat_server is running
docker compose ps cvat_server

# Check cvat_server logs
docker compose logs cvat_server

# Verify network
docker network ls
docker network inspect cvat_cvat

# Try accessing from container
docker exec cvat_server curl http://localhost:8080
```

### Bridge Not Accessible from CVAT

**Symptoms**: Webhooks fail with connection errors

**Solutions**:
```bash
# Verify bridge is running
docker compose ps bridge

# Test bridge from cvat_server container
docker exec cvat_server curl http://bridge.gateway:8000/health

# Check if extra_hosts is configured
docker inspect cvat_server | grep -A 5 ExtraHosts
# Should show: bridge.gateway:host-gateway

# If missing, ensure docker-compose.bridge.yml has:
# services:
#   cvat_server:
#     extra_hosts:
#       - "bridge.gateway:host-gateway"
```

### GPU Not Available

**Symptoms**: Nuclio functions deployed but inference is slow

**Solutions**:
```bash
# Verify GPU accessible in Docker
docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu20.04 nvidia-smi

# Check NVIDIA Docker runtime installed
docker info | grep nvidia

# If not installed, install nvidia-docker2:
# Ubuntu/Debian:
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Database Connection Errors

**Symptoms**: CVAT shows database errors

**Solutions**:
```bash
# Check database container
docker compose ps cvat_db

# View database logs
docker compose logs cvat_db

# Connect to database
docker exec -it cvat_db psql -U root -d cvat

# Reset database (WARNING: deletes all data!)
docker compose down -v
docker compose up -d
```

## Security Considerations

!!! warning "Production Deployment"
    If deploying for production use beyond localhost:

    1. **Use strong passwords** for superuser and bridge configuration
    2. **Configure SSL/TLS** for HTTPS access
    3. **Set up firewall rules** to restrict access
    4. **Use secrets management** for sensitive credentials
    5. **Regular database backups**
    6. **Monitor resource usage** and logs

## Next Steps

!!! success "Platform Deployed!"
    You've successfully deployed the QUADRATSEG platform stack!

**What's next**:

1. **[Deploy ML Models](2-ml-models-deployment.md)** - Deploy Nuclio serverless functions with ML models
2. **[Verify Installation](3-verification.md)** - Run verification tests
3. **[Configure Projects](../../configuration/for-end-users/1-cvat-projects.md)** - Set up CVAT projects for your workflow

## Quick Reference

### Deploy CVAT (Step 2)

```bash
CVAT_VERSION=v2.29.0 docker compose up -d
```

### Create Superuser (Step 3)

```bash
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

### Deploy Bridge and Nuclio (First Time - Step 5)

```bash
# Build bridge and rebuild CVAT components
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d --build bridge cvat_server cvat_worker_webhooks

# Start all remaining services
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

### Deploy Bridge and Nuclio (Subsequent)

```bash
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  up -d
```

### Stop Command

```bash
docker compose -f docker-compose.yml \
  -f bridge/docker-compose.bridge.yml \
  -f components/serverless/docker-compose.serverless.yml \
  down
```

---

**Questions?** Check the [verification guide](3-verification.md) or see [Getting Help](../../../community/index.md).
