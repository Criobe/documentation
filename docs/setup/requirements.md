# System Requirements

Review hardware, software, and network requirements before installing the QUADRATSEG platform developed by CRIOBE (Centre de Recherches Insulaires et Observatoire de l'Environnement).

## Hardware Requirements

### Minimum Requirements (Development/Testing)

| Component | Specification |
|-----------|---------------|
| **CPU** | 4 cores, 2.0 GHz |
| **RAM** | 16 GB |
| **Storage** | 100 GB available (SSD recommended) |
| **GPU** | NVIDIA GTX 1070 (8 GB VRAM) |
| **Network** | Stable connection for downloads |

### Recommended Requirements (Desktop Production)

| Component | Specification |
|-----------|---------------|
| **CPU** | 8+ cores, 3.0 GHz |
| **RAM** | 32 GB+ |
| **Storage** | 500 GB+ NVMe SSD |
| **GPU** | NVIDIA RTX 3090 or RTX 4090 (24 GB VRAM) |
| **Network** | Reliable broadband for model downloads |

The platform currently targets a single high-end desktop with ample VRAM (16 GB minimum, 24 GB ideal) for local research use.

### Storage Breakdown

| Component | Approximate Size |
|-----------|------------------|
| Docker images | 15-20 GB |
| CVAT data & database | 10-50 GB (grows with annotations) |
| ML models (all modules) | 5-10 GB |
| Test samples | 500 MB |
| Working datasets | Variable (10-100+ GB) |
| **Total Minimum** | **30-80 GB** |
| **Recommended** | **200+ GB** |

## Software Requirements

### Operating System

| OS | Status | Notes |
|----|--------|-------|
| **Ubuntu 22.04 LTS** | ✅ Primary | Training and inference tested with RTX 4090 |
| **Windows 11 (Docker Desktop + WSL2)** | ⚠️ Deployment-only | Tested for running Nuclio containers; coding workflows unverified |

!!! tip "Stick to Ubuntu for development"
    All authoring, training, and inference workflows were executed on Ubuntu 22.04 with NVIDIA GPUs. Windows 11 has only been exercised to run the deployed Nuclio functions inside Docker containers.

### Required Software

#### Docker & Docker Compose

- **Docker Engine**: 20.10+ or Docker Desktop
- **Docker Compose**: v2.0+
- **nvidia-docker**: Required for GPU support

**Installation**:
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

See [Docker installation guide](https://docs.docker.com/engine/install/).

#### Pixi Package Manager

- **Pixi**: Latest version
- **Purpose**: Manages Python environments for all modules

**Installation**:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

See [Pixi documentation](https://pixi.sh/latest/#installation).

#### Git

- **Git**: 2.20+

```bash
# Ubuntu/Debian
sudo apt-get install git
```

### Optional Software

| Software | Purpose | Priority |
|----------|---------|----------|
| **VSCode / PyCharm** | IDE for development | Recommended |
| **p7zip** | Extract test samples | Required for quickstart |
| **nvidia-smi** | GPU monitoring | Recommended |
| **htop** | System monitoring | Optional |

## GPU Requirements

### NVIDIA GPU

**Required for ML inference**. Supports both CUDA 11.x and CUDA 12.x depending on module.

#### CUDA Version by Module

| Module | CUDA Version | PyTorch Version |
|--------|--------------|-----------------|
| **DINOv2_mmseg** | 11.7 | 2.0.0 |
| **coral_seg_yolo** | 12.1 | 2.5.0 |
| **grid_pose_detection** | 12.x | Latest |
| **grid_inpainting** | 12.1 | Latest |

!!! note "Multiple CUDA Versions"
    Different modules use different CUDA versions. Pixi handles this automatically by creating isolated environments. Your system only needs `nvidia-docker` installed.

#### Tested GPUs

| GPU Model | VRAM | Status | Notes |
|-----------|------|--------|-------|
| **NVIDIA RTX 4090** | 24 GB | ✅ Tested | Reference production setup on Ubuntu 22.04 |
| **NVIDIA RTX 3090** | 24 GB | ⚠️ Recommended | Same class as 4090; slated for production but not field-tested |
| **NVIDIA GTX 1070** | 8 GB | ✅ Tested | Development laptop; expect longer training/inference times |

### AMD GPU / Intel GPU

**Not supported**. The pipeline requires NVIDIA CUDA. AMD ROCm and Intel oneAPI are not currently supported.

## Network Requirements

### Internet Connectivity

**Required for**:
- Initial setup (downloading Docker images, models, dependencies)
- Optional: External CVAT access for remote reviewers

### Bandwidth Estimates

| Activity | Download | Upload |
|----------|----------|--------|
| **Initial Setup** | 20-30 GB | < 100 MB |
| **Model Downloads** | 5-10 GB | None |
| **Test Data Downloads** | 500 MB | None |
| **Daily Operation** | Minimal | Minimal |
| **Annotation Upload** | None | Variable |

### Ports

The following ports must be available:

| Service | Port | Purpose | Public? |
|---------|------|---------|---------|
| **CVAT Web** | 8080 | Web interface | Optional |
| **CVAT WebSocket** | 8080 (WS) | Real-time updates | Optional |
| **Nuclio Dashboard** | 8070 | Function management | No |
| **Nuclio Functions** | 32768-33768 | ML inference | No |
| **Bridge API** | 8000 | Webhook service | No |
| **PostgreSQL** | 5432 | Database (internal) | No |
| **Redis** | 6379 | Cache (internal) | No |

!!! danger "Security Warning"
    **Do not** expose Nuclio dashboard (8070) or Bridge API (8000) to the public internet without proper authentication. These are meant for internal network use only.

## Python Requirements

### Python Versions by Module

| Module | Python Version | Notes |
|--------|----------------|-------|
| **bridge** | 3.11 | Latest features |
| **data_engineering** | 3.9 | FiftyOne compatibility |
| **grid_pose_detection** | 3.9 | PyTorch stability |
| **grid_inpainting** | 3.9+ | Flexible |
| **coral_seg_yolo** | 3.9 | NumPy compatibility |
| **DINOv2_mmseg** | 3.9 | MMSegmentation requirement |
| **documentation** | 3.10+ | MkDocs Material |

!!! tip "Pixi Manages Python Versions"
    You don't need to install Python manually. Pixi automatically creates isolated environments with the correct Python version for each module.

## Deployment Scenarios

### Local Desktop Deployment

**Suitable for**:
- Individual researchers
- Desktop workstations with high-end NVIDIA GPUs
- On-premise experimentation without external dependencies

**Baseline**:
- All services run on the same machine
- Ubuntu 22.04 with Docker, Nuclio, and Pixi
- RTX 3090/4090 preferred; GTX 1070 verified for smaller experiments

## Browser Requirements (for CVAT)

| Browser | Version | Status |
|---------|---------|--------|
| **Chrome** | Latest | ✅ Recommended |
| **Firefox** | Latest | ✅ Supported |
| **Edge** | Latest | ✅ Supported |
| **Safari** | 14+ | ⚠️ Partial support |

## Pre-Installation Checklist

Use this checklist before proceeding to installation:

### Hardware
- [ ] CPU meets minimum requirements (4+ cores)
- [ ] RAM meets minimum requirements (16+ GB)
- [ ] Storage has 100+ GB available
- [ ] NVIDIA GPU with 8+ GB VRAM (GTX 1070 tested; 24 GB recommended)

### Software
- [ ] Operating system matches tested environments (Ubuntu 22.04 for development, Windows 11 for container runtime)
- [ ] Docker Engine 20.10+ installed
- [ ] Docker Compose v2.0+ installed
- [ ] nvidia-docker installed (for GPU support)
- [ ] Pixi package manager installed
- [ ] Git installed

### Network
- [ ] Internet connectivity for downloads
- [ ] Required ports available (8080, 8070, 8000)
- [ ] Firewall configured appropriately

### Access
- [ ] User has sudo/admin privileges (for Docker setup)
- [ ] User is in `docker` group
- [ ] GPU drivers installed (if using GPU)

### Verification Commands

```bash
# Check Docker
docker --version
docker compose version

# Check GPU
nvidia-smi

# Check Pixi
pixi --version

# Check Git
git --version

# Check available storage
df -h

# Check RAM
free -h

# Check CPU cores
nproc
```

## Next Steps

!!! success "Requirements Met?"
    If all requirements are satisfied, proceed to installation.

**Continue to**:

- [Installation Guide](installation/index.md) - Begin installing components
- [Pixi Environment Setup](installation/for-developers/1-pixi-setup.md) - Set up Pixi first
- [CVAT + Nuclio Installation](installation/for-end-users/1-docker-deployment.md) - Deploy core services

---

**Questions about requirements?** See [Getting Help](../community/index.md).
