# System Requirements

Review hardware, software, and network requirements before installing the CRIOBE coral segmentation pipeline.

## Hardware Requirements

### Minimum Requirements (Development/Testing)

| Component | Specification |
|-----------|---------------|
| **CPU** | 4 cores, 2.0 GHz |
| **RAM** | 16 GB |
| **Storage** | 100 GB available (SSD recommended) |
| **GPU** | NVIDIA GPU with 6 GB VRAM, CUDA 11.7+ support |
| **Network** | 10 Mbps download (for model/data downloads) |

!!! warning "GPU Required for ML Inference"
    While the system can run on CPU, ML inference will be 10-50x slower. An NVIDIA GPU with CUDA support is **strongly recommended** for production use.

### Recommended Requirements (Production/Multi-User)

| Component | Specification |
|-----------|---------------|
| **CPU** | 8+ cores, 3.0 GHz |
| **RAM** | 32 GB+ |
| **Storage** | 500 GB+ NVMe SSD |
| **GPU** | NVIDIA GPU with 8+ GB VRAM (RTX 3070 or better) |
| **Network** | 100 Mbps+ (1 Gbps for multi-user) |

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
| **Ubuntu 20.04/22.04 LTS** | ✅ Recommended | Best supported, all features work |
| **Debian 11/12** | ✅ Supported | Fully compatible |
| **Other Linux** | ⚠️ May work | Community tested |
| **macOS 12+** | ⚠️ Partial | CPU only, no CUDA support |
| **Windows 10/11** | ⚠️ Via WSL2 | Use WSL2 + Ubuntu for best experience |

!!! tip "Windows Users"
    For Windows, install WSL2 with Ubuntu 22.04 for the best experience. Native Windows support is limited.

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
| **NVIDIA RTX 4090** | 24 GB | ✅ Excellent | Maximum performance |
| **NVIDIA RTX 3090** | 24 GB | ✅ Excellent | Great for production |
| **NVIDIA RTX 3080** | 10 GB | ✅ Good | Recommended minimum |
| **NVIDIA RTX 3070** | 8 GB | ✅ Good | Works well with reduced batch size |
| **NVIDIA GTX 1070** | 8 GB | ✅ Acceptable | Tested, slower but functional |
| **NVIDIA T4** | 16 GB | ✅ Good | Cloud GPU option |
| **NVIDIA V100** | 16/32 GB | ✅ Excellent | Cloud GPU option |

### AMD GPU / Intel GPU

**Not supported**. The pipeline requires NVIDIA CUDA. AMD ROCm and Intel oneAPI are not currently supported.

### CPU-Only Mode

**Possible but slow**. All modules can run on CPU, but inference is 10-50x slower:

| Module | GPU Time | CPU Time | Speedup |
|--------|----------|----------|---------|
| Grid Detection | ~2s | ~10s | 5x |
| Grid Removal | ~6s | ~60s | 10x |
| YOLO Segmentation | ~7s | ~2min | 17x |
| DINOv2 Segmentation | ~20s | ~15min | 45x |

**Recommendation**: Use CPU mode only for testing/development with small datasets.

## Network Requirements

### Internet Connectivity

**Required for**:
- Initial setup (downloading Docker images, models, dependencies)
- Optional: External CVAT access, cloud deployment

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

### Single Server (Development)

**Suitable for**:
- Individual researchers
- Small-scale testing
- Development work

**Requirements**:
- Minimum specs listed above
- All services on one machine
- Shared GPU for inference

### Multi-Server (Production)

**Suitable for**:
- Research teams
- Production workflows
- High-throughput processing

**Architecture**:
```
Server 1: CVAT + Database + Redis
Server 2: Nuclio + ML Functions (GPU required)
Server 3: Bridge Service
Storage: NFS/NAS for shared data
```

**Requirements per server**:
- Server 1: 16 GB RAM, 4 cores, 100 GB storage
- Server 2: 32 GB RAM, 8 cores, GPU with 8+ GB VRAM
- Server 3: 8 GB RAM, 2 cores, 50 GB storage
- Storage: 500+ GB shared storage

### Cloud Deployment

**Supported platforms**:
- **AWS**: EC2 with GPU instances (p3/p4), ECS/EKS
- **Google Cloud**: Compute Engine with GPU, GKE
- **Azure**: GPU VMs, AKS

**Recommended instance types**:

| Provider | Instance Type | vCPU | RAM | GPU | Cost/hr* |
|----------|---------------|------|-----|-----|----------|
| **AWS** | p3.2xlarge | 8 | 61 GB | V100 (16GB) | ~$3.06 |
| **AWS** | g4dn.xlarge | 4 | 16 GB | T4 (16GB) | ~$0.526 |
| **GCP** | n1-standard-8 + T4 | 8 | 30 GB | T4 (16GB) | ~$0.80 |
| **Azure** | NC6s_v3 | 6 | 112 GB | V100 (16GB) | ~$3.06 |

*Approximate on-demand pricing, varies by region

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
- [ ] NVIDIA GPU with 6+ GB VRAM (or accepting CPU-only limitations)

### Software
- [ ] Operating system is supported (Linux recommended)
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

# Check GPU (if applicable)
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
- [Pixi Environment Setup](installation/pixi-environment.md) - Set up Pixi first
- [CVAT + Nuclio Installation](installation/cvat-nuclio.md) - Deploy core services

---

**Questions about requirements?** See [Getting Help](../community/getting-help.md).
