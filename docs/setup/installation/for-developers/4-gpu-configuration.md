# GPU Configuration

Configure and optimize NVIDIA GPU settings for training QUADRATSEG models with maximum performance.

!!! info "For Developers"
    This guide configures **CUDA and GPU settings** for training. While GPUs significantly accelerate training (10-100x faster), CPU-only training is possible for testing (see [CPU Training section](#cpu-training-alternative)).

**Time Required**: 20-30 minutes
**Prerequisites**: [Module environments installed](2-module-environments.md), NVIDIA GPU with CUDA support

## GPU Requirements

### Minimum GPU Specifications

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **GPU Memory** | 6GB | 11GB | 24GB+ |
| **CUDA Cores** | 1000+ | 2000+ | 5000+ |
| **CUDA Compute** | 6.0+ | 7.5+ | 8.0+ |
| **Examples** | GTX 1060 | RTX 3060 | RTX 4090, A100 |

### Memory Requirements by Task

| Task | Minimum VRAM | Recommended | Notes |
|------|--------------|-------------|-------|
| Corner Detection Training | 4GB | 6GB | YOLOv11n, batch size 16 |
| Grid Pose Training | 4GB | 6GB | YOLOv11n, batch size 16 |
| Coral Seg (YOLO) Training | 8GB | 11GB | YOLOv11m-seg, batch size 8 |
| DINOv2 Training | 16GB | 24GB | Large models, batch size 4 |
| Inference (all models) | 4GB | 6GB | Single image at a time |

## Step 1: Verify CUDA Installation

Check that CUDA toolkit is installed and accessible:

### Check NVIDIA Driver

```bash
# Check driver version and CUDA version
nvidia-smi

# Expected output showing:
# - Driver Version
# - CUDA Version
# - GPU name, memory, utilization
```

**Example Output**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.129.03   Driver Version: 535.129.03   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   45C    P8    15W / 180W |   1024MiB / 12288MiB |      5%      Default |
+-------------------------------+----------------------+----------------------+
```

### Check CUDA Toolkit Version

```bash
# Check nvcc compiler version
nvcc --version

# Expected output:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2023 NVIDIA Corporation
# Built on ...
# Cuda compilation tools, release 12.1, V12.1.xxx
```

!!! warning "CUDA Version Compatibility"
    Different modules require different CUDA versions:
    - **coral_seg_yolo, grid_pose_detection**: CUDA 12.1
    - **DINOv2_mmseg**: CUDA 11.7

    Both can coexist in separate Pixi environments!

### Install CUDA Toolkit (If Not Installed)

**Ubuntu/Debian**:
```bash
# Add NVIDIA package repository
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA 12.1
sudo apt-get install cuda-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

**Alternative: Download from NVIDIA**:
- Visit: https://developer.nvidia.com/cuda-downloads
- Select your OS, architecture, and CUDA version
- Follow installation instructions

## Step 2: Verify GPU Access from PyTorch

Test that PyTorch can access your GPU from each environment:

### Test coral_seg_yolo Environment

```bash
cd ~/Projects/coral-segmentation/coral_seg_yolo

# Test PyTorch CUDA
pixi run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
"
```

**Expected Output**:
```
PyTorch version: 2.5.0
CUDA available: True
CUDA version: 12.1
GPU count: 1
GPU name: NVIDIA GeForce RTX 3060
GPU memory: 12.00 GB
```

### Test DINOv2_mmseg Environment

```bash
cd ~/Projects/coral-segmentation/DINOv2_mmseg

# Test with CUDA 11.7
pixi run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
"
```

**Expected Output**:
```
PyTorch version: 2.0.0
CUDA available: True
CUDA version: 11.7
```

!!! success "GPU Accessible"
    If both tests show `CUDA available: True`, your GPU is correctly configured!

## Step 3: Optimize GPU Settings

### Set GPU Memory Growth (Optional)

For training with multiple processes or experiments:

```bash
# Allow GPU memory to grow as needed (PyTorch default)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Add to ~/.bashrc for persistence
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc
```

### Set Specific GPU Device

If you have multiple GPUs, select which to use:

```bash
# Use GPU 0 (default)
export CUDA_VISIBLE_DEVICES=0

# Use GPU 1
export CUDA_VISIBLE_DEVICES=1

# Use GPUs 0 and 2
export CUDA_VISIBLE_DEVICES=0,2

# Hide all GPUs (CPU mode)
export CUDA_VISIBLE_DEVICES=""
```

**In Training Scripts**:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Before importing torch

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Configure cuDNN for Performance

```bash
# Enable cuDNN autotuner (finds fastest algorithms)
export TORCH_CUDNN_V8_API_ENABLED=1

# Add to ~/.bashrc
echo 'export TORCH_CUDNN_V8_API_ENABLED=1' >> ~/.bashrc
```

**In Python**:
```python
import torch
torch.backends.cudnn.benchmark = True  # Enable autotuner
torch.backends.cudnn.deterministic = False  # Faster but non-deterministic
```

## Step 4: Monitor GPU Usage

### Real-Time Monitoring

```bash
# Watch GPU usage in real-time (updates every 1 second)
watch -n 1 nvidia-smi

# Or with more detail
watch -n 1 'nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv'
```

### GPU Monitoring During Training

**Option A: Built-in nvidia-smi**:
```bash
# In one terminal: start training
cd coral_seg_yolo
pixi run python src/training/train.py

# In another terminal: monitor GPU
watch -n 1 nvidia-smi
```

**Option B: gpustat (More User-Friendly)**:
```bash
# Install gpustat
pip install gpustat

# Monitor with gpustat
watch -n 1 gpustat -cp

# Shows:
# - GPU utilization %
# - Memory usage (used/total)
# - Temperature
# - Running processes
```

**Option C: nvtop (Interactive TUI)**:
```bash
# Install nvtop
sudo apt-get install nvtop

# Launch interactive monitor
nvtop
```

### Log GPU Stats During Training

Add GPU monitoring to training scripts:

```python
import torch

def log_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f'GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved')

# During training loop
for epoch in range(num_epochs):
    log_gpu_memory()
    # ... training code ...
```

## Step 5: Benchmark GPU Performance

Test GPU performance with actual training workload:

### Quick Inference Benchmark

```bash
cd ~/Projects/coral-segmentation/coral_seg_yolo

# Download test image
mkdir -p /tmp/benchmark
wget -O /tmp/benchmark/test.jpg https://storage.googleapis.com/criobe_public/test_samples/1-raw_jpg/sample.jpg

# Benchmark inference
pixi run python -c "
import torch
import time
from ultralytics import YOLO

# Load model
model = YOLO('models/coralsegv4_yolo11m_best.pt')
model.to('cuda')

# Warmup
for _ in range(5):
    model('/tmp/benchmark/test.jpg', verbose=False)

# Benchmark
times = []
for _ in range(50):
    start = time.time()
    model('/tmp/benchmark/test.jpg', verbose=False)
    torch.cuda.synchronize()  # Wait for GPU to finish
    times.append(time.time() - start)

print(f'Mean inference time: {sum(times)/len(times)*1000:.2f} ms')
print(f'Throughput: {1/(sum(times)/len(times)):.2f} FPS')
"
```

**Expected Performance** (RTX 3060):
```
Mean inference time: 135 ms
Throughput: 7.4 FPS
```

### Training Benchmark

```bash
cd ~/Projects/coral-segmentation/coral_seg_yolo

# Short training benchmark (1 epoch)
pixi run python src/training/train.py \
    --config experiments/benchmark_config.yaml \
    --epochs 1 \
    --batch-size 8 \
    --workers 4

# Check training speed in output:
# "train: Epoch 1/1: 100%|█| 125/125 [02:30<00:00, 1.20s/it]"
# Speed: ~1.2 seconds per batch (8 images)
```

## GPU Memory Optimization

### Reduce Batch Size

If you encounter out-of-memory errors:

```yaml
# experiments/train_config.yaml
batch_size: 4  # Reduce from 8 or 16
```

### Use Mixed Precision Training

Enable automatic mixed precision (AMP) for 2x speed and 50% memory savings:

```python
# In training script
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()

    # Forward pass with autocast
    with autocast():
        outputs = model(batch)
        loss = criterion(outputs, targets)

    # Backward pass with gradient scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**YOLO** (built-in AMP):
```yaml
# experiments/train_config.yaml
amp: true  # Enable mixed precision
```

### Gradient Accumulation

Simulate larger batch sizes with limited GPU memory:

```python
# Accumulate gradients over 4 mini-batches
accumulation_steps = 4

for i, batch in enumerate(dataloader):
    outputs = model(batch)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**YOLO** (built-in):
```yaml
# experiments/train_config.yaml
batch_size: 4
gradient_accumulation: 4  # Effective batch size = 4 * 4 = 16
```

### Clear GPU Cache

If experiencing memory fragmentation:

```python
import torch

# Clear cached memory
torch.cuda.empty_cache()

# Synchronize (wait for all operations to complete)
torch.cuda.synchronize()
```

## Multi-GPU Training

If you have multiple GPUs:

### PyTorch DataParallel

```python
import torch
import torch.nn as nn

model = MyModel()

# Wrap model for multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model.to('cuda')
```

### PyTorch DistributedDataParallel (Recommended)

```bash
# Train with 2 GPUs
torchrun --nproc_per_node=2 train.py

# Or with specific GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py
```

**YOLO Multi-GPU**:
```yaml
# experiments/train_config.yaml
device: [0, 1]  # Use GPUs 0 and 1
```

## CPU Training (Alternative)

If you don't have a GPU or want to test on CPU:

### Enable CPU Mode

```bash
# Disable CUDA
export CUDA_VISIBLE_DEVICES=""

# Train on CPU
cd coral_seg_yolo
pixi run python src/training/train.py --device cpu
```

**Performance**:
- CPU training is 10-100x slower than GPU
- Useful for testing, not practical for full training
- Inference on CPU: ~2-5 seconds per image (vs 0.1-0.2s on GPU)

### Optimize CPU Training

```python
import torch

# Use all CPU cores
torch.set_num_threads(8)

# Optimize for CPU
torch.backends.mkldnn.enabled = True
```

## Troubleshooting

### CUDA Not Available in PyTorch

**Symptoms**: `torch.cuda.is_available()` returns `False`

**Solutions**:
```bash
# Check CUDA version compatibility
nvidia-smi  # Note CUDA version
python -c "import torch; print(torch.version.cuda)"
# Should match or be compatible

# Reinstall PyTorch with correct CUDA version
# In pixi.toml, ensure pytorch-cuda matches your system:
# pytorch-cuda = "12.1"

# Reinstall environment
cd <module>
pixi clean
pixi install

# Verify
pixi run python -c "import torch; print(torch.cuda.is_available())"
```

### Out of Memory (OOM) Errors

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# 1. Reduce batch size
# In config: batch_size: 4 (or lower)

# 2. Enable gradient checkpointing
# Trades compute for memory

# 3. Use mixed precision training
# In config: amp: true

# 4. Clear cache before training
pixi run python -c "import torch; torch.cuda.empty_cache()"

# 5. Reduce image resolution
# In config: imgsz: 1280 (instead of 1920)

# 6. Close other GPU processes
nvidia-smi  # Check for other processes
# Kill unnecessary processes using GPU
```

### GPU Utilization Low

**Symptoms**: GPU utilization <50% during training

**Solutions**:
```bash
# 1. Increase batch size
# In config: batch_size: 16 (or higher)

# 2. Increase dataloader workers
# In config: workers: 8

# 3. Enable pin_memory
# In dataloader: pin_memory=True

# 4. Check CPU bottleneck
# Run `htop` to see if CPU is at 100%

# 5. Use faster data augmentation
# Disable expensive augmentations during testing

# 6. Profile training
pixi run python -m torch.utils.bottleneck train.py
```

### CUDA Version Mismatch

**Symptoms**: `CUDA version mismatch: PyTorch was compiled with CUDA X but system has Y`

**Solutions**:
```bash
# Check versions
nvidia-smi  # System CUDA version
python -c "import torch; print(torch.version.cuda)"  # PyTorch CUDA

# Option 1: Update PyTorch to match system CUDA
# Edit pixi.toml:
# pytorch-cuda = "12.2"  # Match nvidia-smi version
pixi clean && pixi install

# Option 2: Install compatible CUDA toolkit
# See Step 1 for CUDA installation instructions
```

### Multiple CUDA Versions Conflict

**Symptoms**: Different modules need different CUDA versions

**Solutions**:
```bash
# This is expected! Pixi isolates environments.
# Each module can use its own CUDA version:

# coral_seg_yolo uses CUDA 12.1
cd coral_seg_yolo
pixi run python -c "import torch; print(torch.version.cuda)"
# Output: 12.1

# DINOv2_mmseg uses CUDA 11.7
cd ../DINOv2_mmseg
pixi run python -c "import torch; print(torch.version.cuda)"
# Output: 11.7

# No conflict! Each environment is isolated.
```

### GPU Memory Leak

**Symptoms**: GPU memory usage grows over time, doesn't release after training

**Solutions**:
```python
# Ensure proper cleanup
import torch

try:
    # Training code
    pass
finally:
    # Cleanup
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

# Delete models/tensors explicitly
del model, optimizer, scaler
torch.cuda.empty_cache()
```

## Performance Tuning Tips

### Optimal Batch Size

Find the maximum batch size for your GPU:

```bash
# Start with small batch size
pixi run python train.py --batch-size 4

# Increase until OOM
pixi run python train.py --batch-size 8
pixi run python train.py --batch-size 16
pixi run python train.py --batch-size 32

# Use largest batch size that doesn't OOM
# Rule of thumb: Fill 80-90% of GPU memory
```

### DataLoader Workers

Optimize CPU-GPU data transfer:

```python
# Too few workers: GPU idle waiting for data
workers = 0  # Slow

# Too many workers: CPU overhead
workers = 16  # Excessive

# Optimal: 2-4 workers per GPU
workers = 4  # Good balance
```

### Pin Memory

Speed up CPU-GPU transfers:

```python
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    pin_memory=True,  # Enable for GPU training
    persistent_workers=True  # Keep workers alive
)
```

### Prefetch Factor

Load batches ahead of time:

```python
dataloader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    prefetch_factor=2  # Prefetch 2 batches per worker
)
```

## GPU Configuration Best Practices

1. **Match CUDA versions**: System CUDA ≥ PyTorch CUDA
2. **Monitor temperature**: Keep GPU <85°C during training
3. **Use mixed precision**: Enable AMP for faster training
4. **Start small**: Test with small batch size, then increase
5. **Profile first**: Identify bottlenecks before optimizing
6. **Clean between runs**: Clear GPU cache between experiments
7. **Log GPU stats**: Monitor memory usage during training
8. **Use separate GPUs**: If multiple users, assign one GPU per user

## Next Steps

!!! success "GPU Configured!"
    Your GPU is optimized for training QUADRATSEG models!

**What's next**:

1. **[Configure CVAT Integration](../../configuration/for-developers/2-cvat-integration.md)** - Connect to CVAT for dataset management
2. **[Configure Training](../../configuration/for-developers/3-training-config.md)** - Set up training configurations
3. **Start Training** - Train your first coral segmentation model!

## Quick Reference

### GPU Check Commands

```bash
# GPU info
nvidia-smi
nvidia-smi -L  # List GPUs

# CUDA version
nvcc --version
nvidia-smi | grep "CUDA Version"

# PyTorch GPU check
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Monitor GPU
watch -n 1 nvidia-smi
gpustat -cp
nvtop
```

### Environment Variables

```bash
# Select GPU
export CUDA_VISIBLE_DEVICES=0

# Optimize performance
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDNN_V8_API_ENABLED=1

# Debug mode
export CUDA_LAUNCH_BLOCKING=1
```

### GPU Verification Checklist

- [ ] NVIDIA driver installed
- [ ] CUDA toolkit installed
- [ ] `nvidia-smi` shows GPU
- [ ] PyTorch detects GPU in all environments
- [ ] GPU memory sufficient for training
- [ ] Inference benchmark passes
- [ ] Training benchmark completes
- [ ] GPU monitoring tools installed

---

**Questions?** See [CVAT integration guide](../../configuration/for-developers/2-cvat-integration.md) or [Getting Help](../../../community/index.md).
