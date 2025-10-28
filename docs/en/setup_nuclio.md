# Serverless Setup (Nuclio)

Deploy the Nuclio functions that power grid detection, grid removal, and coral segmentation. Complete the [platform setup](setup_cvat_with_nuclio_and_bridge.md) first so Docker networks and credentials are in place.

## Prerequisites
- CVAT + bridge stack running (Docker Compose).
- `nuctl` CLI installed on Linux/macOS hosts (see [Nuclio installation docs](https://docs.nuclio.io/en/stable/reference/nuctl/nuctl.html)).
- Access to the `serverless/` directory shipped with the CRIOBE CVAT fork.

## Linux / macOS Deployment (`nuctl`)
From `PROJECT_ROOT/cvat`, deploy each function into the `cvat` Nuclio project:

```bash
cd PROJECT_ROOT/cvat

# Segment Anything (GPU variant)
nuctl deploy --project-name cvat \
  --path "./serverless/pytorch/facebookresearch/sam/nuclio/" \
  --file "./serverless/pytorch/facebookresearch/sam/nuclio/function-gpu.yaml" \
  --platform local -v

# Coral segmentation (YOLO)
nuctl deploy --project-name cvat \
  --path "./serverless/pytorch/yolo/coralsegv4/nuclio/" \
  --file "./serverless/pytorch/yolo/coralsegv4/nuclio/function.yaml" \
  --platform local -v

# Grid pose detection
nuctl deploy --project-name cvat \
  --path "./serverless/pytorch/yolo/gridpose/nuclio/" \
  --platform local -v

# Grid corners (optional manual QA)
nuctl deploy --project-name cvat \
  --path "./serverless/pytorch/yolo/gridcorners/nuclio/" \
  --platform local -v

# Grid removal (LaMa)
nuctl deploy --project-name cvat \
  --path "./serverless/pytorch/lama/nuclio/" \
  --platform local -v
```

Additional models (`coralsegbanggai`, `coralscopsegformer`, etc.) follow the same pattern. After each deploy, confirm the function lists under `http://localhost:8070/projects/cvat`.

## Re-deploy Using Prebuilt Images
If a reboot causes Nuclio to lose build artifacts, re-import the `function_no_build.yaml` descriptor for each function:

1. Open the Nuclio dashboard → `cvat` project → **New Function**.
2. Choose **Import** and upload the matching `function_no_build.yaml`.
3. In the **Code** tab, switch **Code Entry Type** to `Image` and specify the image name (e.g., `cvat.pth.yolo.coralsegv4:latest-gpu`).
4. Click **Deploy** to relaunch the container without rebuilding.

This flow also works on hosts where `nuctl` is unavailable.

## Windows Workflow
`nuctl` is not supported on Windows, so manage functions through the dashboard:

1. Install CVAT in `D:\coraltag\cvat` following the [official guide](https://docs.cvat.ai/docs/administration/basics/installation/#windows-10).
2. Start CVAT with serverless capabilities:
   ```powershell
   d:
   cd coraltag\cvat
   docker compose -f docker-compose.yml `
     -f components/serverless/docker-compose.serverless.yml up -d
   ```
3. For functions whose models are downloadable (e.g., SAM), import `function-gpu.yaml` via the dashboard, then paste the code into a single editor pane if required.
4. For models that rely on local weights, transfer prebuilt images from a Linux machine:
   ```bash
   # On Linux
   nuctl deploy ...  # build the image
   docker save -o coralsegv4.tar cvat.pth.yolo.coralsegv4:latest-gpu

   # On Windows
   docker load -i coralsegv4.tar
   ```
   Then import `function_no_build.yaml` and select `cvat.pth.yolo.coralsegv4:latest-gpu` as the image.

## Custom Image Builds (No Linux Access)
When neither online weights nor Linux builds are possible, craft a Dockerfile that mirrors the Nuclio runtime:

1. Create a Dockerfile beside the function spec (see `serverless/pytorch/facebookresearch/detectron2/coraldet/nuclio/Dockerfile`).
2. Build it locally:
   ```bash
   docker build -t cvat.pth.facebookresearch.detectron2.coraldet:latest-gpu .
   ```
3. Update the YAML to reference the image (`function_no_build.yaml`) and deploy through the dashboard with **Code Entry Type = Image**.

## Post-Deployment Checks
- Verify each function is attached to the `cvat_cvat` network (Dashboard → function → **Configurations → Runtime**).
- Run a smoke test by invoking the Nuclio URL from the bridge container:
  ```bash
  curl -s -H "Content-Type: application/json" \
    --data @payload.json \
    http://nuclio-nuclio-pth-yolo-gridpose:8080
  ```
- If functions fail to start, consult the [bridge troubleshooting guide](bridge/troubleshooting.md) for network and Smokescreen fixes.
