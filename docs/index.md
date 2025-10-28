# CRIOBE Tooling Documentation

This handbook accompanies the CRIOBE labeling and automation stack. It covers CVAT deployment, Nuclio serverless functions, and the bridge service that links annotation tasks to inference pipelines.

## How to Navigate
- **Getting set up** – Follow the [CVAT + Nuclio setup guide](en/setup_cvat_with_nuclio_and_bridge.md) and then deploy the Nuclio functions in [Serverless setup](en/setup_nuclio.md).
- **Bridge automation** – Start with the [Bridge overview](en/bridge/index.md), then dive into deployment, daily operations, or troubleshooting.
- **Pipeline background** – Review the [Coral segmentation pipeline](en/coral_segmentation_pipeline.md) to understand the process end-to-end.

## Contributing
- Use `pixi run serve-8010` for local previews on port 8010.
- Run `pixi run build` (strict mode) before opening a pull request.
- Report gaps or stale content by filing an issue or pinging the docs maintainers in your pull request description.
