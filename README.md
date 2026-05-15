# AI Human Thermal Imaging Detection

Embedded thermal person detection pipeline using an MLX90640 (32x24) sensor, Edge Impulse model training, and Arduino on-device inference.

## Repository Layout
- `COM683-MLX90640-Rescue/` - Main project folder.
- `COM683-MLX90640-Rescue/arduino/` - Sensor streaming and Edge Impulse inference sketches.
- `COM683-MLX90640-Rescue/python/` - Data capture, dataset preparation, and live visualization scripts.
- `COM683-MLX90640-Rescue/data/` - Session logs and captured thermal artifacts.

## Quick Start
1. Open the full project guide in `COM683-MLX90640-Rescue/README.md`.
2. Set up Python dependencies from `COM683-MLX90640-Rescue/python/requirements.txt`.
3. Flash `arduino/mlx90640_streamer/` to collect frames.
4. Train in Edge Impulse and deploy with `arduino/mlx90640_ei_inference/`.

## Edge Impulse Project
https://studio.edgeimpulse.com/public/970214/live