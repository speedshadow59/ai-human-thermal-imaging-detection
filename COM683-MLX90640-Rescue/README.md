# COM683 Coursework 2 - AI-Assisted Thermal Imaging (MLX90640 + Edge Impulse)

This is a standalone project pack for an embedded + machine learning coursework portfolio using the MLX90640 thermal sensor, with training and testing completed in Edge Impulse.

## 1) Project Goal
Build an edge-intelligent thermal detection system that can support fire-rescue style search in low-visibility conditions by detecting heat signatures from MLX90640 frames.

## 2) Folder Structure
- `arduino/mlx90640_streamer/` - Arduino sketch that streams 32x24 thermal frames over serial.
- `arduino/mlx90640_ei_inference/` - Arduino sketch that runs the exported Edge Impulse person detector.
- `python/src/data/` - Data capture and Edge Impulse dataset export scripts.
- `python/src/viz/` - Live viewer for validation and demonstration.
- `data/logs/` - Live-session JSONL telemetry used for online evaluation.
- `data/photos/` - Thermal captures kept as submission evidence.
- `docs/slide_assets/` - Final visual assets used in the presentation (submitted separately via Blackboard).
- `docs/` - Submission-facing presentation and evidence files (submitted separately via Blackboard).

## 3) Quick Start
### A. Arduino streaming
1. Open `arduino/mlx90640_streamer/mlx90640_streamer.ino`.
2. Install libraries:
   - Adafruit MLX90640
   - Adafruit BusIO
3. Select the correct board and COM port.
4. Upload sketch and open serial monitor at `115200` (or close monitor before Python capture).

### B. Python environment
```powershell
cd C:\Users\user\COM683-MLX90640-Rescue\python
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### C. Collect data by class
```powershell
python src\data\capture_mlx.py --port COM4 --label person --frames 600
python src\data\capture_mlx.py --port COM4 --label pet --frames 600
python src\data\capture_mlx.py --port COM4 --label empty --frames 600
python src\data\capture_mlx.py --port COM4 --label heater --frames 600
```

### D. Export image dataset for Edge Impulse
```powershell
python src\data\export_edge_impulse_images.py --raw-root data\raw --out-dir data\edge_impulse --train-ratio 0.7 --test-ratio 0.2
```

### E. Build mixed dataset (public + own data)
If you also have public thermal datasets in class-folder format, merge them with your own captures:

```powershell
python src\data\prepare_mixed_edge_impulse_dataset.py --own-raw-root data\raw --external-dir C:\path\to\public_dataset_a --external-dir C:\path\to\public_dataset_b --out-dir data\edge_impulse_mixed --train-ratio 0.7 --test-ratio 0.2
```

Optional label harmonization (for example `human`, `person`, `pedestrian` -> `person`) via JSON mapping file:

```powershell
python src\data\prepare_mixed_edge_impulse_dataset.py --own-raw-root data\raw --external-dir C:\path\to\public_dataset --label-map data\label_map.json --out-dir data\edge_impulse_mixed
```

### F. Train and test in Edge Impulse Studio
1. Use either `python/data/edge_impulse/...` (own data only) or `python/data/edge_impulse_mixed/...` (own + public).
2. Zip training and testing folders.
2. In Edge Impulse, create an image classification project.
3. Upload train/test data preserving label folders.
4. Configure impulse (Image block + Classification block).
5. Train model and capture metrics (accuracy, confusion matrix, per-class F1).
6. Run Model Testing in Edge Impulse and export results screenshots/tables.

### G. Deploy from Edge Impulse
1. Use Deployment in Edge Impulse to export Arduino library.
2. Import the generated `.zip` in Arduino IDE.
3. Integrate inference with MLX90640 frame capture on-device.

### H. Run live viewer
```powershell
python src\viz\live_viewer.py --port COM4
```

## 4) Evidence You Should Capture
- Data collection logs and class balance table.
- Edge Impulse data split and data quality controls.
- Dataset provenance from `manifest.csv` and `dataset_summary.json` (own vs public source counts).
- Impulse design and training configuration.
- Offline metrics from Edge Impulse: accuracy, precision, recall, F1, confusion matrix.
- Online metrics: inference speed, robustness tests, failure cases.
- Demonstration video with split-screen hardware + dashboard.

## 5) External Datasets
No third-party public thermal datasets were ultimately incorporated into the final submitted dataset. All training and testing samples were captured directly using the MLX90640 sensor during dedicated recording sessions. The `prepare_mixed_edge_impulse_dataset.py` script was developed to support potential dataset merging but was not used in the final pipeline. The Edge Impulse project linked in Section 8 reflects training on own-captured data only.

## 7) Recommended Classes
- `person`
- `pet`
- `empty`
- `heater`
- `hot-object`

Adjust to match your exact use case and available scenes.

## 8) Safety and Ethics
- Do not claim system replaces firefighters.
- Present as decision support tool.
- Include false positive/negative risks and mitigation strategy.

## 9) Submission Components
- Code zip (this project + comments, including Edge Impulse deployment integration code)
- Dataset zip (`data/photos`, logs, and the processed train/test metadata you are submitting)
- Slides (your final PPT/PDF plus `docs/slide_assets/` if needed for rebuild evidence)
- 1-minute demo video (split-screen)

## 10) Edge Impulse Public Project Link

Your coursework solution (training, model, and deployment) is implemented in Edge Impulse Studio and can be viewed here:

[https://studio.edgeimpulse.com/public/970214/live](https://studio.edgeimpulse.com/public/970214/live)

Please refer to this link for full project details, model metrics, and deployment artifacts.
