import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import serial
from matplotlib.patches import Rectangle


FRAME_SIZE = 32 * 24


def parse_line(line: str):
    parts = line.strip().split(",")
    if len(parts) != FRAME_SIZE:
        return None
    try:
        arr = np.array(parts, dtype=np.float32).reshape(24, 32)
    except ValueError:
        return None
    return arr


def save_photo(file_path: Path, frame: np.ndarray, vmin: float, vmax: float):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(file_path, frame, cmap="inferno", vmin=vmin, vmax=vmax)


def main():
    parser = argparse.ArgumentParser(description="Live thermal viewer for MLX90640")
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--photo-dir", default="data/photos")
    parser.add_argument("--photo-prefix", default="thermal")
    parser.add_argument("--auto-photo-sec", type=float, default=0.0)
    parser.add_argument("--log-dir", default="data/logs")
    parser.add_argument("--log-file", default="")
    parser.add_argument("--log-frame-stats", action="store_true")
    parser.add_argument("--log-frame-every", type=int, default=10)
    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1)

    photo_dir = Path(args.photo_dir)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = Path(args.log_file)
    else:
        log_path = log_dir / f"viewer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    if not log_path.is_absolute():
        log_path = Path.cwd() / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fp = log_path.open("a", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 6))
    img = ax.imshow(np.zeros((24 * args.upscale, 32 * args.upscale)), cmap="inferno", vmin=20, vmax=40)
    cbar = plt.colorbar(img)
    cbar.set_label("Temperature (C)")

    txt = ax.text(0.01, 1.02, "Waiting for frames...", transform=ax.transAxes)
    bbox_patch = Rectangle((0, 0), 1, 1, fill=False, edgecolor="lime", linewidth=2, visible=False)
    ax.add_patch(bbox_patch)
    ax.set_title("MLX90640 Live Viewer")

    status = {
        "mode": "unknown",
        "person_detected": 0,
        "confidence": 0.0,
        "x_norm": 0.0,
        "y_norm": 0.0,
        "w_norm": 0.0,
        "h_norm": 0.0,
    }
    latest_frame = None
    latest_vmin = 20.0
    latest_vmax = 40.0
    last_photo_ts = 0.0
    frame_counter = 0
    bbox_smooth_alpha = 0.35
    bbox_state = {
        "initialized": False,
        "x0": 0.0,
        "y0": 0.0,
        "bw": 1.0,
        "bh": 1.0,
    }

    def write_log(record: dict):
        record["host_ts"] = datetime.now().isoformat(timespec="milliseconds")
        log_fp.write(json.dumps(record) + "\n")
        log_fp.flush()

    def update_text(prefix: str):
        txt.set_text(
            "\n".join(
                [
                    prefix,
                    f"mode: {status['mode']}",
                    f"person: {status['person_detected']}    conf: {status['confidence']:.2f}",
                ]
            )
        )

    def update_bbox(frame_shape):
        if status["person_detected"] != 1 or status["w_norm"] <= 0.0 or status["h_norm"] <= 0.0:
            bbox_patch.set_visible(False)
            bbox_state["initialized"] = False
            return

        h, w = frame_shape
        cx = float(status["x_norm"]) * w
        cy = float(status["y_norm"]) * h
        bw = float(status["w_norm"]) * w
        bh = float(status["h_norm"]) * h

        x0 = max(0.0, cx - (bw / 2.0))
        y0 = max(0.0, cy - (bh / 2.0))
        bw = min(bw, max(1.0, w - x0))
        bh = min(bh, max(1.0, h - y0))

        if not bbox_state["initialized"]:
            bbox_state["x0"] = x0
            bbox_state["y0"] = y0
            bbox_state["bw"] = bw
            bbox_state["bh"] = bh
            bbox_state["initialized"] = True
        else:
            alpha = bbox_smooth_alpha
            bbox_state["x0"] = ((1.0 - alpha) * bbox_state["x0"]) + (alpha * x0)
            bbox_state["y0"] = ((1.0 - alpha) * bbox_state["y0"]) + (alpha * y0)
            bbox_state["bw"] = ((1.0 - alpha) * bbox_state["bw"]) + (alpha * bw)
            bbox_state["bh"] = ((1.0 - alpha) * bbox_state["bh"]) + (alpha * bh)

        bbox_patch.set_xy((bbox_state["x0"], bbox_state["y0"]))
        bbox_patch.set_width(bbox_state["bw"])
        bbox_patch.set_height(bbox_state["bh"])
        bbox_patch.set_visible(True)

    def capture_photo():
        if latest_frame is None:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        file_name = (
            f"{args.photo_prefix}_{timestamp}_p{int(status['person_detected'])}_c{float(status['confidence']):.2f}.png"
        )
        file_path = photo_dir / file_name
        save_photo(file_path, latest_frame, latest_vmin, latest_vmax)
        print(f"Saved photo: {file_path}")

    def on_key(event):
        if event.key and event.key.lower() == "p":
            capture_photo()

    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(_):
        nonlocal latest_frame, latest_vmin, latest_vmax, last_photo_ts, frame_counter
        line = ser.readline().decode(errors="ignore").strip()
        if not line or line in {"READY", "FRAME_ERROR"}:
            return [img, txt, bbox_patch]

        if line.startswith("{") and line.endswith("}"):
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return [img, txt, bbox_patch]

            status["mode"] = str(event.get("mode", status["mode"]))
            status["person_detected"] = int(event.get("person_detected", status["person_detected"]))
            status["confidence"] = float(event.get("confidence", status["confidence"]))
            status["x_norm"] = float(event.get("x_norm", status["x_norm"]))
            status["y_norm"] = float(event.get("y_norm", status["y_norm"]))
            status["w_norm"] = float(event.get("w_norm", status["w_norm"]))
            status["h_norm"] = float(event.get("h_norm", status["h_norm"]))
            write_log({"type": "event", "event": event})

            if latest_frame is not None:
                update_bbox(latest_frame.shape)

            if "min_temp" in event and "max_temp" in event:
                update_text(
                    "Inference-only stream\n"
                    f"min: {float(event['min_temp']):.2f}C    max: {float(event['max_temp']):.2f}C"
                )
            else:
                update_text("Inference-only stream")
            return [img, txt, bbox_patch]

        frame = parse_line(line)
        if frame is None:
            return [img, txt, bbox_patch]

        up = np.repeat(np.repeat(frame, args.upscale, axis=0), args.upscale, axis=1)
        frame_counter += 1
        latest_frame = up
        img.set_data(up)
        latest_vmin = float(np.min(frame))
        latest_vmax = float(np.max(frame))
        img.set_clim(vmin=latest_vmin, vmax=latest_vmax)
        update_bbox(up.shape)
        txt.set_text(
            "\n".join(
                [
                    "Thermal frame",
                    f"min: {np.min(frame):.2f}C    max: {np.max(frame):.2f}C",
                    f"mean: {np.mean(frame):.2f}C",
                    f"mode: {status['mode']}    person: {status['person_detected']}    conf: {status['confidence']:.2f}",
                ]
            )
        )

        if args.log_frame_stats and frame_counter % args.log_frame_every == 0:
            write_log(
                {
                    "type": "frame_stats",
                    "frame_idx": frame_counter,
                    "min_temp": float(np.min(frame)),
                    "max_temp": float(np.max(frame)),
                    "mean_temp": float(np.mean(frame)),
                    "person_detected": int(status["person_detected"]),
                    "confidence": float(status["confidence"]),
                }
            )

        if args.auto_photo_sec > 0:
            now = time.monotonic()
            if now - last_photo_ts >= args.auto_photo_sec:
                capture_photo()
                last_photo_ts = now

        return [img, txt, bbox_patch]

    animation.FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    plt.show()
    log_fp.close()


if __name__ == "__main__":
    main()
