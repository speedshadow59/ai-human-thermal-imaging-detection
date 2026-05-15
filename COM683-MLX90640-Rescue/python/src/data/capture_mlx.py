import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import serial


FRAME_SIZE = 32 * 24


def parse_frame(line: str):
    parts = line.strip().split(",")
    if len(parts) != FRAME_SIZE:
        return None
    try:
        values = np.array(parts, dtype=np.float32)
    except ValueError:
        return None
    return values.reshape(24, 32)


def main():
    parser = argparse.ArgumentParser(description="Capture MLX90640 frames over serial")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--label", required=True, help="Class label for this session")
    parser.add_argument("--frames", type=int, default=600, help="Number of valid frames to collect")
    parser.add_argument("--root", default="data/raw", help="Output root directory")
    parser.add_argument("--timeout", type=float, default=1.0)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.root) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    session_name = f"{args.label}_{timestamp}"
    npz_path = out_dir / f"{session_name}.npz"
    meta_path = out_dir / f"{session_name}_meta.json"

    print(f"Opening {args.port} @ {args.baud}")
    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)

    frames = []
    bad_lines = 0
    total_lines = 0

    print(f"Collecting {args.frames} valid frames for label='{args.label}'...")
    while len(frames) < args.frames:
        line = ser.readline().decode(errors="ignore").strip()
        if not line or line in {"READY", "FRAME_ERROR"}:
            continue

        total_lines += 1
        frame = parse_frame(line)
        if frame is None:
            bad_lines += 1
            continue

        frames.append(frame)
        if len(frames) % 50 == 0:
            print(f"Captured {len(frames)}/{args.frames}")

    ser.close()

    data = np.stack(frames, axis=0)
    np.savez_compressed(npz_path, frames=data, label=args.label)

    meta = {
        "session_name": session_name,
        "label": args.label,
        "port": args.port,
        "baud": args.baud,
        "frames_collected": int(data.shape[0]),
        "frame_shape": [24, 32],
        "bad_lines": bad_lines,
        "total_lines": total_lines,
        "min_temp": float(np.min(data)),
        "max_temp": float(np.max(data)),
        "mean_temp": float(np.mean(data)),
        "captured_at": timestamp,
    }

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    index_file = Path(args.root) / "sessions_index.csv"
    write_header = not index_file.exists()
    with index_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_name",
                "label",
                "frames_collected",
                "min_temp",
                "max_temp",
                "mean_temp",
                "bad_lines",
                "captured_at",
                "npz_path",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "session_name": session_name,
                "label": args.label,
                "frames_collected": data.shape[0],
                "min_temp": meta["min_temp"],
                "max_temp": meta["max_temp"],
                "mean_temp": meta["mean_temp"],
                "bad_lines": bad_lines,
                "captured_at": timestamp,
                "npz_path": str(npz_path).replace("\\", "/"),
            }
        )

    print(f"Saved frames to {npz_path}")
    print(f"Saved metadata to {meta_path}")


if __name__ == "__main__":
    main()
