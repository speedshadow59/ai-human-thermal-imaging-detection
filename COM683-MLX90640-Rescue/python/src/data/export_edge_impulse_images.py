import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def normalize_to_u8(frame: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
    clipped = np.clip(frame, min_temp, max_temp)
    norm = (clipped - min_temp) / max(max_temp - min_temp, 1e-6)
    return (norm * 255.0).astype(np.uint8)


def load_raw_frames(raw_root: Path):
    samples = []
    labels = []

    for npz_file in sorted(raw_root.rglob("*.npz")):
        data = np.load(npz_file)
        frames = data["frames"].astype(np.float32)
        label = str(data["label"]) if "label" in data else npz_file.parent.name
        for f in frames:
            samples.append(f)
            labels.append(label)

    if not samples:
        raise RuntimeError(f"No .npz files found under {raw_root}")

    return np.stack(samples, axis=0), np.array(labels)


def save_split(X: np.ndarray, y: np.ndarray, split_root: Path, min_temp: float, max_temp: float):
    split_root.mkdir(parents=True, exist_ok=True)
    counters = {}

    for frame, label in zip(X, y):
        label_dir = split_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        idx = counters.get(label, 0)
        counters[label] = idx + 1

        img_u8 = normalize_to_u8(frame, min_temp, max_temp)
        image = Image.fromarray(img_u8, mode="L")
        image.save(label_dir / f"{label}_{idx:06d}.png")


def main():
    parser = argparse.ArgumentParser(description="Export MLX90640 captures as Edge Impulse image dataset")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-dir", default="data/edge_impulse")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-temp", type=float, default=10.0)
    parser.add_argument("--max-temp", type=float, default=80.0)
    args = parser.parse_args()

    val_ratio = 1.0 - args.train_ratio - args.test_ratio
    if val_ratio <= 0:
        raise ValueError("train_ratio + test_ratio must be less than 1.0")

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)

    X, y = load_raw_frames(raw_root)

    X_train, X_other, y_train, y_other = train_test_split(
        X,
        y,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=y,
    )

    test_size_relative = args.test_ratio / (args.test_ratio + val_ratio)
    X_test, X_val, y_test, y_val = train_test_split(
        X_other,
        y_other,
        test_size=(1.0 - test_size_relative),
        random_state=args.seed,
        stratify=y_other,
    )

    save_split(X_train, y_train, out_dir / "training", args.min_temp, args.max_temp)
    save_split(X_test, y_test, out_dir / "testing", args.min_temp, args.max_temp)
    save_split(X_val, y_val, out_dir / "validation", args.min_temp, args.max_temp)

    labels = sorted(set(y.tolist()))
    summary = {
        "labels": labels,
        "normalization": {"min_temp": args.min_temp, "max_temp": args.max_temp},
        "splits": {
            "training": int(X_train.shape[0]),
            "testing": int(X_test.shape[0]),
            "validation": int(X_val.shape[0]),
        },
    }

    with (out_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Edge Impulse dataset export complete")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
