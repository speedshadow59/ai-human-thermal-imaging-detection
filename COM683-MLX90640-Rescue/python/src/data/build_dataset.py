import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def load_sessions(raw_root: Path):
    all_frames = []
    all_labels = []

    for npz_file in sorted(raw_root.rglob("*.npz")):
        d = np.load(npz_file)
        frames = d["frames"].astype(np.float32)
        label = str(d["label"]) if "label" in d else npz_file.parent.name

        all_frames.append(frames)
        all_labels.extend([label] * frames.shape[0])

    if not all_frames:
        raise RuntimeError(f"No .npz sessions found under {raw_root}")

    X = np.concatenate(all_frames, axis=0)
    y = np.array(all_labels)
    return X, y


def normalize_frames(X: np.ndarray, min_temp: float, max_temp: float):
    X = np.clip(X, min_temp, max_temp)
    X = (X - min_temp) / max(max_temp - min_temp, 1e-6)
    return X


def main():
    parser = argparse.ArgumentParser(description="Build train/val/test dataset from raw captures")
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-temp", type=float, default=10.0)
    parser.add_argument("--max-temp", type=float, default=80.0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_sessions(raw_root)
    X = normalize_frames(X, args.min_temp, args.max_temp)

    labels = sorted(set(y.tolist()))
    label_to_id = {label: i for i, label in enumerate(labels)}
    y_id = np.array([label_to_id[v] for v in y], dtype=np.int64)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y_id,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_id,
    )

    relative_val = args.val_size / (1.0 - args.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val,
        random_state=args.seed,
        stratify=y_train_val,
    )

    np.savez_compressed(
        out_dir / "dataset.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )

    with (out_dir / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "labels": labels,
                "label_to_id": label_to_id,
                "normalization": {
                    "min_temp": args.min_temp,
                    "max_temp": args.max_temp,
                },
            },
            f,
            indent=2,
        )

    print("Dataset built successfully")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Labels: {labels}")


if __name__ == "__main__":
    main()
