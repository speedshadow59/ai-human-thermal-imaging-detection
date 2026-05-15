import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_label_map(path: Path) -> Dict[str, str]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def map_label(label: str, label_map: Dict[str, str]) -> str:
    return label_map.get(label, label)


def normalize_frame_to_u8(frame: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
    clipped = np.clip(frame, min_temp, max_temp)
    norm = (clipped - min_temp) / max(max_temp - min_temp, 1e-6)
    return (norm * 255.0).astype(np.uint8)


def ensure_size_grayscale(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.mode != "L":
        image = image.convert("L")
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)
    return image


def load_own_npz(raw_root: Path, label_map: Dict[str, str], min_temp: float, max_temp: float) -> List[Tuple[np.ndarray, str, str]]:
    samples: List[Tuple[np.ndarray, str, str]] = []
    for npz_path in sorted(raw_root.rglob("*.npz")):
        data = np.load(npz_path)
        frames = data["frames"].astype(np.float32)
        label_raw = str(data["label"]) if "label" in data else npz_path.parent.name
        label = map_label(label_raw, label_map)
        for frame in frames:
            image_arr = normalize_frame_to_u8(frame, min_temp, max_temp)
            samples.append((image_arr, label, "own"))
    return samples


def load_external_images(external_roots: List[Path], label_map: Dict[str, str], width: int, height: int) -> List[Tuple[np.ndarray, str, str]]:
    samples: List[Tuple[np.ndarray, str, str]] = []
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for root in external_roots:
        if not root.exists():
            raise FileNotFoundError(f"External directory does not exist: {root}")

        for class_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
            label_raw = class_dir.name
            label = map_label(label_raw, label_map)

            for image_path in class_dir.rglob("*"):
                if image_path.suffix.lower() not in valid_ext:
                    continue
                try:
                    img = Image.open(image_path)
                    img = ensure_size_grayscale(img, width, height)
                    arr = np.array(img, dtype=np.uint8)
                    samples.append((arr, label, root.name))
                except Exception:
                    continue

    return samples


def limit_per_label(samples: List[Tuple[np.ndarray, str, str]], max_per_label: int) -> List[Tuple[np.ndarray, str, str]]:
    if max_per_label <= 0:
        return samples

    counts: Dict[str, int] = {}
    limited: List[Tuple[np.ndarray, str, str]] = []
    for item in samples:
        _, label, _ = item
        current = counts.get(label, 0)
        if current < max_per_label:
            limited.append(item)
            counts[label] = current + 1
    return limited


def save_split(split_samples: List[Tuple[np.ndarray, str, str]], split_name: str, out_dir: Path, writer: csv.DictWriter):
    counters: Dict[str, int] = {}
    split_root = out_dir / split_name
    split_root.mkdir(parents=True, exist_ok=True)

    for arr, label, source in split_samples:
        label_dir = split_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        idx = counters.get(label, 0)
        counters[label] = idx + 1

        file_name = f"{label}_{idx:06d}_{source}.png"
        file_path = label_dir / file_name
        Image.fromarray(arr, mode="L").save(file_path)

        writer.writerow(
            {
                "split": split_name,
                "label": label,
                "source": source,
                "relative_path": str(file_path.relative_to(out_dir)).replace("\\", "/"),
            }
        )


def class_distribution(samples: List[Tuple[np.ndarray, str, str]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for _, label, _ in samples:
        dist[label] = dist.get(label, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: kv[0]))


def source_distribution(samples: List[Tuple[np.ndarray, str, str]]) -> Dict[str, int]:
    dist: Dict[str, int] = {}
    for _, _, source in samples:
        dist[source] = dist.get(source, 0) + 1
    return dict(sorted(dist.items(), key=lambda kv: kv[0]))


def main():
    parser = argparse.ArgumentParser(description="Create mixed Edge Impulse dataset from own MLX frames and public thermal images")
    parser.add_argument("--own-raw-root", default="data/raw", help="Path to own .npz captures")
    parser.add_argument("--external-dir", action="append", default=[], help="Path to external dataset root with class subfolders (repeatable)")
    parser.add_argument("--out-dir", default="data/edge_impulse_mixed")
    parser.add_argument("--label-map", default="", help="JSON file mapping source labels to target labels")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--height", type=int, default=24)
    parser.add_argument("--min-temp", type=float, default=10.0)
    parser.add_argument("--max-temp", type=float, default=80.0)
    parser.add_argument("--max-per-label", type=int, default=0, help="Optional cap per label to reduce imbalance (0 means unlimited)")
    args = parser.parse_args()

    val_ratio = 1.0 - args.train_ratio - args.test_ratio
    if val_ratio <= 0:
        raise ValueError("train_ratio + test_ratio must be less than 1.0")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(Path(args.label_map)) if args.label_map else {}

    samples: List[Tuple[np.ndarray, str, str]] = []

    own_root = Path(args.own_raw_root)
    if own_root.exists():
        samples.extend(load_own_npz(own_root, label_map, args.min_temp, args.max_temp))

    external_roots = [Path(p) for p in args.external_dir]
    if external_roots:
        samples.extend(load_external_images(external_roots, label_map, args.width, args.height))

    if not samples:
        raise RuntimeError("No samples found. Check own captures and external dataset paths.")

    samples = limit_per_label(samples, args.max_per_label)

    labels = np.array([s[1] for s in samples])
    idx = np.arange(len(samples))

    train_idx, other_idx = train_test_split(
        idx,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=labels,
    )

    other_labels = labels[other_idx]
    test_relative = args.test_ratio / (args.test_ratio + val_ratio)
    test_idx, val_idx = train_test_split(
        other_idx,
        test_size=(1.0 - test_relative),
        random_state=args.seed,
        stratify=other_labels,
    )

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(mf, fieldnames=["split", "label", "source", "relative_path"])
        writer.writeheader()

        save_split([samples[i] for i in train_idx], "training", out_dir, writer)
        save_split([samples[i] for i in test_idx], "testing", out_dir, writer)
        save_split([samples[i] for i in val_idx], "validation", out_dir, writer)

    summary = {
        "labels": sorted(set(labels.tolist())),
        "normalization": {
            "own_data_min_temp": args.min_temp,
            "own_data_max_temp": args.max_temp,
            "target_width": args.width,
            "target_height": args.height,
        },
        "counts": {
            "total": int(len(samples)),
            "training": int(len(train_idx)),
            "testing": int(len(test_idx)),
            "validation": int(len(val_idx)),
        },
        "class_distribution": class_distribution(samples),
        "source_distribution": source_distribution(samples),
        "label_map": label_map,
    }

    summary_path = out_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    print("Mixed Edge Impulse dataset created")
    print(f"Output: {out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
