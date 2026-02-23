import json
import random
import argparse
from pathlib import Path
import shutil
from collections import Counter
def merge_all_splits(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    train_dir = input_root / "train"
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    print(f"Found {len(classes)} classes: {classes}")

    # create class folders
    for cls in classes:
        (output_root / cls).mkdir(parents=True, exist_ok=True)

    class_counts = Counter()

    for split in ["train", "val", "test"]:
        split_dir = input_root / split

        if not split_dir.exists():
            print(f"Warning: {split_dir} not found")
            continue

        print(f"\nMerging {split}...")

        for cls in classes:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue

            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue

                # prefix keeps filenames unique
                new_name = f"{split}_{img_path.name}"
                dst = output_root / cls / new_name

                shutil.copy2(img_path, dst)
                class_counts[cls] += 1

    print("\nMerged counts:")
    total = sum(class_counts.values())
    for cls, count in class_counts.items():
        print(f"{cls:20s}: {count:4d} ({count/total*100:.2f}%)")

    return class_counts

def stratified_split(
    merged_root,
    output_root,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    random.seed(seed)

    merged_root = Path(merged_root)
    output_root = Path(output_root)

    classes = sorted([d.name for d in merged_root.iterdir() if d.is_dir()])

    # create folders
    for split in ["train", "val", "test"]:
        for cls in classes:
            (output_root / split / cls).mkdir(parents=True, exist_ok=True)

    stats = {
        "train": Counter(),
        "val": Counter(),
        "test": Counter(),
    }

    print("\nCreating stratified split:")

    for cls in classes:
        samples = sorted((merged_root / cls).glob("*"))
        random.shuffle(samples)

        n_total = len(samples)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_samples = samples[:n_train]
        val_samples = samples[n_train:n_train+n_val]
        test_samples = samples[n_train+n_val:]

        for s in train_samples:
            shutil.copy2(s, output_root / "train" / cls / s.name)
            stats["train"][cls] += 1

        for s in val_samples:
            shutil.copy2(s, output_root / "val" / cls / s.name)
            stats["val"][cls] += 1

        for s in test_samples:
            shutil.copy2(s, output_root / "test" / cls / s.name)
            stats["test"][cls] += 1

        print(
            f"{cls:20s}: "
            f"{len(train_samples):4d} / "
            f"{len(val_samples):4d} / "
            f"{len(test_samples):4d}"
        )

    print("\nSplit totals:")
    for split in stats:
        print(f"{split:5s}: {sum(stats[split].values())}")

    return stats
def compute_class_weights(dataset_root, save_path):
    dataset_root = Path(dataset_root)

    train_dir = dataset_root / "train"
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])

    counts = {}
    for cls in classes:
        counts[cls] = len(list((train_dir / cls).glob("*")))

    total = sum(counts.values())
    num_classes = len(classes)

    weights = {
        cls: total / (num_classes * max(count,1))
        for cls, count in counts.items()
    }

    s = sum(weights.values())
    weights = {k: v * num_classes / s for k, v in weights.items()}

    print("\nClass weights:")
    for k, v in weights.items():
        print(f"  {k:20s}: {v:.3f}")

    with open(save_path, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\n✓ Saved weights → {save_path}")

    return weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dataset split and weight computation"
    )

    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep merged dataset"
    )

    return parser.parse_args()
def main():
    args=parse_args()
    input_root = Path('../data/dataset_preprocessed')
    temp_merged = Path('../data/dataset_merged_temp')
    output_root = Path('../data/dataset_balanced')
    total_ration=args.train_ratio+args.val_ratio+args.test_ratio
    if not abs(args.train_ratio +
                   args.val_ratio +
                   args.test_ratio - 1.0) < 1e-6:
            raise ValueError(f"Ratios sum to {total_ration} Split ratios must sum to 1")
    print("="*80)
    print("DATASET REBALANCING")
    print("="*80)
    print("\nStep 1: Merging all splits...")
    if temp_merged.exists():
        shutil.rmtree(temp_merged)
    class_counts = merge_all_splits(input_root, temp_merged)
    print("\nStep 2: Creating stratified split...")

    if output_root.exists():
        print(f"Removing existing {output_root}")
        shutil.rmtree(output_root)

    split_stats = stratified_split(
        temp_merged,
        output_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    print("\nStep 3: Computing class weights...")
    weights_path = output_root / "class_weights.json"
    compute_class_weights(output_root, weights_path)
    print(f"\nCleaning up temporary directory...")
    shutil.rmtree(temp_merged)
    print("\n" + "="*80)
    print("REBALANCING COMPLETE")
    print("="*80)
    print(f"\n✓ Balanced dataset created at: {output_root}")
    print(f"\nTo use the balanced dataset, update config.py:")
    print(f"""
    TRAIN_DIR = "../data/dataset_balanced/train"
    VAL_DIR = "../data/dataset_balanced/val"
    TEST_DIR = "../data/dataset_balanced/test"
    """)
if __name__ == "__main__":
    main()
