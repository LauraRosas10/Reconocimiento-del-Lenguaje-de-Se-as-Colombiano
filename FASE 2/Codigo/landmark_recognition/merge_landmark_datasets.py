import argparse
from pathlib import Path

import numpy as np


def merge_datasets(
    input_paths: list[Path],
    output_path: Path,
    keep_classes: list[str] | None = None,
) -> None:
    if len(input_paths) < 2:
        raise ValueError("Provide at least two prepared datasets to merge.")

    datasets = []
    all_class_names: set[str] = set()
    expected_shape = None
    for path in input_paths:
        with np.load(path, allow_pickle=False) as archive:
            features = archive["X"].astype(np.float32)
            labels = archive["y"].astype(np.int64)
            signers = archive["signers"].astype(str)
            sample_ids = archive["sample_ids"].astype(str)
            class_names = archive["class_names"].astype(str).tolist()

        if features.ndim != 4 or features.shape[2:] != (75, 3):
            raise ValueError(f"{path} must contain X shaped [samples, frames, 75, 3], got {features.shape}.")
        sample_count = len(features)
        if not (
            len(labels) == sample_count
            and len(signers) == sample_count
            and len(sample_ids) == sample_count
        ):
            raise ValueError(f"{path} has inconsistent sample, label, signer, or sample-ID counts.")
        if np.any(labels < 0) or np.any(labels >= len(class_names)):
            raise ValueError(f"{path} contains labels outside its class_names array.")
        if expected_shape is None:
            expected_shape = features.shape[1:]
        elif features.shape[1:] != expected_shape:
            raise ValueError(
                f"Frame/landmark dimensions differ: expected {expected_shape}, "
                f"got {features.shape[1:]} in {path}."
            )
        datasets.append((path, features, labels, signers, sample_ids, class_names))
        all_class_names.update(class_names)

    if keep_classes is not None:
        requested_classes = set(keep_classes)
        missing_classes = requested_classes.difference(all_class_names)
        if missing_classes:
            raise ValueError(f"Requested classes are absent from every input dataset: {sorted(missing_classes)}")
        merged_class_names = sorted(requested_classes)
    else:
        merged_class_names = sorted(all_class_names)
    class_to_index = {name: index for index, name in enumerate(merged_class_names)}
    merged_features = []
    merged_labels = []
    merged_signers = []
    merged_sample_ids = []

    for path, features, labels, signers, sample_ids, class_names in datasets:
        label_names = np.asarray(class_names, dtype=str)[labels]
        keep_mask = np.isin(label_names, merged_class_names)
        if not np.any(keep_mask):
            continue
        merged_features.append(features[keep_mask])
        merged_labels.append(
            np.asarray([class_to_index[name] for name in label_names[keep_mask]], dtype=np.int64)
        )
        merged_signers.append(np.asarray([f"{path.stem}::{signer}" for signer in signers[keep_mask]], dtype=str))
        merged_sample_ids.append(
            np.asarray([f"{path.stem}/{sample_id}" for sample_id in sample_ids[keep_mask]], dtype=str)
        )

    if not merged_features:
        raise ValueError("No samples remained after filtering the requested classes.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        X=np.concatenate(merged_features),
        y=np.concatenate(merged_labels),
        signers=np.concatenate(merged_signers),
        sample_ids=np.concatenate(merged_sample_ids),
        class_names=np.asarray(merged_class_names, dtype=str),
    )
    total_sequences = sum(len(features) for features in merged_features)
    print(
        f"Saved {total_sequences} sequences from {len(input_paths)} datasets, "
        f"{len(merged_class_names)} classes to {output_path}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge prepared landmark datasets while remapping labels by class name."
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--keep-classes",
        nargs="+",
        default=None,
        help="Optional exact class-name allowlist. By default all classes are kept.",
    )
    args = parser.parse_args()
    merge_datasets(args.inputs, args.output, keep_classes=args.keep_classes)


if __name__ == "__main__":
    main()
