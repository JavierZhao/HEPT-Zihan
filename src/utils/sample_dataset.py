import argparse
from pathlib import Path
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(
        description="Sample a fraction of the processed tracking dataset and save as a smaller file."
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        default="tracking-6k",
        choices=["tracking-6k", "tracking-60k"],
        help="Which processed dataset to sample from.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/j-jepa-vol/HEPT-Zihan/data",
        help="Root data directory (expects tracking/processed/<dataset_name>/data-*.pt).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.01,
        help="Fraction of events to sample (e.g., 0.01 for 1%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="If set, overwrite the original processed file (a .bak backup will be made).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="sample",
        help="Suffix to append to the output file when not using --inplace (e.g., data-6k-<suffix>.pt).",
    )
    args = parser.parse_args()

    # Lazy import to avoid importing torch_geometric unless needed
    from datasets.tracking import Tracking

    root = Path(args.data_dir) / "tracking"
    dataset = Tracking(root, args.dataset_name, debug=False)

    num_graphs = len(dataset)
    num_sample = max(1, int(round(num_graphs * args.ratio)))

    rng = np.random.default_rng(args.seed)
    selected = np.sort(rng.choice(num_graphs, size=num_sample, replace=False)).tolist()

    # Materialize sampled graphs and collate back to (data, slices)
    sampled_list = [dataset[i] for i in selected]
    sampled_data, sampled_slices = dataset.collate(sampled_list)

    size_tag = args.dataset_name.split("-")[-1]
    processed_dir = root / "processed" / args.dataset_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    src_file = processed_dir / f"data-{size_tag}.pt"

    if args.inplace:
        # Backup original and overwrite with the sampled data
        if src_file.exists():
            backup = processed_dir / f"data-{size_tag}.pt.bak"
            backup.write_bytes(src_file.read_bytes())
        torch.save((sampled_data, sampled_slices, {}), src_file)
        print(
            f"Overwrote {src_file} with a {num_sample}/{num_graphs} ({args.ratio:.2%}) sample. Backup saved as .bak."
        )
    else:
        dst_file = processed_dir / f"data-{size_tag}-{args.suffix}.pt"
        torch.save((sampled_data, sampled_slices, {}), dst_file)
        print(
            f"Saved sampled dataset to {dst_file} ({num_sample}/{num_graphs}, {args.ratio:.2%}).\n"
            f"To use it temporarily, you can swap filenames or pass --inplace to overwrite."
        )


if __name__ == "__main__":
    main()
