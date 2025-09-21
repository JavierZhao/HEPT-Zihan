import argparse
from pathlib import Path
import sys
import numpy as np


def add_src_to_path():
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def summarize_array(name, arr_list):
    if len(arr_list) == 0:
        print(f"{name}: <empty>")
        return
    arr = np.asarray(arr_list)
    print(
        f"{name}: count={arr.size}, min={arr.min()}, p50={np.median(arr):.1f}, p90={np.percentile(arr,90):.1f}, max={arr.max()}, mean={arr.mean():.1f}, std={arr.std():.1f}"
    )


def inspect_tracking(dataset, max_events):
    from datasets.tracking import Tracking

    print("Dataset type: Tracking")
    print(f"Num graphs: {len(dataset)}")
    print(f"x_dim: {dataset.x_dim}, coords_dim: {dataset.coords_dim}")
    if hasattr(dataset, "idx_split"):
        print(
            "Split sizes:",
            {k: len(v) for k, v in dataset.idx_split.items()},
        )

    num_nodes, num_edges, num_pairs = [], [], []
    has_keys = {}
    scan_n = min(len(dataset), max_events)
    for i in range(scan_n):
        data = dataset[i]
        # Track key presence
        for k in [
            "x",
            "pos",
            "coords",
            "edge_index",
            "point_pairs_index",
            "particle_id",
            "reconstructable",
            "pt",
        ]:
            has_keys[k] = has_keys.get(k, 0) + int(hasattr(data, k))

        n = data.num_nodes
        num_nodes.append(n)
        if getattr(data, "edge_index", None) is not None:
            num_edges.append(int(data.edge_index.size(1)))
        if getattr(data, "point_pairs_index", None) is not None:
            num_pairs.append(int(data.point_pairs_index.size(1)))

    print("Field presence (over first", scan_n, "events):")
    for k, v in has_keys.items():
        print(f"  {k}: {v}/{scan_n}")
    summarize_array("num_nodes", num_nodes)
    if num_edges:
        summarize_array("num_edges", num_edges)
    if num_pairs:
        summarize_array("num_point_pairs", num_pairs)

    # Show shapes for the first item
    if len(dataset) > 0:
        d0 = dataset[0]
        print("\nFirst sample shapes:")
        for k in d0.keys:
            t = getattr(d0, k)
            try:
                print(f"  {k}: {tuple(t.shape)} {t.dtype}")
            except Exception:
                print(f"  {k}: <non-tensor>")


def inspect_pileup(dataset, max_events):
    from datasets.pileup import Pileup

    print("Dataset type: Pileup")
    print(f"Num graphs: {len(dataset)}")
    if hasattr(dataset, "idx_split"):
        print(
            "Split sizes:",
            {k: len(v) for k, v in dataset.idx_split.items()},
        )

    num_nodes = []
    scan_n = min(len(dataset), max_events)
    for i in range(scan_n):
        data = dataset[i]
        num_nodes.append(data.num_nodes)
    summarize_array("num_nodes", num_nodes)

    if len(dataset) > 0:
        d0 = dataset[0]
        print("\nFirst sample shapes:")
        for k in d0.keys:
            t = getattr(d0, k)
            try:
                print(f"  {k}: {tuple(t.shape)} {t.dtype}")
            except Exception:
                print(f"  {k}: <non-tensor>")


def main():
    parser = argparse.ArgumentParser(
        description="Explore processed dataset (shapes and stats)"
    )
    parser.add_argument(
        "-d",
        "--dataset_name",
        type=str,
        required=True,
        choices=["tracking-6k", "tracking-60k", "pileup"],
        help="Dataset to explore",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Root data directory",
    )
    parser.add_argument(
        "--data_suffix",
        type=str,
        default="10percent",
        help="Processed file suffix to load (e.g., '10percent' for data-*-10percent.pt)",
    )
    parser.add_argument(
        "--max_events",
        type=int,
        default=200,
        help="Max number of events to scan for stats",
    )
    args = parser.parse_args()

    add_src_to_path()
    from datasets.tracking import Tracking
    from datasets.pileup import Pileup, PileupTransform

    root = Path(args.data_dir)
    if "tracking" in args.dataset_name:
        ds_root = root / "tracking"
        dataset = Tracking(ds_root, args.dataset_name, data_suffix=args.data_suffix)
        inspect_tracking(dataset, args.max_events)
    else:
        ds_root = root / "pileup"
        # Pileup does not use suffix by default; simply load the processed dataset
        dataset = Pileup(ds_root, transform=PileupTransform())
        inspect_pileup(dataset, args.max_events)


if __name__ == "__main__":
    main()
