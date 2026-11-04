"""Generate the synthetic invoice dataset and write leakage-free splits.

Run from the project root:

    python -m scripts.generate_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.dataset import DATASET_SEED, DatasetBundle, build_dataset, write_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fictional DistillOps invoices and write train, val, and test splits."
    )
    parser.add_argument("--seed", type=int, default=DATASET_SEED, help="Random seed. Default: 42.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory that contains raw/, processed/, and splits/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    bundle = build_dataset(args.seed)
    write_dataset(args.data_dir, bundle)
    print(format_generation_summary(bundle, args.data_dir))
    return 0


def format_generation_summary(bundle: DatasetBundle, data_dir: Path) -> str:
    lines = [f"Generated {len(bundle.records)} synthetic invoices (seed={bundle.seed})."]
    for split_name in ("train", "val", "test"):
        groups = sorted({record.template_group for record in bundle.splits[split_name]})
        lines.append(
            f"{split_name}: {len(bundle.splits[split_name])} documents, groups={', '.join(groups)}"
        )
    lines.append(f"Wrote splits under {data_dir / 'splits'}")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
