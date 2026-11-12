"""Write the 40 hybrid invoices and their deterministic labels.

    python -m scripts.generate_hybrid_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.hybrid_dataset import HYBRID_SEED, generate_hybrid_invoices, write_hybrid_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate 40 fictional hybrid training invoices.")
    parser.add_argument("--seed", type=int, default=HYBRID_SEED)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/hybrid"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    invoices = generate_hybrid_invoices(args.seed)
    write_hybrid_dataset(args.output_dir, invoices)
    groups: dict[str, int] = {}
    for invoice in invoices:
        groups[invoice.template_group] = groups.get(invoice.template_group, 0) + 1
    print(f"Generated {len(invoices)} hybrid invoices (seed={args.seed}).")
    print(f"Ids {invoices[0].document_id} through {invoices[-1].document_id}.")
    print("Groups: " + ", ".join(f"{name}={count}" for name, count in groups.items()))
    print(f"Wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
