#!/usr/bin/env python3
from __future__ import annotations

import argparse

from codewam.codebook_eval.evaluation import evaluate_frozen_codebooks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen causal RQ codebooks on val/test episodes."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    report = evaluate_frozen_codebooks(args.config)
    for row in report["rows"]:
        usage = ", ".join(
            f"L{value['level']}={value['perplexity_fraction']:.3f}"
            for value in row["code_usage"]
        )
        print(
            f"{row['family']} {row['split']}: N={row['vectors']} "
            f"initial_mse={row['residual_mse'][0]:.4f} "
            f"total_reduction={row['residual_total_reduction']:.3f} "
            f"perplexity=[{usage}]"
        )


if __name__ == "__main__":
    main()
