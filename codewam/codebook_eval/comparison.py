from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from codewam.data.droid_manifest import write_json_report

from .shards import file_sha256


COMPARISON_SCHEMA = "codewam.codebook-comparison.v1"


def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Missing comparison input `{path}`.")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _camera_label(camera_ids: list[str] | None) -> str:
    return "all" if camera_ids is None else "+".join(camera_ids)


def _summarize_row(
    *,
    label: str,
    run_dir: Path,
    train: dict[str, Any],
    heldout: dict[str, Any],
) -> dict[str, Any]:
    identity_fields = ("family", "stride", "pool", "camera_ids", "k", "levels")
    mismatches = [
        field
        for field in identity_fields
        if train.get(field) != heldout.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            f"Train/held-out metadata differ for `{label}` in {mismatches}."
        )
    if int(train["dim"]) != int(heldout["dimension"]):
        raise RuntimeError(
            f"Train/held-out dimensions differ for `{label}`."
        )
    code_usage = heldout["code_usage"]
    temporal = heldout["temporal"]
    if len(code_usage) != int(heldout["levels"]):
        raise ValueError(f"`{label}` has incomplete code-usage levels.")
    if len(temporal) != int(heldout["levels"]):
        raise ValueError(f"`{label}` has incomplete temporal levels.")

    train_reduction = float(train["residual_total_reduction"])
    heldout_reduction = float(heldout["residual_total_reduction"])
    return {
        "label": label,
        "run_dir": str(run_dir.resolve()),
        "family": heldout["family"],
        "split": heldout["split"],
        "camera_ids": heldout.get("camera_ids"),
        "stride": int(heldout["stride"]),
        "pool": int(heldout["pool"]),
        "dimension": int(heldout["dimension"]),
        "k": int(heldout["k"]),
        "levels": int(heldout["levels"]),
        "train_vectors": int(train["normalization_count"]),
        "heldout_vectors": int(heldout["vectors"]),
        "heldout_episodes": int(heldout["episodes"]),
        "iterations_per_level": [
            int(value) for value in train["iterations_per_level"]
        ],
        "train_residual_total_reduction": train_reduction,
        "heldout_residual_total_reduction": heldout_reduction,
        "generalization_gap": train_reduction - heldout_reduction,
        "heldout_residual_reduction_by_level": [
            float(value)
            for value in heldout["residual_reduction_by_level"]
        ],
        "active_codes_by_level": [
            int(value["active_codes"]) for value in code_usage
        ],
        "dead_fraction_by_level": [
            float(value["dead_fraction"]) for value in code_usage
        ],
        "perplexity_fraction_by_level": [
            float(value["perplexity_fraction"]) for value in code_usage
        ],
        "maximum_cluster_fraction_by_level": [
            float(value["maximum_cluster_fraction"]) for value in code_usage
        ],
        "same_next_fraction_by_level": [
            float(value["same_next_fraction"]) for value in temporal
        ],
        "change_next_fraction_by_level": [
            float(value["change_next_fraction"]) for value in temporal
        ],
        "transition_perplexity_by_level": [
            float(value["transition_perplexity"]) for value in temporal
        ],
        "joint_active_capacity_fraction": float(
            heldout["joint_usage"]["active_capacity_fraction"]
        ),
        "joint_perplexity_fraction": float(
            heldout["joint_usage"]["perplexity_fraction"]
        ),
        "maximum_tuple_fraction": float(
            heldout["joint_usage"]["maximum_tuple_fraction"]
        ),
    }


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _format_levels(values: Iterable[float], *, percent: bool = True) -> str:
    if percent:
        return "/".join(_format_percent(float(value)) for value in values)
    return "/".join(f"{float(value):.2f}" for value in values)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Streaming RQ comparison",
        "",
        "Distortion is measured in each candidate's own normalized descriptor "
        "space. It is useful within a controlled comparison, but it is not an "
        "automatic winner criterion across different cameras or dimensions.",
        "",
        "| run | family | split | cameras | g | D | train reduction | held-out "
        "reduction | gap | active codes | perplexity/K | max cluster | joint "
        "active | joint perplexity/K | change L1/L2/L3 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            "| {label} | {family} | {split} | {cameras} | {pool} | "
            "{dimension} | {train} | {heldout} | {gap} | {active} | "
            "{perplexity} | {maximum} | {joint_active} | {joint_perplexity} | "
            "{change} |".format(
                label=row["label"],
                family=row["family"],
                split=row["split"],
                cameras=_camera_label(row["camera_ids"]),
                pool=row["pool"],
                dimension=row["dimension"],
                train=_format_percent(
                    row["train_residual_total_reduction"]
                ),
                heldout=_format_percent(
                    row["heldout_residual_total_reduction"]
                ),
                gap=_format_percent(row["generalization_gap"]),
                active="/".join(
                    str(value) for value in row["active_codes_by_level"]
                ),
                perplexity=_format_levels(
                    row["perplexity_fraction_by_level"]
                ),
                maximum=_format_levels(
                    row["maximum_cluster_fraction_by_level"]
                ),
                joint_active=_format_percent(
                    row["joint_active_capacity_fraction"]
                ),
                joint_perplexity=_format_percent(
                    row["joint_perplexity_fraction"]
                ),
                change=_format_levels(
                    row["change_next_fraction_by_level"]
                ),
            )
        )
    lines.extend(
        [
            "",
            "Distortion alone leaves scene/camera concentration, geometry and "
            "retrieval quality unresolved.",
            "",
        ]
    )
    if report["association_rows"]:
        lines.extend(
            [
                "## Full-prefix held-out association",
                "",
                "| run | family | split | target | normalized MSE gain | "
                "exact tuple coverage | any prefix coverage |",
                "|---|---|---|---|---:|---:|---:|",
            ]
        )
        for row in report["association_rows"]:
            lines.append(
                "| {label} | {family} | {split} | {target} | {gain} | "
                "{exact} | {any_coverage} |".format(
                    label=row["label"],
                    family=row["family"],
                    split=row["split"],
                    target=row["target"],
                    gain=_format_percent(
                        row["normalized_mse_reduction"]
                    ),
                    exact=_format_percent(
                        row["exact_prefix_coverage"]
                    ),
                    any_coverage=_format_percent(
                        row["any_code_coverage"]
                    ),
                )
            )
        lines.append("")
    else:
        lines.extend(
            [
                "No frozen association reports were present in the run "
                "directories.",
                "",
            ]
        )
    return "\n".join(lines)


def compare_streaming_runs(
    runs: Iterable[tuple[str, str | Path]],
    output_dir: str | Path,
    *,
    families: Iterable[str] | None = None,
) -> dict[str, Any]:
    normalized = [(str(label), Path(path)) for label, path in runs]
    labels = [label for label, _ in normalized]
    if not normalized:
        raise ValueError("At least one streaming RQ run is required.")
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError("Comparison labels must be nonempty and unique.")
    selected_families = (
        None
        if families is None
        else tuple(str(family) for family in families)
    )
    if selected_families is not None and (
        not selected_families
        or len(selected_families) != len(set(selected_families))
    ):
        raise ValueError("Comparison families must be nonempty and unique.")

    rows: list[dict[str, Any]] = []
    association_rows: list[dict[str, Any]] = []
    inputs = []
    for label, run_dir in normalized:
        train_path = run_dir / "train_summary.json"
        heldout_path = run_dir / "heldout/evaluation_report.json"
        train_rows = _load_json(train_path)
        heldout_report = _load_json(heldout_path)
        if not isinstance(train_rows, list):
            raise ValueError(f"Train summary must be a list: `{train_path}`.")
        train_by_family = {
            str(row["family"]): row for row in train_rows
        }
        if len(train_by_family) != len(train_rows):
            raise ValueError(f"Duplicate train families in `{train_path}`.")
        heldout_rows = [
            row
            for row in heldout_report.get("rows", ())
            if (
                selected_families is None
                or row["family"] in selected_families
            )
        ]
        if not heldout_rows:
            raise ValueError(
                f"Held-out report has no selected rows: `{heldout_path}`."
            )
        heldout_identities = {
            (
                row["family"],
                row["split"],
                row["stride"],
                row["pool"],
                json.dumps(row.get("camera_ids"), sort_keys=True),
                row["k"],
                row["levels"],
            )
            for row in heldout_rows
        }
        for heldout in heldout_rows:
            family = str(heldout["family"])
            if family not in train_by_family:
                raise ValueError(
                    f"Held-out family `{family}` is absent from `{train_path}`."
                )
            rows.append(
                _summarize_row(
                    label=label,
                    run_dir=run_dir,
                    train=train_by_family[family],
                    heldout=heldout,
                )
            )
        input_row = {
            "label": label,
            "run_dir": str(run_dir.resolve()),
            "train_summary_sha256": file_sha256(train_path),
            "heldout_report_sha256": file_sha256(heldout_path),
            "heldout_contract_hash": heldout_report.get("contract_hash"),
            "association_report_sha256": None,
            "association_contract_hash": None,
        }
        association_path = run_dir / "association/association_report.json"
        if association_path.is_file():
            association = _load_json(association_path)
            input_row["association_report_sha256"] = file_sha256(
                association_path
            )
            input_row["association_contract_hash"] = association.get(
                "contract_hash"
            )
            for row in association.get("rows", ()):
                if (
                    selected_families is not None
                    and row["family"] not in selected_families
                ):
                    continue
                if int(row["prefix_depth"]) != int(row["levels"]):
                    continue
                identity = (
                    row["family"],
                    row["split"],
                    row["stride"],
                    row["pool"],
                    json.dumps(row.get("camera_ids"), sort_keys=True),
                    row["k"],
                    row["levels"],
                )
                if identity not in heldout_identities:
                    raise RuntimeError(
                        f"Association row does not match held-out run `{label}`."
                    )
                association_rows.append(
                    {
                        "label": label,
                        "run_dir": str(run_dir.resolve()),
                        "family": row["family"],
                        "split": row["split"],
                        "target": row["target"],
                        "prefix_depth": int(row["prefix_depth"]),
                        "normalized_mse_reduction": float(
                            row["normalized_mse_reduction"]
                        ),
                        "exact_prefix_coverage": float(
                            row["exact_prefix_coverage"]
                        ),
                        "any_code_coverage": float(
                            row["any_code_coverage"]
                        ),
                    }
                )
        inputs.append(input_row)

    report = {
        "schema": COMPARISON_SCHEMA,
        "families": (
            None
            if selected_families is None
            else list(selected_families)
        ),
        "inputs": inputs,
        "rows": sorted(
            rows,
            key=lambda row: (
                row["label"],
                row["family"],
                row["split"],
            ),
        ),
        "association_rows": sorted(
            association_rows,
            key=lambda row: (
                row["label"],
                row["family"],
                row["split"],
                row["target"],
            ),
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(output_dir / "comparison_report.json", report)
    markdown_path = output_dir / "comparison_report.md"
    _write_text(markdown_path, _markdown(report))
    return report
