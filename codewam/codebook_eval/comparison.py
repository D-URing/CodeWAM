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


def _format_optional_levels(values: Iterable[float | None]) -> str:
    return "/".join(
        "n/a" if value is None else _format_percent(float(value))
        for value in values
    )


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
                "## Held-out association by prefix",
                "",
                "| run | family | split | target | best prefix | best gain | "
                "full-prefix gain | full exact coverage |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in report["association_rows"]:
            lines.append(
                "| {label} | {family} | {split} | {target} | L{best_depth} | "
                "{best_gain} | {full_gain} | {exact} |".format(
                    label=row["label"],
                    family=row["family"],
                    split=row["split"],
                    target=row["target"],
                    best_depth=row["best_prefix_depth"],
                    best_gain=_format_percent(
                        row["best_normalized_mse_reduction"]
                    ),
                    full_gain=_format_percent(
                        row["full_prefix_normalized_mse_reduction"]
                    ),
                    exact=_format_percent(
                        row["full_prefix_exact_coverage"]
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
    if report["concentration_rows"]:
        lines.extend(
            [
                "## Held-out context concentration",
                "",
                "Higher scene or institution concentration can indicate "
                "background memorization. Exact-task concentration is reported "
                "without semantic task merging.",
                "",
                "| run | family | split | grouping | cross-parent accuracy "
                "gain L1/.../Ln | exact-code coverage L1/.../Ln | descriptive "
                "information gain L1/.../Ln | eligible vectors | groups |",
                "|---|---|---|---|---|---|---|---:|---:|",
            ]
        )
        for row in report["concentration_rows"]:
            lines.append(
                "| {label} | {family} | {split} | {grouping} | {prediction} "
                "| {coverage} | {information} | {eligible} | {groups} |".format(
                    label=row["label"],
                    family=row["family"],
                    split=row["split"],
                    grouping=row["grouping"],
                    prediction=_format_optional_levels(
                        row[
                            "cross_parent_normalized_accuracy_gain_by_prefix"
                        ]
                    ),
                    coverage=_format_optional_levels(
                        row["cross_parent_exact_code_coverage_by_prefix"]
                    ),
                    information=_format_levels(
                        row["group_information_gain_by_prefix"]
                    ),
                    eligible=_format_percent(
                        row["cross_parent_eligible_vector_fraction"]
                    ),
                    groups=row["groups"],
                )
            )
        lines.append("")
    else:
        lines.extend(
            [
                "No frozen context-concentration reports were present in the "
                "run directories.",
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
    concentration_rows: list[dict[str, Any]] = []
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
            "concentration_report_sha256": None,
            "concentration_contract_hash": None,
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
            selected_association_rows = []
            for row in association.get("rows", ()):
                if (
                    selected_families is not None
                    and row["family"] not in selected_families
                ):
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
                selected_association_rows.append(row)
            groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
            for row in selected_association_rows:
                groups.setdefault(
                    (row["family"], row["split"], row["target"]),
                    [],
                ).append(row)
            for (family, split, target), group in groups.items():
                full = [
                    row
                    for row in group
                    if int(row["prefix_depth"]) == int(row["levels"])
                ]
                if len(full) != 1:
                    raise RuntimeError(
                        f"Association group has no unique full prefix for `{label}`."
                    )
                best = max(
                    group,
                    key=lambda row: (
                        float(row["normalized_mse_reduction"]),
                        -int(row["prefix_depth"]),
                    ),
                )
                association_rows.append(
                    {
                        "label": label,
                        "run_dir": str(run_dir.resolve()),
                        "family": family,
                        "split": split,
                        "target": target,
                        "best_prefix_depth": int(best["prefix_depth"]),
                        "best_normalized_mse_reduction": float(
                            best["normalized_mse_reduction"]
                        ),
                        "full_prefix_depth": int(full[0]["prefix_depth"]),
                        "full_prefix_normalized_mse_reduction": float(
                            full[0]["normalized_mse_reduction"]
                        ),
                        "full_prefix_exact_coverage": float(
                            full[0]["exact_prefix_coverage"]
                        ),
                        "full_prefix_any_code_coverage": float(
                            full[0]["any_code_coverage"]
                        ),
                    }
                )
        concentration_path = (
            run_dir / "concentration/concentration_report.json"
        )
        if concentration_path.is_file():
            concentration = _load_json(concentration_path)
            input_row["concentration_report_sha256"] = file_sha256(
                concentration_path
            )
            input_row["concentration_contract_hash"] = concentration.get(
                "contract_hash"
            )
            selected_concentration_rows = []
            for row in concentration.get("rows", ()):
                if (
                    selected_families is not None
                    and row["family"] not in selected_families
                ):
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
                        "Concentration row does not match held-out run "
                        f"`{label}`."
                    )
                selected_concentration_rows.append(row)
            groups: dict[
                tuple[str, str, str],
                list[dict[str, Any]],
            ] = {}
            for row in selected_concentration_rows:
                groups.setdefault(
                    (row["family"], row["split"], row["grouping"]),
                    [],
                ).append(row)
            for (family, split, grouping), group in groups.items():
                ordered = sorted(
                    group,
                    key=lambda row: int(row["prefix_depth"]),
                )
                levels = int(ordered[0]["levels"])
                if [int(row["prefix_depth"]) for row in ordered] != list(
                    range(1, levels + 1)
                ):
                    raise RuntimeError(
                        "Concentration group has incomplete RQ prefixes for "
                        f"`{label}`."
                    )
                if any(
                    int(row["groups"]) != int(ordered[0]["groups"])
                    for row in ordered
                ):
                    raise RuntimeError(
                        f"Concentration group counts differ for `{label}`."
                    )
                if any(
                    float(row["missing_group_fraction"])
                    != float(ordered[0]["missing_group_fraction"])
                    for row in ordered
                ):
                    raise RuntimeError(
                        f"Concentration missing fractions differ for `{label}`."
                    )
                if any(
                    float(row["cross_parent_eligible_vector_fraction"])
                    != float(
                        ordered[0][
                            "cross_parent_eligible_vector_fraction"
                        ]
                    )
                    for row in ordered
                ):
                    raise RuntimeError(
                        "Concentration eligible fractions differ for "
                        f"`{label}`."
                    )
                concentration_rows.append(
                    {
                        "label": label,
                        "run_dir": str(run_dir.resolve()),
                        "family": family,
                        "split": split,
                        "grouping": grouping,
                        "levels": levels,
                        "groups": int(ordered[0]["groups"]),
                        "missing_group_fraction": float(
                            ordered[0]["missing_group_fraction"]
                        ),
                        "cross_parent_eligible_vector_fraction": float(
                            ordered[0][
                                "cross_parent_eligible_vector_fraction"
                            ]
                        ),
                        "group_information_gain_by_prefix": [
                            float(row["group_information_gain"])
                            for row in ordered
                        ],
                        "normalized_mutual_information_by_prefix": [
                            float(row["normalized_mutual_information"])
                            for row in ordered
                        ],
                        "normalized_purity_gain_by_prefix": [
                            float(row["normalized_purity_gain"])
                            for row in ordered
                        ],
                        "cross_parent_normalized_accuracy_gain_by_prefix": [
                            (
                                None
                                if row[
                                    "cross_parent_normalized_accuracy_gain"
                                ]
                                is None
                                else float(
                                    row[
                                        "cross_parent_normalized_accuracy_gain"
                                    ]
                                )
                            )
                            for row in ordered
                        ],
                        "cross_parent_exact_code_coverage_by_prefix": [
                            (
                                None
                                if row[
                                    "cross_parent_exact_code_coverage"
                                ]
                                is None
                                else float(
                                    row[
                                        "cross_parent_exact_code_coverage"
                                    ]
                                )
                            )
                            for row in ordered
                        ],
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
        "concentration_rows": sorted(
            concentration_rows,
            key=lambda row: (
                row["label"],
                row["family"],
                row["split"],
                row["grouping"],
            ),
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(output_dir / "comparison_report.json", report)
    markdown_path = output_dir / "comparison_report.md"
    _write_text(markdown_path, _markdown(report))
    return report
