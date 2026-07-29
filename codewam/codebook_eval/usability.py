from __future__ import annotations

import hashlib
import json
import os
import statistics
from pathlib import Path
from typing import Any, Iterable

from codewam.data.droid_manifest import write_json_report

from .shards import file_sha256


USABILITY_CONTRACT_SCHEMA = "codewam.rq-usability-contract.v1"
USABILITY_REPORT_SCHEMA = "codewam.rq-usability-report.v1"
GATE_ORDER = (
    "causal_reproduction",
    "quantizer_health",
    "rq_hierarchy",
    "family_complementarity",
    "context_leakage",
    "photometric_robustness",
    "geometry_sensitivity",
    "action_event_semantics",
    "seed_stability",
    "cross_domain_stress",
)
STATUS_RANK = {
    "pass": 0,
    "conditional": 1,
    "fail": 2,
    "not_run": 3,
}


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing usability input `{path}`.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Usability input must be an object: `{path}`.")
    return payload


def _input_row(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = _load_json(path)
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "schema": payload.get("schema"),
        "contract_hash": payload.get("contract_hash"),
    }


def _report_contract(
    report_path: str | Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    contract_path = Path(report_path).with_name("contract.json")
    contract = _load_json(contract_path)
    report_hash = report.get("contract_hash")
    contract_hash = contract.get("contract_hash")
    contract_payload = {
        key: value
        for key, value in contract.items()
        if key != "contract_hash"
    }
    if (
        not report_hash
        or report_hash != contract_hash
        or contract_hash != _canonical_hash(contract_payload)
    ):
        raise RuntimeError(
            f"Report `{report_path}` does not match `{contract_path}`."
        )
    return contract


def _artifact_hashes(
    contract: dict[str, Any],
    *,
    source: str,
) -> dict[str, str]:
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"{source} contract has no frozen artifacts.")
    result = {}
    for family, row in artifacts.items():
        if not isinstance(row, dict) or not row.get("sha256"):
            raise ValueError(
                f"{source} contract has invalid artifact `{family}`."
            )
        result[str(family)] = str(row["sha256"])
    return result


def _validate_artifact_identity(
    reference: dict[str, str],
    observed: dict[str, str],
    *,
    source: str,
) -> None:
    if observed != reference:
        missing = sorted(set(reference) - set(observed))
        extra = sorted(set(observed) - set(reference))
        changed = sorted(
            family
            for family in set(reference) & set(observed)
            if reference[family] != observed[family]
        )
        raise RuntimeError(
            f"{source} does not use the canonical frozen artifacts: "
            f"missing={missing}, extra={extra}, changed={changed}."
        )


def _manifest_fingerprint(
    contract: dict[str, Any],
    *,
    source: str,
    key: str = "manifest",
) -> str:
    manifest = contract.get(key)
    if not isinstance(manifest, dict) or not manifest.get("fingerprint"):
        raise ValueError(f"{source} contract has no manifest fingerprint.")
    return str(manifest["fingerprint"])


def _validate_manifest_identity(
    reference: str,
    observed: str,
    *,
    source: str,
) -> None:
    if observed != reference:
        raise RuntimeError(
            f"{source} uses pooled manifest `{observed}`, expected "
            f"`{reference}`."
        )


def _provenance_audit(
    *,
    comparison_report: str | Path,
    comparison: dict[str, Any],
    family_association_report: str | Path,
    family_association: dict[str, Any],
    retrieval_report: str | Path,
    retrieval: dict[str, Any],
    temporal_paths: tuple[Path, ...],
    temporal: list[dict[str, Any]],
    visual_paths: tuple[Path, ...],
    visual: list[dict[str, Any]],
    action_event_report: str | Path | None,
    action_events: dict[str, Any] | None,
    seed_stability_report: str | Path | None,
    stability: dict[str, Any] | None,
) -> dict[str, Any]:
    family_contract = _report_contract(
        family_association_report,
        family_association,
    )
    reference_artifacts = _artifact_hashes(
        family_contract,
        source="family association",
    )
    if set(reference_artifacts) != {"Q2", "Q3", "Q5"}:
        raise ValueError(
            "Canonical usability evidence must contain Q2/Q3/Q5."
        )
    reference_manifest = _manifest_fingerprint(
        family_contract,
        source="family association",
    )
    validated_sources = ["family_association"]

    comparison_artifacts = {}
    for row in comparison.get("inputs", ()):
        family = str(row["label"])
        artifact_path = Path(row["run_dir"]) / family / "codebook.pt"
        comparison_artifacts[family] = file_sha256(artifact_path)
    _validate_artifact_identity(
        reference_artifacts,
        comparison_artifacts,
        source="canonical comparison",
    )
    validated_sources.append("comparison")

    retrieval_contract = _report_contract(retrieval_report, retrieval)
    retrieval_manifest = _manifest_fingerprint(
        retrieval_contract,
        source="retrieval",
        key="pooled_manifest",
    )
    _validate_manifest_identity(
        reference_manifest,
        retrieval_manifest,
        source="retrieval",
    )
    retrieval_artifacts = {}
    for row in retrieval_contract.get("evaluation_reports", ()):
        evaluation_path = Path(row["path"])
        if file_sha256(evaluation_path) != row.get("sha256"):
            raise RuntimeError(
                f"Retrieval input `{evaluation_path}` changed after rendering."
            )
        evaluation = _load_json(evaluation_path)
        evaluation_contract = _report_contract(
            evaluation_path,
            evaluation,
        )
        _validate_manifest_identity(
            reference_manifest,
            _manifest_fingerprint(
                evaluation_contract,
                source="retrieval evaluation",
            ),
            source="retrieval evaluation",
        )
        retrieval_artifacts.update(
            _artifact_hashes(
                evaluation_contract,
                source="retrieval evaluation",
            )
        )
    _validate_artifact_identity(
        reference_artifacts,
        retrieval_artifacts,
        source="retrieval",
    )
    validated_sources.append("retrieval")

    for path, report in zip(temporal_paths, temporal):
        contract = _report_contract(path, report)
        _validate_artifact_identity(
            reference_artifacts,
            _artifact_hashes(contract, source="temporal sensitivity"),
            source=f"temporal sensitivity `{path}`",
        )
        _validate_manifest_identity(
            reference_manifest,
            _manifest_fingerprint(
                contract,
                source="temporal sensitivity",
            ),
            source=f"temporal sensitivity `{path}`",
        )
    validated_sources.append("temporal_sensitivity")

    for path, report in zip(visual_paths, visual):
        contract = _report_contract(path, report)
        _validate_artifact_identity(
            reference_artifacts,
            _artifact_hashes(contract, source="visual perturbation"),
            source=f"visual perturbation `{path}`",
        )
        if report.get("source") == "droid":
            source_contract = contract.get("source")
            if not isinstance(source_contract, dict):
                raise ValueError(
                    f"Visual report `{path}` has no DROID source contract."
                )
            _validate_manifest_identity(
                reference_manifest,
                _manifest_fingerprint(
                    source_contract,
                    source="DROID visual perturbation",
                    key="pooled_manifest",
                ),
                source=f"visual perturbation `{path}`",
            )
    if visual:
        validated_sources.append("visual_perturbation")

    if action_events is not None and action_event_report is not None:
        contract = _report_contract(action_event_report, action_events)
        _validate_artifact_identity(
            reference_artifacts,
            _artifact_hashes(contract, source="action events"),
            source="action events",
        )
        _validate_manifest_identity(
            reference_manifest,
            _manifest_fingerprint(contract, source="action events"),
            source="action events",
        )
        validated_sources.append("action_events")

    if stability is not None and seed_stability_report is not None:
        contract = _report_contract(seed_stability_report, stability)
        reference_run = str(stability.get("reference_run", ""))
        runs = contract.get("runs")
        if (
            not reference_run
            or not isinstance(runs, dict)
            or not isinstance(runs.get(reference_run), dict)
        ):
            raise ValueError(
                "Seed-stability contract has no canonical reference run."
            )
        seed_artifacts = {
            str(family): str(row["sha256"])
            for family, row in runs[reference_run].items()
        }
        _validate_artifact_identity(
            reference_artifacts,
            seed_artifacts,
            source="seed stability reference",
        )
        _validate_manifest_identity(
            reference_manifest,
            _manifest_fingerprint(contract, source="seed stability"),
            source="seed stability",
        )
        validated_sources.append("seed_stability")

    return {
        "pooled_manifest_fingerprint": reference_manifest,
        "canonical_artifact_sha256": reference_artifacts,
        "validated_sources": validated_sources,
    }


def _gate(
    name: str,
    status: str,
    *,
    title: str,
    evidence: dict[str, Any],
    criteria: list[str],
    conclusion: str,
) -> dict[str, Any]:
    if name not in GATE_ORDER:
        raise ValueError(f"Unknown usability gate `{name}`.")
    if status not in STATUS_RANK:
        raise ValueError(f"Unknown usability status `{status}`.")
    return {
        "name": name,
        "title": title,
        "status": status,
        "criteria": criteria,
        "evidence": evidence,
        "conclusion": conclusion,
    }


def _quantizer_health(comparison: dict[str, Any]) -> dict[str, Any]:
    rows = [
        row
        for row in comparison.get("rows", ())
        if row.get("family") in {"Q2", "Q3", "Q5"}
        and row.get("split") in {"val", "test"}
        and int(row.get("k", 0)) == 8
        and int(row.get("levels", 0)) == 3
    ]
    if len(rows) != 6:
        raise ValueError(
            "Canonical comparison must contain Q2/Q3/Q5 val/test rows."
        )
    minimum_perplexity = min(
        float(value)
        for row in rows
        for value in row["perplexity_fraction_by_level"]
    )
    maximum_cluster = max(
        float(value)
        for row in rows
        for value in row["maximum_cluster_fraction_by_level"]
    )
    minimum_total_reduction = min(
        float(row["heldout_residual_total_reduction"]) for row in rows
    )
    minimum_level_reduction = min(
        float(value)
        for row in rows
        for value in row["heldout_residual_reduction_by_level"]
    )
    maximum_gap = max(
        abs(float(row["generalization_gap"])) for row in rows
    )
    all_active = all(
        all(
            int(active) == int(row["k"])
            for active in row["active_codes_by_level"]
        )
        for row in rows
    )
    passed = (
        all_active
        and minimum_perplexity >= 0.80
        and maximum_cluster <= 0.25
        and minimum_total_reduction >= 0.25
        and minimum_level_reduction >= 0.04
        and maximum_gap <= 0.05
    )
    status = "pass" if passed else "fail"
    return _gate(
        "quantizer_health",
        status,
        title="Streaming RQ-KMeans health",
        evidence={
            "rows": len(rows),
            "all_levels_use_all_centers": all_active,
            "minimum_perplexity_fraction": minimum_perplexity,
            "maximum_cluster_fraction": maximum_cluster,
            "minimum_heldout_total_residual_reduction": (
                minimum_total_reduction
            ),
            "minimum_heldout_level_residual_reduction": (
                minimum_level_reduction
            ),
            "maximum_train_heldout_reduction_gap": maximum_gap,
        },
        criteria=[
            "all K=8 centers active at every level and held-out split",
            "minimum perplexity/K >= 0.80",
            "maximum cluster fraction <= 0.25",
            "held-out total residual reduction >= 0.25",
            "every RQ level contributes >= 0.04 residual reduction",
            "absolute train-held-out reduction gap <= 0.05",
        ],
        conclusion=(
            "The quantizer is numerically healthy and every residual level "
            "earns capacity."
            if passed
            else "At least one numerical health requirement failed."
        ),
    )


def _hierarchy_gate(
    temporal_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        row
        for report in temporal_reports
        for row in report.get("summary_rows", ())
        if row.get("perturbation") in {
            "reverse_time",
            "static_current",
        }
    ]
    reverse = [
        row for row in rows if row["perturbation"] == "reverse_time"
    ]
    static = [
        row for row in rows if row["perturbation"] == "static_current"
    ]
    if len(reverse) < 6 or len(static) < 6:
        return _gate(
            "rq_hierarchy",
            "not_run",
            title="RQ hierarchy and temporal sensitivity",
            evidence={"rows": len(rows)},
            criteria=["val/test reverse-time and static-current rows"],
            conclusion="Temporal hierarchy evidence is incomplete.",
        )
    minimum_reverse_full = min(
        float(row["full_prefix_change_fraction"]) for row in reverse
    )
    minimum_reverse_gap = min(
        float(row["full_prefix_change_fraction"])
        - float(row["l1_code_change_fraction"])
        for row in reverse
    )
    minimum_static_full = min(
        float(row["full_prefix_change_fraction"]) for row in static
    )
    passed = (
        minimum_reverse_full >= 0.20
        and minimum_reverse_gap >= 0.15
        and minimum_static_full >= 0.60
    )
    return _gate(
        "rq_hierarchy",
        "pass" if passed else "fail",
        title="RQ hierarchy and temporal sensitivity",
        evidence={
            "minimum_reverse_full_prefix_change": minimum_reverse_full,
            "minimum_reverse_full_minus_l1_change": minimum_reverse_gap,
            "minimum_static_current_full_prefix_change": minimum_static_full,
        },
        criteria=[
            "reverse-time full-prefix change >= 0.20",
            "reverse-time full-prefix minus L1 change >= 0.15",
            "static-current full-prefix change >= 0.60",
        ],
        conclusion=(
            "L1 behaves as a coarse content/state coordinate while later "
            "residual levels carry substantial temporal information."
            if passed
            else "The expected coarse-to-dynamic RQ hierarchy was not stable."
        ),
    )


def _complementarity_gate(
    family_association: dict[str, Any],
) -> dict[str, Any]:
    full_rows = [
        row
        for row in family_association.get("summary_rows", ())
        if int(row.get("prefix_depth", 0)) == 3
    ]
    expected = {
        (split, target)
        for split in ("val", "test")
        for target in (
            "current_action",
            "common_future_proprio_change",
            "common_future_latent_moment_change",
        )
    }
    observed = {
        (str(row["split"]), str(row["target"])) for row in full_rows
    }
    gains = [
        float(row["full_gain_over_best_single"]) for row in full_rows
    ]
    future_increments = [
        float(value)
        for row in full_rows
        if row["target"] != "current_action"
        for value in row["incremental_gain_by_family"].values()
    ]
    profiles = [
        row
        for row in family_association.get("profile_rows", ())
        if row.get("profile") == "policy-hybrid"
    ]
    hybrid_action = {
        row["split"]: float(row["normalized_mse_reduction"])
        for row in profiles
        if row["target"] == "current_action"
    }
    all_l3_action = {
        row["split"]: float(row["full_normalized_mse_reduction"])
        for row in full_rows
        if row["target"] == "current_action"
    }
    hybrid_delta = {
        split: hybrid_action[split] - all_l3_action[split]
        for split in set(hybrid_action) & set(all_l3_action)
    }
    passed = (
        observed == expected
        and gains
        and min(gains) > 0
        and future_increments
        and min(future_increments) > 0
        and set(hybrid_delta) == {"val", "test"}
        and min(hybrid_delta.values()) >= 0
    )
    return _gate(
        "family_complementarity",
        "pass" if passed else "fail",
        title="Q2/Q3/Q5 complementarity",
        evidence={
            "minimum_joint_gain_over_best_single": (
                min(gains) if gains else None
            ),
            "minimum_future_leave_one_family_increment": (
                min(future_increments) if future_increments else None
            ),
            "policy_hybrid_action_gain_over_all_l3": hybrid_delta,
        },
        criteria=[
            "joint L3 beats best single family for every target and split",
            "each family has positive leave-one-out value on future targets",
            "Q2-L3+Q3-L3+Q5-L2 does not reduce held-out action association",
        ],
        conclusion=(
            "The coprime temporal families are complementary, with role-"
            "specific depth routing preferred over uniform exposure."
            if passed
            else "The evidence does not justify retaining every family/depth."
        ),
    )


def _context_gate(
    comparison: dict[str, Any],
    retrieval: dict[str, Any],
) -> dict[str, Any]:
    scene_rows = [
        row
        for row in comparison.get("concentration_rows", ())
        if row.get("grouping") == "scene"
    ]
    full_scene_gains = [
        float(row["cross_parent_normalized_accuracy_gain_by_prefix"][-1])
        for row in scene_rows
        if row["cross_parent_normalized_accuracy_gain_by_prefix"][-1]
        is not None
    ]
    full_scene_nmi = [
        float(row["normalized_mutual_information_by_prefix"][-1])
        for row in scene_rows
    ]
    montages = retrieval.get("montages", ())
    diversity_fraction = [
        float(row["selection_summary"]["codes_with_full_diversity"])
        / len(row["codes"])
        for row in montages
    ]
    example_fraction = [
        float(row["selection_summary"]["examples"])
        / float(row["selection_summary"]["expected_examples"])
        for row in montages
    ]
    if not full_scene_gains or not diversity_fraction:
        return _gate(
            "context_leakage",
            "not_run",
            title="Scene leakage and cross-scene retrieval",
            evidence={},
            criteria=["scene concentration and scene-diverse retrieval"],
            conclusion="Context leakage evidence is incomplete.",
        )
    maximum_gain = max(full_scene_gains)
    maximum_nmi = max(full_scene_nmi)
    minimum_diversity = min(diversity_fraction)
    minimum_examples = min(example_fraction)
    status = (
        "fail"
        if maximum_gain > 0.30 or minimum_diversity < 0.50
        else "conditional"
        if maximum_gain > 0.15 or maximum_nmi > 0.35
        else "pass"
    )
    return _gate(
        "context_leakage",
        status,
        title="Scene leakage and cross-scene retrieval",
        evidence={
            "maximum_cross_parent_scene_accuracy_gain": maximum_gain,
            "maximum_scene_nmi": maximum_nmi,
            "minimum_codes_with_three_scene_retrieval_fraction": (
                minimum_diversity
            ),
            "minimum_selected_example_fraction": minimum_examples,
        },
        criteria=[
            "cross-parent scene accuracy gain <= 0.15 for pass; > 0.30 fails",
            "scene NMI <= 0.35 for pass",
            "at least half of codes retrieve three distinct scenes",
        ],
        conclusion=(
            "The codes remain usable, but scene and appearance information is "
            "material and must not be described as pure motion semantics."
            if status == "conditional"
            else "Scene leakage is within the current acceptance band."
            if status == "pass"
            else "Scene identity dominates too strongly for the intended use."
        ),
    )


def _causal_gate(
    causal_audit: dict[str, Any],
    visual_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    causal_passed = bool(
        causal_audit.get("comparison", {}).get("passed", False)
    )
    reproduction = [
        row
        for report in visual_reports
        if report.get("source") == "droid"
        for row in report.get("droid_reproduction_rows", ())
    ]
    minimum_match = (
        min(float(row["full_prefix_match_fraction"]) for row in reproduction)
        if reproduction
        else None
    )
    maximum_error = (
        max(float(row["maximum_pooled_absolute_error"]) for row in reproduction)
        if reproduction
        else None
    )
    if not reproduction:
        status = "not_run"
    else:
        status = (
            "pass"
            if causal_passed and minimum_match is not None and minimum_match >= 0.99
            else "fail"
        )
    return _gate(
        "causal_reproduction",
        status,
        title="Causal alignment and RGB-to-cache reproduction",
        evidence={
            "prefix_causality_audit_passed": causal_passed,
            "minimum_reencoded_full_prefix_match": minimum_match,
            "maximum_reencoded_pooled_absolute_error": maximum_error,
        },
        criteria=[
            "Wan prefix causality audit passes",
            "RGB re-encoding reproduces >= 99% of cached full RQ prefixes",
        ],
        conclusion=(
            "Raw RGB, causal Wan latent ticks, pooled cache and RQ codes share "
            "one verified time contract."
            if status == "pass"
            else "The end-to-end visual alignment contract is incomplete."
            if status == "not_run"
            else "RGB re-encoding does not reproduce the canonical cache."
        ),
    )


def _visual_gates(
    visual_reports: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    droid = [
        report for report in visual_reports if report.get("source") == "droid"
    ]
    libero = [
        report for report in visual_reports if report.get("source") == "libero"
    ]
    if not droid:
        missing = _gate(
            "photometric_robustness",
            "not_run",
            title="Photometric robustness",
            evidence={},
            criteria=["DROID RGB perturbation reports"],
            conclusion="Photometric robustness was not run.",
        )
        geometry = _gate(
            "geometry_sensitivity",
            "not_run",
            title="Translation and scale sensitivity",
            evidence={},
            criteria=["DROID RGB perturbation reports"],
            conclusion="Geometry sensitivity was not run.",
        )
    else:
        photo_groups = []
        geometry_groups = []
        dose_checks = []
        for report_index, report in enumerate(droid, start=1):
            clip_splits = sorted(
                {
                    str(clip["split"])
                    for clip in report.get("clips", ())
                    if clip.get("split") is not None
                }
            )
            report_label = (
                "+".join(clip_splits)
                if clip_splits
                else f"report-{report_index}"
            )
            families = sorted(
                {str(row["family"]) for row in report["condition_rows"]}
            )
            lookup = {
                (str(row["family"]), str(row["name"])): row
                for row in report["condition_rows"]
            }
            for family in families:
                family_rows = [
                    row
                    for row in report["condition_rows"]
                    if row["family"] == family
                ]
                photometric = [
                    row
                    for row in family_rows
                    if row["category"] == "photometric_nuisance"
                ]
                endpoint = [
                    row
                    for row in family_rows
                    if row["category"] == "endpoint_geometry"
                    and (
                        row["name"].endswith("_8")
                        or row["name"].startswith("endpoint_scale")
                    )
                ]
                strong_pairs = [
                    row
                    for row in report["direction_rows"]
                    if row["family"] == family
                    and (
                        "negative_8" in row["left_condition"]
                        or row["left_condition"] == "endpoint_scale_090"
                    )
                ]
                if not photometric or not endpoint or not strong_pairs:
                    raise ValueError(
                        f"Incomplete visual rows for {report_label}/{family}."
                    )
                photo_groups.append(
                    {
                        "report": report_label,
                        "family": family,
                        "l1_change_fraction": statistics.mean(
                            float(row["prefix_change_fraction"][0])
                            for row in photometric
                        ),
                        "full_prefix_change_fraction": statistics.mean(
                            float(row["prefix_change_fraction"][-1])
                            for row in photometric
                        ),
                        "input_relative_to_natural_next": statistics.mean(
                            float(
                                row[
                                    "relative_to_natural_next_displacement"
                                ]
                            )
                            for row in photometric
                            if row[
                                "relative_to_natural_next_displacement"
                            ]
                            is not None
                        ),
                        "quantized_relative_to_natural_next": (
                            statistics.mean(
                                float(
                                    row[
                                        "quantized_prefix_relative_to_natural_next"
                                    ][-1]
                                )
                                for row in photometric
                                if row[
                                    "quantized_prefix_relative_to_natural_next"
                                ][-1]
                                is not None
                            )
                        ),
                    }
                )
                geometry_groups.append(
                    {
                        "report": report_label,
                        "family": family,
                        "endpoint_full_prefix_change_fraction": (
                            statistics.mean(
                                float(row["prefix_change_fraction"][-1])
                                for row in endpoint
                            )
                        ),
                        "opposite_transform_prefix_distinct_fraction": (
                            statistics.mean(
                                float(
                                    row["prefix_distinct_fraction"][-1]
                                )
                                for row in strong_pairs
                            )
                        ),
                        "quantized_relative_to_natural_next": (
                            statistics.mean(
                                float(
                                    row[
                                        "quantized_prefix_relative_to_natural_next"
                                    ][-1]
                                )
                                for row in endpoint
                                if row[
                                    "quantized_prefix_relative_to_natural_next"
                                ][-1]
                                is not None
                            )
                        ),
                    }
                )
                four = statistics.mean(
                    float(
                        lookup[
                            (
                                family,
                                f"endpoint_translate_x_{sign}_4",
                            )
                        ]["mean_quantized_prefix_displacement_mse"][-1]
                    )
                    for sign in ("negative", "positive")
                )
                eight = statistics.mean(
                    float(
                        lookup[
                            (
                                family,
                                f"endpoint_translate_x_{sign}_8",
                            )
                        ]["mean_quantized_prefix_displacement_mse"][-1]
                    )
                    for sign in ("negative", "positive")
                )
                dose_checks.append(eight >= four)

        photo_change = statistics.mean(
            row["full_prefix_change_fraction"] for row in photo_groups
        )
        photo_l1_change = statistics.mean(
            row["l1_change_fraction"] for row in photo_groups
        )
        photo_input_relative = statistics.mean(
            row["input_relative_to_natural_next"] for row in photo_groups
        )
        photo_quantized_relative = statistics.mean(
            row["quantized_relative_to_natural_next"]
            for row in photo_groups
        )
        maximum_photo_l1 = max(
            row["l1_change_fraction"] for row in photo_groups
        )
        maximum_photo_input = max(
            row["input_relative_to_natural_next"] for row in photo_groups
        )
        maximum_photo_quantized = max(
            row["quantized_relative_to_natural_next"]
            for row in photo_groups
        )
        photo_status = (
            "pass"
            if (
                maximum_photo_l1 <= 0.35
                and maximum_photo_input <= 1.0
                and maximum_photo_quantized <= 1.0
            )
            else "conditional"
            if (
                maximum_photo_l1 <= 0.60
                and maximum_photo_input <= 2.0
                and maximum_photo_quantized <= 2.0
            )
            else "fail"
        )
        missing = _gate(
            "photometric_robustness",
            photo_status,
            title="Photometric robustness",
            evidence={
                "mean_l1_change_fraction": photo_l1_change,
                "mean_full_prefix_change_fraction": photo_change,
                "mean_input_displacement_relative_to_natural_next": (
                    photo_input_relative
                ),
                "mean_quantized_displacement_relative_to_natural_next": (
                    photo_quantized_relative
                ),
                "maximum_family_split_l1_change_fraction": maximum_photo_l1,
                "maximum_family_split_input_relative_to_natural_next": (
                    maximum_photo_input
                ),
                "maximum_family_split_quantized_relative_to_natural_next": (
                    maximum_photo_quantized
                ),
                "family_split_rows": photo_groups,
            },
            criteria=[
                "pass in every family/split: L1 change <= 0.35",
                "pass in every family/split: input and full-RQ displacement "
                "<= one natural next step",
                "fail if any family/split has L1 change > 0.60 or "
                "displacement > two natural steps",
            ],
            conclusion=(
                "Mild brightness and contrast changes are below the nuisance "
                "budget."
                if photo_status == "pass"
                else "Photometric appearance materially changes the code; "
                "augmentation or invariant routing is required."
                if photo_status == "conditional"
                else "Photometric nuisance overwhelms the intended code use."
            ),
        )

        endpoint_change = statistics.mean(
            row["endpoint_full_prefix_change_fraction"]
            for row in geometry_groups
        )
        endpoint_quantized_relative = statistics.mean(
            row["quantized_relative_to_natural_next"]
            for row in geometry_groups
        )
        direction_distinct = statistics.mean(
            row["opposite_transform_prefix_distinct_fraction"]
            for row in geometry_groups
        )
        minimum_endpoint_change = min(
            row["endpoint_full_prefix_change_fraction"]
            for row in geometry_groups
        )
        minimum_endpoint_quantized = min(
            row["quantized_relative_to_natural_next"]
            for row in geometry_groups
        )
        minimum_direction_distinct = min(
            row["opposite_transform_prefix_distinct_fraction"]
            for row in geometry_groups
        )
        dose_fraction = sum(dose_checks) / max(len(dose_checks), 1)
        geometry_status = (
            "pass"
            if minimum_endpoint_change >= 0.20
            and minimum_direction_distinct >= 0.25
            and minimum_endpoint_quantized >= 0.15
            and dose_fraction >= 0.80
            else "conditional"
            if minimum_endpoint_change >= 0.08
            and minimum_direction_distinct >= 0.10
            and minimum_endpoint_quantized >= 0.05
            and dose_fraction >= 0.50
            else "fail"
        )
        geometry = _gate(
            "geometry_sensitivity",
            geometry_status,
            title="Translation and scale sensitivity",
            evidence={
                "mean_endpoint_full_prefix_change_fraction": endpoint_change,
                "mean_opposite_transform_prefix_distinct_fraction": (
                    direction_distinct
                ),
                "mean_quantized_displacement_relative_to_natural_next": (
                    endpoint_quantized_relative
                ),
                "minimum_family_split_endpoint_change_fraction": (
                    minimum_endpoint_change
                ),
                "minimum_family_split_opposite_distinct_fraction": (
                    minimum_direction_distinct
                ),
                "minimum_family_split_quantized_relative_to_natural_next": (
                    minimum_endpoint_quantized
                ),
                "dose_response_pass_fraction": dose_fraction,
                "family_split_rows": geometry_groups,
            },
            criteria=[
                "pass in every family/split: endpoint change >= 0.20",
                "pass in every family/split: opposite distinction >= 0.25",
                "pass in every family/split: quantized displacement >= 0.15 "
                "natural next step",
                "pass: 8px quantized displacement >= 4px in >= 80% rows",
            ],
            conclusion=(
                "Frozen codes respond monotonically and directionally to "
                "synthetic endpoint translation/scale."
                if geometry_status == "pass"
                else "Geometry is represented, but not strongly enough to "
                "claim an object-motion code without continuous latent support."
                if geometry_status == "conditional"
                else "The current descriptor/codebook does not reliably "
                "separate controlled translation and scale."
            ),
        )

    if not libero:
        cross_domain = _gate(
            "cross_domain_stress",
            "not_run",
            title="LIBERO cross-domain stress",
            evidence={},
            criteria=["frozen DROID codebooks evaluated on LIBERO wrist RGB"],
            conclusion="Cross-domain stress was not run.",
        )
    else:
        libero_groups = []
        for report_index, report in enumerate(libero, start=1):
            families = sorted(
                {str(row["family"]) for row in report["condition_rows"]}
            )
            for family in families:
                usage = [
                    row
                    for row in report["identity_usage_rows"]
                    if row["family"] == family
                ]
                endpoint_rows = [
                    row
                    for row in report["condition_rows"]
                    if row["family"] == family
                    and row["category"] == "endpoint_geometry"
                    and (
                        row["name"].endswith("_8")
                        or row["name"].startswith("endpoint_scale")
                    )
                ]
                direction_rows = [
                    row
                    for row in report["direction_rows"]
                    if row["family"] == family
                    and (
                        "negative_8" in row["left_condition"]
                        or row["left_condition"] == "endpoint_scale_090"
                    )
                ]
                photometric_rows = [
                    row
                    for row in report["condition_rows"]
                    if row["family"] == family
                    and row["category"] == "photometric_nuisance"
                ]
                libero_groups.append(
                    {
                        "report": report_index,
                        "family": family,
                        "minimum_active_code_fraction": min(
                            float(row["active_codes"])
                            / float(row["capacity"])
                            for row in usage
                        ),
                        "minimum_perplexity_fraction": min(
                            float(row["perplexity_fraction"])
                            for row in usage
                        ),
                        "endpoint_full_prefix_change_fraction": (
                            statistics.mean(
                                float(row["prefix_change_fraction"][-1])
                                for row in endpoint_rows
                            )
                        ),
                        "opposite_transform_prefix_distinct_fraction": (
                            statistics.mean(
                                float(
                                    row["prefix_distinct_fraction"][-1]
                                )
                                for row in direction_rows
                            )
                        ),
                        "photometric_quantized_relative_to_natural_next": (
                            statistics.mean(
                                float(
                                    row[
                                        "quantized_prefix_relative_to_natural_next"
                                    ][-1]
                                )
                                for row in photometric_rows
                                if row[
                                    "quantized_prefix_relative_to_natural_next"
                                ][-1]
                                is not None
                            )
                        ),
                    }
                )
        minimum_active = min(
            row["minimum_active_code_fraction"] for row in libero_groups
        )
        minimum_perplexity = min(
            row["minimum_perplexity_fraction"] for row in libero_groups
        )
        endpoint_change = statistics.mean(
            row["endpoint_full_prefix_change_fraction"]
            for row in libero_groups
        )
        direction_distinct = statistics.mean(
            row["opposite_transform_prefix_distinct_fraction"]
            for row in libero_groups
        )
        photometric_quantized_relative = statistics.mean(
            row["photometric_quantized_relative_to_natural_next"]
            for row in libero_groups
        )
        minimum_endpoint_change = min(
            row["endpoint_full_prefix_change_fraction"]
            for row in libero_groups
        )
        minimum_direction_distinct = min(
            row["opposite_transform_prefix_distinct_fraction"]
            for row in libero_groups
        )
        status = (
            "pass"
            if (
                minimum_active >= 0.75
                and minimum_perplexity >= 0.50
                and minimum_endpoint_change >= 0.15
                and minimum_direction_distinct >= 0.20
            )
            else "conditional"
            if (
                minimum_active >= 0.50
                and minimum_endpoint_change >= 0.05
                and minimum_direction_distinct >= 0.05
            )
            else "fail"
        )
        cross_domain = _gate(
            "cross_domain_stress",
            status,
            title="LIBERO cross-domain stress",
            evidence={
                "minimum_active_code_fraction": minimum_active,
                "minimum_perplexity_fraction": minimum_perplexity,
                "mean_endpoint_full_prefix_change_fraction": endpoint_change,
                "mean_opposite_transform_prefix_distinct_fraction": (
                    direction_distinct
                ),
                "minimum_family_endpoint_change_fraction": (
                    minimum_endpoint_change
                ),
                "minimum_family_opposite_distinct_fraction": (
                    minimum_direction_distinct
                ),
                "mean_photometric_quantized_relative_to_natural_next": (
                    photometric_quantized_relative
                ),
                "family_rows": libero_groups,
            },
            criteria=[
                "informational pass: >= 75% centers active and perplexity/K >= 0.50",
                "every family endpoint full-prefix change >= 0.15",
                "every family opposite transform distinction >= 0.20",
                "DROID-trained codebooks are not treated as LIBERO-trained models",
            ],
            conclusion=(
                "The frozen coordinate system remains populated under LIBERO "
                "domain shift."
                if status == "pass"
                else "LIBERO exposes domain collapse risk; this is a stress "
                "signal, not a replacement for LIBERO-trained validation."
            ),
        )
    return missing, geometry, cross_domain


def _event_gate(action_events: dict[str, Any] | None) -> dict[str, Any]:
    if action_events is None:
        return _gate(
            "action_event_semantics",
            "not_run",
            title="Held-out action-event semantics",
            evidence={},
            criteria=["train-fit event map evaluated on val/test scenes"],
            conclusion="Action-event semantics were not run.",
        )
    rows = [
        row
        for row in action_events.get("rows", ())
        if int(row.get("prefix_depth", 0)) == int(row.get("levels", 0))
        and row.get("event")
        in {
            "translation_magnitude_quartile",
            "translation_direction",
            "rotation_magnitude_quartile",
            "rotation_direction",
            "gripper_change",
        }
    ]
    expected_splits = {"val", "test"}
    best_motion_by_split = {
        split: max(
            float(row["normalized_accuracy_gain"])
            for row in rows
            if row["split"] == split
            and row["event"]
            in {
                "translation_magnitude_quartile",
                "translation_direction",
                "rotation_magnitude_quartile",
                "rotation_direction",
            }
        )
        for split in expected_splits
        if any(row["split"] == split for row in rows)
    }
    best_gripper_by_split = {
        split: max(
            float(row["normalized_accuracy_gain"])
            for row in rows
            if row["split"] == split
            and row["event"] == "gripper_change"
        )
        for split in expected_splits
        if any(
            row["split"] == split and row["event"] == "gripper_change"
            for row in rows
        )
    }
    motion_rows = [
        row for row in rows if row["event"] != "gripper_change"
    ]
    median_motion_gain = (
        statistics.median(
            float(row["normalized_accuracy_gain"])
            for row in motion_rows
        )
        if motion_rows
        else None
    )
    minimum_coverage = (
        min(float(row["any_code_coverage"]) for row in rows)
        if rows
        else None
    )
    if (
        set(best_motion_by_split) != expected_splits
        or set(best_gripper_by_split) != expected_splits
        or median_motion_gain is None
        or minimum_coverage is None
    ):
        status = "not_run"
    elif (
        min(best_motion_by_split.values()) >= 0.02
        and median_motion_gain > 0
        and minimum_coverage >= 0.95
    ):
        status = (
            "pass"
            if min(best_gripper_by_split.values()) >= 0.01
            else "conditional"
        )
    elif (
        min(best_motion_by_split.values()) > 0
        and minimum_coverage >= 0.80
    ):
        status = "conditional"
    else:
        status = "fail"
    return _gate(
        "action_event_semantics",
        status,
        title="Held-out action-event semantics",
        evidence={
            "best_motion_event_gain_by_split": best_motion_by_split,
            "median_motion_event_normalized_accuracy_gain": (
                median_motion_gain
            ),
            "best_gripper_event_gain_by_split": best_gripper_by_split,
            "minimum_any_code_coverage": minimum_coverage,
        },
        criteria=[
            "best translation/rotation event gain >= 0.02 on val and test",
            "median motion event gain > 0",
            "pass requires best gripper gain >= 0.01 on val and test",
            "any-code coverage >= 0.95",
        ],
        conclusion=(
            "Code prefixes carry motion/event information across unseen scenes."
            if status == "pass"
            else "Motion association transfers across scenes, but gripper or "
            "other event detail is too weak for code-only control; continuous "
            "latent and proprioceptive context remain mandatory."
            if status == "conditional"
            else "Held-out event evidence is absent or negative."
        ),
    )


def _stability_gate(stability: dict[str, Any] | None) -> dict[str, Any]:
    if stability is None:
        return _gate(
            "seed_stability",
            "not_run",
            title="Independent K-Means seed stability",
            evidence={},
            criteria=["three independent seeds on shared held-out descriptors"],
            conclusion="Seed stability was not run.",
        )
    across = [
        row
        for row in stability.get("distortion_rows", ())
        if row.get("run") == "__across_runs__"
    ]
    pairs = stability.get("pair_rows", ())
    if not across or not pairs:
        status = "not_run"
        max_cv = None
        max_range = None
        min_nmi = None
        min_ari = None
        min_full_prefix_ari = None
        min_prefix = None
        median_prefix = None
        max_prefix = None
    else:
        max_cv = max(float(row["coefficient_of_variation"]) for row in across)
        max_range = max(
            float(row["maximum_relative_range"]) for row in across
        )
        nmis = [
            float(level["normalized_mutual_information"])
            for row in pairs
            for level in row["levels"]
        ]
        aris = [
            float(level["adjusted_rand_index"])
            for row in pairs
            for level in row["levels"]
            if level.get("adjusted_rand_index") is not None
        ]
        full_prefix_rows = [
            next(
                prefix
                for prefix in row["prefix_partitions"]
                if int(prefix["depth"]) == len(row["levels"])
            )
            for row in pairs
        ]
        full_prefix_nmis = [
            float(prefix["normalized_mutual_information"])
            for prefix in full_prefix_rows
        ]
        full_prefix_aris = [
            float(prefix["adjusted_rand_index"])
            for prefix in full_prefix_rows
            if prefix.get("adjusted_rand_index") is not None
        ]
        prefix = [
            float(row["mapped_prefix_agreement"][-1]) for row in pairs
        ]
        min_nmi = min(nmis)
        min_ari = min(aris) if aris else None
        min_full_prefix_nmi = min(full_prefix_nmis)
        min_full_prefix_ari = (
            min(full_prefix_aris) if full_prefix_aris else None
        )
        min_prefix = min(prefix)
        median_prefix = statistics.median(prefix)
        max_prefix = max(prefix)
        status = (
            "pass"
            if max_cv <= 0.03
            and min_nmi >= 0.50
            and min_full_prefix_nmi >= 0.50
            and median_prefix >= 0.40
            else "conditional"
            if (
                max_cv <= 0.05
                and min_nmi >= 0.30
                and min_full_prefix_nmi >= 0.30
            )
            else "fail"
        )
    if not across or not pairs:
        min_full_prefix_nmi = None
    return _gate(
        "seed_stability",
        status,
        title="Independent K-Means seed stability",
        evidence={
            "maximum_residual_mse_coefficient_of_variation": max_cv,
            "maximum_residual_mse_relative_range": max_range,
            "minimum_level_nmi": min_nmi,
            "minimum_level_ari": min_ari,
            "minimum_full_prefix_partition_nmi": min_full_prefix_nmi,
            "minimum_full_prefix_partition_ari": min_full_prefix_ari,
            "minimum_mapped_full_prefix_agreement": min_prefix,
            "median_mapped_full_prefix_agreement": median_prefix,
            "maximum_mapped_full_prefix_agreement": max_prefix,
        },
        criteria=[
            "pass: residual MSE CV <= 0.03",
            "pass: every level NMI >= 0.50",
            "pass: every full-prefix partition NMI >= 0.50",
            "pass: median mapped full-prefix agreement >= 0.40",
        ],
        conclusion=(
            "Independent initializations recover comparable partitions and "
            "distortion."
            if status == "pass"
            else "Distortion is usable but code identities/partitions require "
            "artifact freezing and must not be retrained casually."
            if status == "conditional"
            else "The RQ partition is too seed-sensitive for stable use."
        ),
    )


def _capacity_summary(capacity: dict[str, Any] | None) -> list[dict[str, Any]]:
    if capacity is None:
        return []
    rows = []
    for row in capacity.get("rows", ()):
        rows.append(
            {
                "label": row["label"],
                "split": row["split"],
                "k": int(row["k"]),
                "heldout_residual_total_reduction": float(
                    row["heldout_residual_total_reduction"]
                ),
                "joint_perplexity_fraction": float(
                    row["joint_perplexity_fraction"]
                ),
                "minimum_level_perplexity_fraction": min(
                    float(value)
                    for value in row["perplexity_fraction_by_level"]
                ),
            }
        )
    return rows


def _overall_decision(gates: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {gate["name"]: gate for gate in gates}
    blockers = [
        name
        for name in (
            "causal_reproduction",
            "quantizer_health",
            "rq_hierarchy",
            "family_complementarity",
            "seed_stability",
        )
        if by_name[name]["status"] in {"fail", "not_run"}
    ]
    semantic_limiters = [
        name
        for name in (
            "context_leakage",
            "photometric_robustness",
            "geometry_sensitivity",
            "action_event_semantics",
            "cross_domain_stress",
        )
        if by_name[name]["status"] != "pass"
    ]
    if blockers:
        verdict = "not_ready"
    elif semantic_limiters:
        verdict = "conditional_pass"
    else:
        verdict = "pass"
    required_followups = []
    if by_name["context_leakage"]["status"] != "pass":
        required_followups.append(
            "retain scene-balanced evaluation and prevent code-only scene "
            "shortcuts"
        )
    if by_name["photometric_robustness"]["status"] != "pass":
        required_followups.append(
            "test paired photometric augmentation and quantization-margin "
            "confidence before exposing unstable suffixes"
        )
    if by_name["geometry_sensitivity"]["status"] != "pass":
        required_followups.append(
            "run object-pose interventions in a simulator; synthetic whole-"
            "frame geometry is not object causality"
        )
    if by_name["action_event_semantics"]["status"] != "pass":
        required_followups.append(
            "retain continuous latent, proprioception and explicit gripper "
            "state for precision control"
        )
    seed_status = by_name["seed_stability"]["status"]
    if seed_status == "not_run":
        required_followups.append(
            "complete the independent three-seed partition gate before code "
            "integration"
        )
    elif seed_status == "conditional":
        required_followups.append(
            "freeze artifact hashes and carry every independent seed into "
            "continuous-latent-plus-code Gate 2"
        )
    elif seed_status == "fail":
        required_followups.append(
            "revisit the descriptor or initialization, or demonstrate "
            "downstream robustness across independently trained artifacts "
            "before selecting a canonical codebook"
        )
    if by_name["cross_domain_stress"]["status"] != "pass":
        required_followups.append(
            "independently refit or calibrate the same specification on "
            "LIBERO before any cross-domain claim"
        )
    return {
        "verdict": verdict,
        "blocking_gates": blockers,
        "semantic_limiters": semantic_limiters,
        "deployment_scope": (
            "blocked pending required gates"
            if verdict == "not_ready"
            else "DROID-in-domain research only"
        ),
        "approved_use": (
            [
                "DROID-in-domain frozen read-only multiscale visual measurement",
                "DROID role-specific Policy/World routing experiments",
                "DROID continuous-latent-plus-code Gate 2 probes",
            ]
            if verdict != "not_ready"
            else []
        ),
        "not_approved": [
            "code-only precision control",
            "online codebook center updates",
            "object-level motion primitive claims",
            "removing the continuous Wan latent path",
            "using frozen DROID artifacts as a universal cross-domain tokenizer",
        ],
        "required_followups": required_followups,
    }


def _format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.2f}%"


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RQ-KMeans visual usability report",
        "",
        f"Overall verdict: **{report['decision']['verdict']}**",
        "",
        f"Scope: **{report['decision']['deployment_scope']}**",
        "",
        "No scalar score is used. A strong average cannot hide a failed causal, "
        "stability, nuisance, or semantic gate.",
        "",
        "| gate | status | conclusion |",
        "|---|---|---|",
    ]
    for gate in report["gates"]:
        lines.append(
            f"| {gate['title']} | **{gate['status']}** | "
            f"{gate['conclusion']} |"
        )
    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- pooled manifest: "
            f"`{report['provenance']['pooled_manifest_fingerprint']}`",
        ]
    )
    lines.extend(
        f"- {family} artifact: `{sha256}`"
        for family, sha256 in sorted(
            report["provenance"]["canonical_artifact_sha256"].items()
        )
    )
    lines.append(
        "- validated sources: "
        + ", ".join(report["provenance"]["validated_sources"])
    )
    lines.extend(
        [
            "",
            "## Approved use",
            "",
        ]
    )
    lines.extend(
        f"- {value}" for value in report["decision"]["approved_use"]
    )
    lines.extend(["", "## Not approved", ""])
    lines.extend(
        f"- {value}" for value in report["decision"]["not_approved"]
    )
    lines.extend(["", "## Required follow-ups", ""])
    lines.extend(
        f"- {value}" for value in report["decision"]["required_followups"]
    )
    lines.extend(["", "## Interpretation guardrails", ""])
    lines.extend(
        f"- {value}" for value in report["interpretation_guardrails"]
    )
    lines.extend(["", "## Gate evidence", ""])
    for gate in report["gates"]:
        lines.extend(
            [
                f"### {gate['title']}",
                "",
                f"Status: **{gate['status']}**",
                "",
                "```json",
                json.dumps(
                    gate["evidence"],
                    indent=2,
                    sort_keys=True,
                ),
                "```",
                "",
            ]
        )
    if report["capacity_tradeoff"]:
        lines.extend(
            [
                "## Capacity tradeoff",
                "",
                "| candidate | split | K | held-out reduction | joint "
                "perplexity/capacity |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in report["capacity_tradeoff"]:
            lines.append(
                f"| {row['label']} | {row['split']} | {row['k']} | "
                f"{_format_percent(row['heldout_residual_total_reduction'])} | "
                f"{_format_percent(row['joint_perplexity_fraction'])} |"
            )
        lines.extend(
            [
                "",
                "K=16/32 reduce distortion further, but semantic, perturbation "
                "and stability gates in this report apply to K=8. Capacity must "
                "not be selected from distortion alone.",
                "",
            ]
        )
    return "\n".join(lines)


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_rq_usability_report(
    *,
    comparison_report: str | Path,
    family_association_report: str | Path,
    retrieval_report: str | Path,
    temporal_reports: Iterable[str | Path],
    causal_audit: str | Path,
    output_dir: str | Path,
    visual_reports: Iterable[str | Path] = (),
    action_event_report: str | Path | None = None,
    seed_stability_report: str | Path | None = None,
    capacity_comparison_report: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    temporal_paths = tuple(Path(path) for path in temporal_reports)
    visual_paths = tuple(Path(path) for path in visual_reports)
    if not temporal_paths:
        raise ValueError("Usability report requires temporal reports.")
    comparison = _load_json(comparison_report)
    family_association = _load_json(family_association_report)
    retrieval = _load_json(retrieval_report)
    temporal = [_load_json(path) for path in temporal_paths]
    causal = _load_json(causal_audit)
    visual = [_load_json(path) for path in visual_paths]
    action_events = (
        None
        if action_event_report is None
        else _load_json(action_event_report)
    )
    stability = (
        None
        if seed_stability_report is None
        else _load_json(seed_stability_report)
    )
    capacity = (
        None
        if capacity_comparison_report is None
        else _load_json(capacity_comparison_report)
    )
    provenance = _provenance_audit(
        comparison_report=comparison_report,
        comparison=comparison,
        family_association_report=family_association_report,
        family_association=family_association,
        retrieval_report=retrieval_report,
        retrieval=retrieval,
        temporal_paths=temporal_paths,
        temporal=temporal,
        visual_paths=visual_paths,
        visual=visual,
        action_event_report=action_event_report,
        action_events=action_events,
        seed_stability_report=seed_stability_report,
        stability=stability,
    )

    input_paths: dict[str, Any] = {
        "comparison": _input_row(comparison_report),
        "family_association": _input_row(family_association_report),
        "retrieval": _input_row(retrieval_report),
        "temporal": [_input_row(path) for path in temporal_paths],
        "causal_audit": _input_row(causal_audit),
        "visual": [_input_row(path) for path in visual_paths],
        "action_events": (
            None
            if action_event_report is None
            else _input_row(action_event_report)
        ),
        "seed_stability": (
            None
            if seed_stability_report is None
            else _input_row(seed_stability_report)
        ),
        "capacity_comparison": (
            None
            if capacity_comparison_report is None
            else _input_row(capacity_comparison_report)
        ),
    }
    contract_payload = {
        "schema": USABILITY_CONTRACT_SCHEMA,
        "inputs": input_paths,
        "gate_order": list(GATE_ORDER),
        "provenance": provenance,
        "implementation_sha256": file_sha256(Path(__file__)),
    }
    contract_hash = _canonical_hash(contract_payload)
    contract = {**contract_payload, "contract_hash": contract_hash}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = output_dir / "contract.json"
    report_path = output_dir / "usability_report.json"
    markdown_path = output_dir / "USABILITY_REPORT.md"
    if contract_path.is_file():
        previous = json.loads(contract_path.read_text(encoding="utf-8"))
        if previous != contract:
            raise RuntimeError("Existing usability contract differs.")
        if not resume:
            raise FileExistsError(
                f"Usability contract exists at `{contract_path}`."
            )
    else:
        write_json_report(contract_path, contract)
    if resume and report_path.is_file() and markdown_path.is_file():
        report = _load_json(report_path)
        if report.get("contract_hash") != contract_hash:
            raise RuntimeError("Usability report contract hash is invalid.")
        return report
    if report_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"Incomplete usability output exists under `{output_dir}`."
        )

    photometric, geometry, cross_domain = _visual_gates(visual)
    gates = [
        _causal_gate(causal, visual),
        _quantizer_health(comparison),
        _hierarchy_gate(temporal),
        _complementarity_gate(family_association),
        _context_gate(comparison, retrieval),
        photometric,
        geometry,
        _event_gate(action_events),
        _stability_gate(stability),
        cross_domain,
    ]
    gates.sort(key=lambda gate: GATE_ORDER.index(gate["name"]))
    report = {
        "schema": USABILITY_REPORT_SCHEMA,
        "contract_hash": contract_hash,
        "provenance": provenance,
        "decision": _overall_decision(gates),
        "gates": gates,
        "capacity_tradeoff": _capacity_summary(capacity),
        "interpretation_guardrails": [
            "The report evaluates one canonical DROID-10k wrist "
            "g=4,K=8,L=3 specification.",
            "Synthetic RGB perturbations measure representation sensitivity, "
            "not physical causal effects.",
            "Association is evidence of information, not proof that code-only "
            "control is sufficient.",
            "LIBERO is an out-of-domain stress test until a controlled "
            "LIBERO-trained or calibrated codebook is evaluated.",
        ],
    }
    write_json_report(report_path, report)
    _write_text(markdown_path, _markdown(report) + "\n")
    return report
