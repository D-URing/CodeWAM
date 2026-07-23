#!/usr/bin/env python3
"""Build and verify a pinned manifest for the official LIBERO demonstrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
EXPECTED_SUITES = {
    "libero_spatial": 10,
    "libero_object": 10,
    "libero_goal": 10,
    "libero_90": 90,
    "libero_10": 10,
}
HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.chmod(0o664)
    temporary.replace(path)


def _isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def build_manifest(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is required; install the `hf` CLI first.", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api = HfApi(endpoint=args.endpoint)
    info = api.dataset_info(
        repo_id=REPO_ID,
        revision=REVISION,
        files_metadata=True,
    )
    resolved_revision = str(info.sha)
    if resolved_revision != REVISION:
        raise RuntimeError(
            f"revision mismatch: requested {REVISION}, resolved {resolved_revision}"
        )

    files: list[dict[str, Any]] = []
    for item in api.list_repo_tree(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        recursive=True,
        expand=True,
    ):
        size = getattr(item, "size", None)
        if size is None:
            continue
        lfs = getattr(item, "lfs", None)
        files.append(
            {
                "path": str(item.path),
                "size": int(size),
                "sha256": getattr(lfs, "sha256", None) if lfs else None,
                "blob_id": getattr(item, "blob_id", None),
            }
        )

    files.sort(key=lambda item: item["path"])
    hdf5_files = [item for item in files if item["path"].endswith(".hdf5")]
    if len(files) != 132 or len(hdf5_files) != sum(EXPECTED_SUITES.values()):
        raise RuntimeError(
            f"unexpected official tree: {len(files)} files, "
            f"{len(hdf5_files)} HDF5 files"
        )
    if any(not item["sha256"] for item in hdf5_files):
        raise RuntimeError("one or more HDF5 files are missing official LFS SHA256 values")

    expected_files = {
        "repo": REPO_ID,
        "revision": REVISION,
        "files": files,
    }
    source = {
        "dataset": "LIBERO official demonstrations",
        "dataset_license": "CC BY 4.0",
        "repository": REPO_ID,
        "repo_type": "dataset",
        "revision": REVISION,
        "manifest_created_at": datetime.now(timezone.utc).isoformat(),
        "upstream_last_modified": _isoformat(getattr(info, "last_modified", None)),
        "reported_repository_bytes": getattr(info, "used_storage", None),
        "expected_download_bytes": sum(item["size"] for item in files),
        "expected_files": len(files),
        "expected_hdf5": len(hdf5_files),
        "suites": EXPECTED_SUITES,
        "official_code": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "official_dataset_page": "https://libero-project.github.io/datasets",
        "download_repository": (
            "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets"
        ),
    }

    _write_json(output_dir / "source.json", source)
    _write_json(output_dir / "expected_files.json", expected_files)
    checksum_path = output_dir / "expected_hdf5.sha256"
    checksum_path.write_text(
        "".join(f"{item['sha256']}  {item['path']}\n" for item in hdf5_files),
        encoding="utf-8",
    )
    checksum_path.chmod(0o664)
    print(
        f"manifest: {len(files)} files, {len(hdf5_files)} HDF5, "
        f"{source['expected_download_bytes']} bytes"
    )
    return 0


def _read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        digest, separator, relative_path = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"invalid checksum line {line_number}: {raw_line}")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe checksum path on line {line_number}: {relative}")
        checksums[relative.as_posix()] = digest.lower()
    return checksums


def _sha256(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return path.as_posix(), digest.hexdigest()


def verify_dataset(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    errors: list[str] = []
    suite_counts: dict[str, int] = {}
    hdf5_files: list[Path] = []

    for suite, expected_count in EXPECTED_SUITES.items():
        suite_files = sorted((root / suite).glob("*.hdf5"))
        suite_counts[suite] = len(suite_files)
        hdf5_files.extend(suite_files)
        if len(suite_files) != expected_count:
            errors.append(
                f"{suite}: expected {expected_count} HDF5 files, found {len(suite_files)}"
            )

    all_hdf5 = sorted(root.rglob("*.hdf5"))
    if set(all_hdf5) != set(hdf5_files):
        errors.append("unexpected HDF5 files exist outside the five official directories")

    bad_magic: list[str] = []
    for path in all_hdf5:
        with path.open("rb") as handle:
            if handle.read(len(HDF5_MAGIC)) != HDF5_MAGIC:
                bad_magic.append(path.relative_to(root).as_posix())
    if bad_magic:
        errors.append(f"{len(bad_magic)} files have an invalid HDF5 signature")

    for metadata_name in (".gitattributes", "README.md"):
        if not (root / metadata_name).is_file():
            errors.append(f"missing repository metadata: {metadata_name}")

    checked_hashes = 0
    checksum_manifest_digest: str | None = None
    mismatched_hashes: list[str] = []
    missing_manifest_files: list[str] = []
    unexpected_manifest_files: list[str] = []
    if args.sha256_manifest is not None:
        checksum_manifest = args.sha256_manifest.resolve()
        expected = _read_checksums(checksum_manifest)
        checksum_manifest_digest = _sha256(checksum_manifest)[1]
        actual_relative = {
            path.relative_to(root).as_posix(): path for path in all_hdf5
        }
        missing_manifest_files = sorted(set(expected) - set(actual_relative))
        unexpected_manifest_files = sorted(set(actual_relative) - set(expected))
        if missing_manifest_files:
            errors.append(
                f"{len(missing_manifest_files)} checksum-listed files are missing"
            )
        if unexpected_manifest_files:
            errors.append(
                f"{len(unexpected_manifest_files)} HDF5 files are absent from the manifest"
            )

        hash_targets = [
            actual_relative[relative]
            for relative in sorted(set(expected) & set(actual_relative))
        ]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for absolute_name, actual_digest in executor.map(_sha256, hash_targets):
                path = Path(absolute_name)
                relative = path.relative_to(root).as_posix()
                checked_hashes += 1
                if actual_digest != expected[relative]:
                    mismatched_hashes.append(relative)
        if mismatched_hashes:
            errors.append(f"{len(mismatched_hashes)} SHA256 values do not match")

    total_bytes = sum(path.stat().st_size for path in all_hdf5)
    report = {
        "dataset": "LIBERO official demonstrations",
        "repository": REPO_ID,
        "revision": REVISION,
        "root": root.as_posix(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not errors else "failed",
        "suite_counts": suite_counts,
        "total_hdf5": len(all_hdf5),
        "total_hdf5_bytes": total_bytes,
        "hdf5_signature_checked": len(all_hdf5),
        "sha256_checked": checked_hashes,
        "sha256_manifest_digest": checksum_manifest_digest,
        "bad_hdf5_signature": bad_magic,
        "missing_manifest_files": missing_manifest_files,
        "unexpected_manifest_files": unexpected_manifest_files,
        "mismatched_sha256": mismatched_hashes,
        "errors": errors,
    }
    if args.report is not None:
        _write_json(args.report.resolve(), report)

    print(json.dumps(report, indent=2, ensure_ascii=True))
    return 0 if not errors else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest", help="fetch the pinned official repository manifest"
    )
    manifest.add_argument("--output-dir", type=Path, required=True)
    manifest.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )
    manifest.set_defaults(func=build_manifest)

    verify = subparsers.add_parser(
        "verify", help="verify suite counts, HDF5 signatures, and optional SHA256"
    )
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--sha256-manifest", type=Path)
    verify.add_argument("--report", type=Path)
    verify.add_argument("--workers", type=int, default=4)
    verify.set_defaults(func=verify_dataset)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers must be at least 1")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
