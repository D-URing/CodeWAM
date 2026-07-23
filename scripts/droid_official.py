#!/usr/bin/env python3
"""Build and verify pinned manifests for the official DROID RLDS releases."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import tempfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    version: str
    bucket: str
    prefix: str
    local_relative: str
    manifest_relative: str
    expected_objects: int
    expected_shards: int
    expected_bytes: int
    shard_prefix: str

    @property
    def source_uri(self) -> str:
        return f"gs://{self.bucket}/{self.prefix}"


SPECS = {
    "full": DatasetSpec(
        key="full",
        version="1.0.1",
        bucket="gresearch",
        prefix="robotics/droid/1.0.1/",
        local_relative="droid/1.0.1",
        manifest_relative="droid/rlds-1.0.1",
        expected_objects=2050,
        expected_shards=2048,
        expected_bytes=1_865_994_705_042,
        shard_prefix="droid_101-train.tfrecord-",
    ),
    "debug": DatasetSpec(
        key="debug",
        version="1.0.0",
        bucket="gresearch",
        prefix="robotics/droid_100/1.0.0/",
        local_relative="droid_100/1.0.0",
        manifest_relative="droid/droid-100-rlds-1.0.0",
        expected_objects=33,
        expected_shards=31,
        expected_bytes=2_192_615_094,
        shard_prefix="r2d2_faceblur-train.tfrecord-",
    ),
}


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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    temporary.chmod(0o664)
    temporary.replace(path)


def _selected_specs(dataset: str) -> list[DatasetSpec]:
    if dataset == "all":
        return [SPECS["debug"], SPECS["full"]]
    return [SPECS[dataset]]


def _list_gcs_objects(spec: DatasetSpec) -> list[dict[str, Any]]:
    fields = "items(name,size,md5Hash,crc32c,generation,updated,etag),nextPageToken"
    page_token: str | None = None
    objects: list[dict[str, Any]] = []
    while True:
        query = {
            "prefix": spec.prefix,
            "maxResults": "1000",
            "fields": fields,
        }
        if page_token is not None:
            query["pageToken"] = page_token
        url = (
            f"https://storage.googleapis.com/storage/v1/b/{spec.bucket}/o?"
            f"{urllib.parse.urlencode(query)}"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = json.load(response)
        for item in payload.get("items", []):
            name = str(item["name"])
            if not name.startswith(spec.prefix):
                raise RuntimeError(f"object escaped prefix: {name}")
            relative = name[len(spec.prefix) :]
            if not relative or relative.startswith("/") or ".." in Path(relative).parts:
                raise RuntimeError(f"unsafe object path: {name}")
            objects.append(
                {
                    "path": relative,
                    "size": int(item["size"]),
                    "md5_base64": item.get("md5Hash"),
                    "crc32c_base64": item.get("crc32c"),
                    "generation": str(item["generation"]),
                    "updated": item.get("updated"),
                    "etag": item.get("etag"),
                }
            )
        page_token = payload.get("nextPageToken")
        if page_token is None:
            break

    objects.sort(key=lambda item: item["path"])
    return objects


def _validate_remote_inventory(
    spec: DatasetSpec, objects: list[dict[str, Any]]
) -> None:
    total_bytes = sum(item["size"] for item in objects)
    shard_count = sum(
        Path(item["path"]).name.startswith(spec.shard_prefix) for item in objects
    )
    if len(objects) != spec.expected_objects:
        raise RuntimeError(
            f"{spec.key}: expected {spec.expected_objects} objects, got {len(objects)}"
        )
    if shard_count != spec.expected_shards:
        raise RuntimeError(
            f"{spec.key}: expected {spec.expected_shards} shards, got {shard_count}"
        )
    if total_bytes != spec.expected_bytes:
        raise RuntimeError(
            f"{spec.key}: expected {spec.expected_bytes} bytes, got {total_bytes}"
        )
    if any(not item["md5_base64"] for item in objects):
        raise RuntimeError(f"{spec.key}: one or more objects have no GCS MD5")


def build_manifest(args: argparse.Namespace) -> int:
    manifest_root = args.manifest_root.resolve()
    for spec in _selected_specs(args.dataset):
        objects = _list_gcs_objects(spec)
        _validate_remote_inventory(spec, objects)
        output_dir = manifest_root / spec.manifest_relative
        source = {
            "dataset": (
                "DROID full RLDS" if spec.key == "full" else "DROID-100 RLDS"
            ),
            "dataset_license": "CC BY 4.0",
            "version": spec.version,
            "source_uri": spec.source_uri,
            "manifest_created_at": datetime.now(timezone.utc).isoformat(),
            "expected_objects": spec.expected_objects,
            "expected_shards": spec.expected_shards,
            "expected_bytes": spec.expected_bytes,
            "local_relative": spec.local_relative,
            "official_docs": (
                "https://droid-dataset.github.io/droid/the-droid-dataset"
            ),
            "official_project": "https://droid-dataset.github.io/",
        }
        checksum_lines = []
        for item in objects:
            digest = base64.b64decode(item["md5_base64"]).hex()
            checksum_lines.append(f"{digest}  {item['path']}\n")

        _write_json(output_dir / "source.json", source)
        _write_json(
            output_dir / "objects.json",
            {
                "bucket": spec.bucket,
                "prefix": spec.prefix,
                "objects": objects,
            },
        )
        _write_text(output_dir / "expected_files.md5", "".join(checksum_lines))
        print(
            f"{spec.key}: {len(objects)} objects, {spec.expected_shards} shards, "
            f"{spec.expected_bytes} bytes"
        )
    return 0


def _md5(path: Path) -> tuple[str, str]:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return path.as_posix(), digest.hexdigest()


def _verify_one(
    spec: DatasetSpec,
    data_root: Path,
    manifest_root: Path,
    workers: int,
    check_hashes: bool,
) -> dict[str, Any]:
    local_root = data_root / spec.local_relative
    manifest_dir = manifest_root / spec.manifest_relative
    object_manifest = json.loads(
        (manifest_dir / "objects.json").read_text(encoding="utf-8")
    )
    objects = object_manifest["objects"]
    _validate_remote_inventory(spec, objects)

    expected = {item["path"]: item for item in objects}
    actual_paths = sorted(path for path in local_root.rglob("*") if path.is_file())
    actual = {path.relative_to(local_root).as_posix(): path for path in actual_paths}
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    size_mismatches = sorted(
        relative
        for relative in set(expected) & set(actual)
        if actual[relative].stat().st_size != expected[relative]["size"]
    )

    md5_mismatches: list[str] = []
    md5_checked = 0
    if check_hashes and not missing and not size_mismatches:
        targets = [actual[relative] for relative in sorted(expected)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for absolute_name, digest in executor.map(_md5, targets):
                path = Path(absolute_name)
                relative = path.relative_to(local_root).as_posix()
                expected_digest = base64.b64decode(
                    expected[relative]["md5_base64"]
                ).hex()
                md5_checked += 1
                if digest != expected_digest:
                    md5_mismatches.append(relative)
                if md5_checked % 128 == 0:
                    print(f"{spec.key}: checked MD5 {md5_checked}/{len(targets)}")

    metadata_errors: list[str] = []
    dataset_info_path = local_root / "dataset_info.json"
    try:
        dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
        split_bytes = int(dataset_info["splits"][0]["numBytes"])
        shard_bytes = sum(
            item["size"]
            for item in objects
            if Path(item["path"]).name.startswith(spec.shard_prefix)
        )
        if split_bytes != shard_bytes:
            metadata_errors.append(
                f"dataset_info split bytes {split_bytes} != shard bytes {shard_bytes}"
            )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as error:
        metadata_errors.append(f"invalid dataset_info.json: {error}")

    errors = []
    if missing:
        errors.append(f"{len(missing)} expected files are missing")
    if unexpected:
        errors.append(f"{len(unexpected)} unexpected files exist")
    if size_mismatches:
        errors.append(f"{len(size_mismatches)} files have the wrong size")
    if md5_mismatches:
        errors.append(f"{len(md5_mismatches)} files have the wrong MD5")
    errors.extend(metadata_errors)
    if check_hashes and md5_checked != spec.expected_objects:
        errors.append(
            f"MD5 coverage is {md5_checked}/{spec.expected_objects}, not complete"
        )

    return {
        "dataset": "DROID full RLDS" if spec.key == "full" else "DROID-100 RLDS",
        "version": spec.version,
        "source_uri": spec.source_uri,
        "root": local_root.as_posix(),
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "status": "ok" if not errors else "failed",
        "expected_objects": spec.expected_objects,
        "local_objects": len(actual),
        "expected_shards": spec.expected_shards,
        "expected_bytes": spec.expected_bytes,
        "local_bytes": sum(path.stat().st_size for path in actual_paths),
        "md5_checked": md5_checked,
        "missing": missing,
        "unexpected": unexpected,
        "size_mismatches": size_mismatches,
        "md5_mismatches": md5_mismatches,
        "metadata_errors": metadata_errors,
        "errors": errors,
    }


def verify_dataset(args: argparse.Namespace) -> int:
    data_root = args.data_root.resolve()
    manifest_root = args.manifest_root.resolve()
    failed = False
    for spec in _selected_specs(args.dataset):
        report = _verify_one(
            spec=spec,
            data_root=data_root,
            manifest_root=manifest_root,
            workers=args.workers,
            check_hashes=not args.skip_hashes,
        )
        output_dir = manifest_root / spec.manifest_relative
        _write_json(output_dir / "verification.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=True))
        failed = failed or report["status"] != "ok"
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser(
        "manifest", help="snapshot GCS generations and checksums"
    )
    manifest.add_argument("--manifest-root", type=Path, required=True)
    manifest.add_argument(
        "--dataset", choices=("full", "debug", "all"), default="all"
    )
    manifest.set_defaults(func=build_manifest)

    verify = subparsers.add_parser(
        "verify", help="verify files, sizes, metadata, and GCS MD5 values"
    )
    verify.add_argument("--data-root", type=Path, required=True)
    verify.add_argument("--manifest-root", type=Path, required=True)
    verify.add_argument(
        "--dataset", choices=("full", "debug", "all"), default="all"
    )
    verify.add_argument("--workers", type=int, default=4)
    verify.add_argument("--skip-hashes", action="store_true")
    verify.set_defaults(func=verify_dataset)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("--workers must be at least 1")
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
