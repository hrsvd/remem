from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from typing import Any

from benchmarks.io import read_json, write_json

MANIFEST_DIR = Path(__file__).parent / "manifests"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(
    name: str, data_dir: str | Path, *, force: bool = False
) -> dict[str, Any]:
    manifest_path = MANIFEST_DIR / f"{name}.json"
    if not manifest_path.exists():
        choices = ", ".join(path.stem for path in sorted(MANIFEST_DIR.glob("*.json")))
        raise ValueError(f"Unknown dataset {name!r}; choose one of: {choices}")
    manifest = read_json(manifest_path)
    destination = Path(data_dir) / "raw" / name
    destination.mkdir(parents=True, exist_ok=True)
    resolved_files = []
    for item in manifest["files"]:
        output = destination / item["name"]
        if force or not output.exists():
            temporary = output.with_suffix(output.suffix + ".tmp")
            request = urllib.request.Request(
                item["url"], headers={"User-Agent": "remem-benchmark/1"}
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                with temporary.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            temporary.replace(output)
        checksum = sha256_file(output)
        expected = item.get("sha256")
        if expected and checksum != expected:
            raise ValueError(f"Checksum mismatch for {output}")
        resolved_files.append(
            {
                "name": item["name"],
                "url": item["url"],
                "sha256": checksum,
                "bytes": output.stat().st_size,
            }
        )
        if output.name.endswith(".tar.gz"):
            extracted = destination / "extracted"
            extracted.mkdir(exist_ok=True)
            with tarfile.open(output, "r:gz") as archive:
                archive.extractall(extracted, filter="data")
    receipt = {
        **manifest,
        "files": resolved_files,
        "local_directory": str(destination.resolve()),
    }
    write_json(destination / "download-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="Download licensed benchmark data")
    parser.add_argument(
        "dataset", choices=[path.stem for path in MANIFEST_DIR.glob("*.json")]
    )
    parser.add_argument("--data-dir", default="benchmarks/data")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    receipt = download_dataset(args.dataset, args.data_dir, force=args.force)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
