#!/usr/bin/env python3
"""Package or unpack Phase 9 training artifacts for Phase 10 analysis."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ARTIFACT_PATHS = (
    Path("outputs/checkpoints"),
    Path("outputs/logs"),
    Path("outputs/evaluation"),
    Path("outputs/onnx"),
    Path("data/splits"),
)


def package_artifacts(project_root: Path, output_zip: Path) -> None:
    """Create a zip file containing Phase 9 artifacts."""
    output_zip = output_zip.resolve()
    found_files = 0
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for relative_dir in ARTIFACT_PATHS:
            artifact_dir = project_root / relative_dir
            if not artifact_dir.exists():
                continue
            for path in artifact_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(project_root))
                    found_files += 1
    if found_files == 0:
        output_zip.unlink(missing_ok=True)
        raise SystemExit(
            "No Phase 9 artifact files found. Expected files under: "
            + ", ".join(str(path) for path in ARTIFACT_PATHS)
        )
    print(f"Packaged {found_files} files into {output_zip}")


def unpack_artifacts(project_root: Path, input_zip: Path) -> None:
    """Extract Phase 9 artifacts into the project root."""
    if not input_zip.exists():
        raise SystemExit(f"Artifact zip not found: {input_zip}")
    with zipfile.ZipFile(input_zip) as zf:
        members = zf.namelist()
        unsafe = [
            member
            for member in members
            if Path(member).is_absolute() or ".." in Path(member).parts
        ]
        if unsafe:
            raise SystemExit(f"Unsafe paths in artifact zip: {unsafe[:5]}")
        zf.extractall(project_root)
    print(f"Unpacked {len(members)} files into {project_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pack/unpack ignored Phase 9 artifacts for Mac Phase 10 tuning."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Run on Windows 5090D.")
    pack_parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase9_artifacts.zip"),
        help="Zip file to create.",
    )

    unpack_parser = subparsers.add_parser("unpack", help="Run on Mac.")
    unpack_parser.add_argument(
        "zipfile",
        type=Path,
        help="Zip file produced by the pack command.",
    )

    args = parser.parse_args()
    project_root = Path.cwd()
    if args.command == "pack":
        package_artifacts(project_root, args.output)
    elif args.command == "unpack":
        unpack_artifacts(project_root, args.zipfile)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
