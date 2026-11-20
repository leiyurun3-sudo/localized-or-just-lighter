#!/usr/bin/env python3
"""Collect compact JSON/CSV profiles and named previews; native PNGs opt in."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

from run import ROOT, now, resolve_path, stamp

EXCLUDED = {"_cache", "_attempts", "__pycache__", "models", "model", ".cache", ".git"}
TEXT = {".json", ".jsonl", ".csv", ".tsv", ".txt", ".log"}
IMAGES = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--out", help="Compact archive path (default: deliveries/<timestamp>_compact.zip)")
    parser.add_argument("--native", action="store_true", help="Also write a SEPARATE archive of native.png images")
    parser.add_argument("--native-out", help="Separate native archive path")
    args = parser.parse_args()
    if args.native_out and not args.native:
        parser.error("--native-out requires --native")
    output = resolve_path(args.output_dir)
    if not output.is_dir():
        parser.error(f"Output directory not found: {output}")
    prefix = f"DoseLens_{stamp()}"
    archive = resolve_path(args.out) if args.out else ROOT / "deliveries" / f"{prefix}_compact.zip"
    native_archive = resolve_path(args.native_out) if args.native_out else archive.with_name(archive.stem + "_native.zip")
    if archive == native_archive:
        parser.error("Compact and native archives must have different paths")
    for path in [archive] + ([native_archive] if args.native else []):
        if path.exists():
            parser.error(f"Refusing to overwrite existing archive: {path}")
    compact, native, skipped = [], [], []
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if not path.is_file() or any(part in EXCLUDED for part in relative.parts):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(output.resolve()):
            skipped.append({"path": str(relative), "reason": "symlink/outside output"})
            continue
        lower = path.name.lower()
        size = path.stat().st_size
        if lower == "native.png":
            native.append((path, f"outputs/{relative.as_posix()}"))
        elif path.suffix.lower() in TEXT and size <= 16 * 1024**2:
            compact.append((path, f"outputs/{relative.as_posix()}"))
        elif path.suffix.lower() in IMAGES and any(word in lower for word in ("preview", "thumb", "comparison", "contact")) and size <= 4 * 1024**2:
            compact.append((path, f"outputs/{relative.as_posix()}"))
        else:
            skipped.append({"path": str(relative), "bytes": size, "reason": "not a compact report/profile/preview"})
    for name in ("jobs.json", "tokens_manifest.json", "requirements.txt"):
        path = ROOT / name
        if path.is_file():
            compact.append((path, name))
    # CPU evaluation writes small tables beside the output root in some workflows.
    for name in ("metrics.csv", "metrics.json", "analysis.json", "analysis.csv", "weights_provenance.json"):
        path = ROOT / name
        if path.is_file() and path.stat().st_size <= 16 * 1024**2:
            compact.append((path, name))

    def pack(destination, entries, kind):
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path, name in entries:
                z.write(path, name)
            z.writestr("collection.json", json.dumps({"created_at": now(), "kind": kind,
                "files": len(entries), "excluded_directories": sorted(EXCLUDED),
                "skipped": skipped if kind == "compact" else [],
                "note": "Snapshot of existing outputs; no generation or evaluation performed."}, ensure_ascii=False, indent=2))
        print(f"{destination} ({len(entries)} files, {destination.stat().st_size / 1024**2:.2f} MiB)")

    pack(archive, compact, "compact")
    if args.native:
        pack(native_archive, native, "native")
    return 0


if __name__ == "__main__":
    sys.exit(main())
