#!/usr/bin/env bash
# Download, verify, merge and extract one file group of the
# MultiModalSpectralTransformer Zenodo data repository.
#
# The three DOI links are complementary halves of one split archive set:
#   record 16076914 : data.tar.xz.partaa   .. partbc   (29 parts)
#                     models.tar.xz.partaa .. partak   (11 parts)
#                     past_experiments.tar.xz.partaa .. partah (8 parts)
#   record 16257786 : data.tar.xz.partae (duplicate), partbd, partbf .. partcx
#   record 17284940 : data.tar.xz.partbe, partcy .. partdk, plus the reviewer zip
# The full data.tar.xz is therefore 89 distinct parts spread over three records.
#
# Usage:
#   bash scripts/download_zenodo_group.sh <group> [repository_root]
#
# Groups:
#   models            11 parts   11.08 GB   record 16076914
#   past_experiments   8 parts    8.29 GB   record 16076914
#   data              89 parts   93.16 GB   records 16076914 16257786 17284940
#   reviewer           1 zip      1.80 GB   record 17284940
#
# Two download modes:
#   default     all parts first (resumable), then one streaming extraction
#   LOW_DISK=1  parts are fetched in archive order, piped into tar one by one
#               and deleted immediately, so peak disk use is one part plus the
#               extracted tree; a broken session cannot be resumed because the
#               already-consumed parts are gone
#
# ONLY_PATHS limits what tar writes to disk, e.g.
#   ONLY_PATHS='data/IBM_dataset/* data/test_data/*'
# The xz stream is single and sequential, so every part is still downloaded and
# read; only the disk footprint shrinks to the selected files.
#
# Every tar archive already contains its own top-level directory (models/,
# data/, past_experiments/), so it is extracted into the repository root.
# Parts are deleted only after a successful extraction.
set -euo pipefail

GROUP="${1:-models}"
ROOT="${2:-/content/MultiModalSpectralTransformer}"
PARTS_DIRECTORY="$ROOT/zenodo_parts"
ARIA2_JOBS="${ARIA2_JOBS:-4}"
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-16}"
LOW_DISK="${LOW_DISK:-0}"
ONLY_PATHS="${ONLY_PATHS:-}"

# tar arguments for selective extraction; empty means "extract everything"
TAR_MEMBER_ARGS=()
if [ -n "$ONLY_PATHS" ]; then
    read -r -a SELECTED_PATHS <<< "$ONLY_PATHS"
    TAR_MEMBER_ARGS=(--wildcards "${SELECTED_PATHS[@]}")
    echo "selective extraction: $ONLY_PATHS (the full stream is still read)"
fi

case "$GROUP" in
    models)
        RECORD_IDS="16076914"
        PREFIX="models.tar.xz"
        TARGET_DIRECTORY="."
        ;;
    past_experiments)
        RECORD_IDS="16076914"
        PREFIX="past_experiments.tar.xz"
        TARGET_DIRECTORY="."
        ;;
    data)
        RECORD_IDS="16076914 16257786 17284940"
        PREFIX="data.tar.xz"
        TARGET_DIRECTORY="."
        ;;
    reviewer)
        RECORD_IDS="17284940"
        PREFIX="Reviewer_Experiment_Global_Impact"
        TARGET_DIRECTORY="past_experiments/ChemXriv"
        ;;
    *)
        echo "unknown group: $GROUP (use models, past_experiments, data or reviewer)" >&2
        exit 2
        ;;
esac

mkdir -p "$ROOT" "$PARTS_DIRECTORY" "$ROOT/$TARGET_DIRECTORY"
ROOT="$(cd "$ROOT" && pwd)"  # normalize so the cd below cannot break later paths
PARTS_DIRECTORY="$ROOT/zenodo_parts"
cd "$ROOT"

# aria2 and unzip are not part of every Colab image
if ! command -v aria2c >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq aria2
fi
if [ "$GROUP" = "reviewer" ] && ! command -v unzip >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq unzip
fi

# Build the manifest once: deduplicate keys shared by several records and
# refuse the download when the destination disk cannot hold it
python3 - "$PREFIX" "$PARTS_DIRECTORY" $RECORD_IDS <<'PYTHON'
import json
import os
import shutil
import sys
import urllib.request

prefix, parts_directory = sys.argv[1], sys.argv[2]
record_ids = sys.argv[3:]
low_disk = os.environ.get("LOW_DISK") == "1"

manifest = {}
for record_id in record_ids:
    api_url = f"https://zenodo.org/api/records/{record_id}"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        files = json.load(response)["files"]
    for entry in files:
        if not entry["key"].startswith(prefix):
            continue
        known = manifest.get(entry["key"])
        if known is not None and known["checksum"] != entry["checksum"]:
            sys.exit(f"conflict: {entry['key']} differs between records")
        manifest[entry["key"]] = entry

entries = sorted(manifest.values(), key=lambda entry: entry["key"])
if not entries:
    sys.exit(f"no files matched prefix {prefix}")

total_bytes = sum(entry["size"] for entry in entries)
largest_part = max(entry["size"] for entry in entries)
free_bytes = shutil.disk_usage(parts_directory).free
mode = "low-disk" if low_disk else "bulk"
print(f"{len(entries)} files, {total_bytes / 1e9:.2f} GB compressed, "
      f"{free_bytes / 1e9:.1f} GB free, mode {mode}")

# Bulk mode keeps every part until extraction, plus a growing extracted tree.
# Low-disk mode keeps a single part at a time, so a few GiB of headroom suffice.
required_bytes = int(largest_part * 3) if low_disk else int(total_bytes * 1.5)
if os.environ.get("SKIP_SPACE_CHECK") != "1" and free_bytes < required_bytes:
    sys.exit(
        f"not enough free space: {free_bytes / 1e9:.1f} GB available, "
        f"about {required_bytes / 1e9:.1f} GB recommended for mode {mode} "
        f"(re-run with SKIP_SPACE_CHECK=1 to override)"
    )

with open(os.path.join(parts_directory, "manifest.json"), "w") as handle:
    json.dump({entry["key"]: entry for entry in entries}, handle)

with open(os.path.join(parts_directory, "aria2_input.txt"), "w") as handle:
    for entry in entries:
        handle.write(entry["links"]["self"] + "\n")
        handle.write(f"  out={entry['key']}\n")
        # The API returns "md5:<digest>", aria2 expects "md5=<digest>"
        handle.write(f"  checksum={entry['checksum'].replace(':', '=')}\n")

# name, size, url and checksum of every part, in archive order
with open(os.path.join(parts_directory, "parts_manifest.tsv"), "w") as handle:
    for entry in entries:
        handle.write(
            f"{entry['key']}\t{entry['size']}\t"
            f"{entry['links']['self']}\t{entry['checksum'].replace(':', '=')}\n"
        )
PYTHON

if [ "$LOW_DISK" = "1" ] && [ "$GROUP" != "reviewer" ]; then
    # Ordered streaming: fetch one part, push it into tar, delete it again.
    # tar reads the concatenation of all parts as a single xz stream, so the
    # parts must arrive in exactly the order written to parts_manifest.tsv.
    echo "low-disk mode: parts are streamed into tar and deleted one by one"
    (
        while IFS=$'\t' read -r PART_NAME PART_SIZE PART_URL PART_CHECKSUM; do
            # All progress output must go to stderr: stdout is the tar stream
            echo "part $PART_NAME (${PART_SIZE} bytes)" >&2
            aria2c --dir="$PARTS_DIRECTORY" --out="$PART_NAME" "$PART_URL" \
                --split="$ARIA2_CONNECTIONS" \
                --max-connection-per-server="$ARIA2_CONNECTIONS" \
                --min-split-size=16M --continue=true \
                --auto-file-renaming=false --file-allocation=none \
                --disk-cache=64M --max-tries=5 --retry-wait=5 \
                --checksum="$PART_CHECKSUM" --summary-interval=20 1>&2
            cat "$PARTS_DIRECTORY/$PART_NAME"
            rm -f "$PARTS_DIRECTORY/$PART_NAME" "$PARTS_DIRECTORY/$PART_NAME.aria2"
        done < "$PARTS_DIRECTORY/parts_manifest.tsv"
    ) | tar -xJf - -C "$ROOT" ${TAR_MEMBER_ARGS[@]+"${TAR_MEMBER_ARGS[@]}"}
else
    # Bulk mode: parallel downloads, 16 connections per file, per-part MD5 check
    cd "$PARTS_DIRECTORY"
    aria2c --input-file=aria2_input.txt --dir="$PARTS_DIRECTORY" \
        --max-concurrent-downloads="$ARIA2_JOBS" \
        --split="$ARIA2_CONNECTIONS" \
        --max-connection-per-server="$ARIA2_CONNECTIONS" \
        --min-split-size=16M \
        --continue=true --auto-file-renaming=false --file-allocation=none \
        --disk-cache=64M --max-tries=5 --retry-wait=5 \
        --summary-interval=20

    # Size check against the API manifest before spending time on extraction
    python3 - "$PARTS_DIRECTORY" <<'PYTHON'
import json
import os
import sys

parts_directory = sys.argv[1]
with open(os.path.join(parts_directory, "manifest.json")) as handle:
    manifest = json.load(handle)

missing = []
for name, entry in manifest.items():
    path = os.path.join(parts_directory, name)
    if not os.path.exists(path) or os.path.getsize(path) != entry["size"]:
        missing.append(name)
if missing:
    sys.exit("incomplete download: " + ", ".join(missing))
print(f"all {len(manifest)} files present with the expected sizes")
PYTHON

    mapfile -t PART_NAMES < <(cut -f1 parts_manifest.tsv)

    if [ "$GROUP" = "reviewer" ]; then
        echo "extracting ${PART_NAMES[0]} into $TARGET_DIRECTORY"
        unzip -oq "${PART_NAMES[0]}" -d "$ROOT/$TARGET_DIRECTORY"
    else
        echo "streaming ${#PART_NAMES[@]} parts through xz into the repository root"
        if command -v pv >/dev/null 2>&1; then
            cat "${PART_NAMES[@]}" | pv | tar -xJf - -C "$ROOT" ${TAR_MEMBER_ARGS[@]+"${TAR_MEMBER_ARGS[@]}"}
        else
            cat "${PART_NAMES[@]}" | tar -xJf - -C "$ROOT" ${TAR_MEMBER_ARGS[@]+"${TAR_MEMBER_ARGS[@]}"}
        fi
    fi

    # Extraction succeeded: only now remove the downloaded parts
    rm -f "${PART_NAMES[@]}" ./*.aria2
fi

rm -f "$PARTS_DIRECTORY"/aria2_input.txt "$PARTS_DIRECTORY"/manifest.json \
    "$PARTS_DIRECTORY"/parts_manifest.tsv "$PARTS_DIRECTORY"/parts_order.txt \
    "$PARTS_DIRECTORY"/*.aria2
echo "done: $GROUP extracted into $ROOT/$TARGET_DIRECTORY"
df -h "$ROOT" | tail -n 1
