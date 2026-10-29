#!/usr/bin/env bash
# =============================================================================
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
# Download strategy: every part is fetched first (parallel, resumable) and
# checked against the MD5 published by Zenodo, then the parts are concatenated
# in archive order and streamed through xz/tar in one pass.
#
# Every tar archive already contains its own top-level directory (models/,
# data/, past_experiments/), so it is extracted into the repository root.
# Parts are deleted only after a successful extraction.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Shell crash course: every construct this file uses, in reading order
# -----------------------------------------------------------------------------
# #!/usr/bin/env bash
#     First line of a script. It names the interpreter that runs the file when
#     the file is started directly, i.e. `./scripts/download_zenodo_group.sh`
#     runs exactly like `bash scripts/download_zenodo_group.sh` (the direct
#     form additionally needs the executable bit: chmod +x).
#
# set -euo pipefail
#     Three safety switches:
#       -e          stop at the first command that fails (non-zero exit code)
#       -u          treat an unset variable as an error instead of an empty text
#       -o pipefail a pipeline (`a | b`) fails if any stage fails, not only the last
#
# NAME="value"          assign text to a variable. No spaces around `=`.
# "$NAME"               read a variable. Always quote it, so spaces in the value
#                       stay one piece of text instead of being split on spaces.
# ${1:-models}          "argument 1, or `models` if the caller passed nothing".
#                       $1 is the first argument, $2 the second, and so on.
# $#                    how many arguments were given.
# $@                    all arguments, each kept as its own word.
#
# $(command)            run a command and paste its output into the line.
# command -v name       print the path of a program; fails if it is not installed.
# a && b                run b only if a succeeded.
# if ...; then ...; fi  classic conditional block; `fi` closes it.
# case "$X" in ... esac match a value against several patterns; `*)` is "anything else".
# mkdir -p path         create a directory, and silently accept that it exists.
# cd path               change the current directory for the rest of the script.
# rm -f files           delete files, never complain if they are already gone.
#
# > file                write stdout to a file, overwriting it.
# >> file               append stdout to a file.
# 1>&2                  send stdout to stderr (progress text that must not be
#                       mixed into real data).
# 2>/dev/null           hide stderr (throw away error text).
# |                     pipeline: stdout of the left side becomes the stdin of
#                       the right side, with no temporary file in between.
#
# <<'WORD' ... WORD      heredoc: a block of text (here: Python code) fed to the
#                       previous command as its stdin. The quotes around WORD
#                       switch off all $variable expansion inside the block.
#
# VAR="$(...)"          capture a command output into a variable.
# mapfile -t ARRAY < <(command)
#                       run `command`, store each output line as one array item;
#                       the inner `<(` is process substitution, i.e. a temporary
#                       stream that behaves like a file.
# cut -f1 file          print only the first TAB-separated field of every line.
# ${#ARRAY[@]}          number of items in an array.
# ${ARRAY[0]}           first item of an array.
# =============================================================================

# -----------------------------------------------------------------------------
# set -euo pipefail
# Runs before anything else, because its whole job is to change how every later
# command is judged. It is a single line, but it is three independent switches:
#
# -e (errexit)
#     Stop the script the moment a command returns a non-zero exit code.
#     In this script that means: a failed `aria2c` download, a failed MD5 check,
#     a failed `apt-get install`, or a failed `unzip` ends the run immediately,
#     so no half-downloaded set of parts is ever merged or extracted.
#     Two spots here rely on the documented exceptions: a failing command inside
#     an `if` condition (`if ! command -v aria2c`) or on the left of `&&` does
#     NOT stop the script, it only makes the condition false.
#
# -u (nounset)
#     Reading an unset variable is an error instead of silently returning "".
#     That is why every variable below is written as "${NAME:-default}": the
#     ":-" form supplies a value when the caller passed none, which keeps -u
#     happy. A typo such as "$PART_NAM" would abort the run instead of feeding
#     aria2 an empty file name.
#
# -o pipefail
#     A pipeline reports failure when ANY stage fails, not just the last one.
#     Without it, `cat parts | tar -xJf -` would report success as long as tar
#     exited cleanly, even if `cat` died halfway through and tar received a
#     truncated stream (tar can finish "successfully" on a short archive).
#     With it, a broken read is reported, and the `rm` after extraction is
#     skipped so the parts stay on disk for a retry.
#
# Practical consequences when editing this script:
#   - never wrap a fallible command in `cmd || true` unless failure is truly
#     acceptable, because that defeats -e;
#   - `set +e` would switch errexit off for the rest of the script, so prefer
#     putting tolerance in an explicit `if` instead;
#   - an empty array expands safely only via "${ARRAY[@]+"${ARRAY[@]}"}"
#     (see the glossary above), which is required because of -u.
# -----------------------------------------------------------------------------
set -euo pipefail

# First argument = which group to fetch; defaults to "models".
GROUP="${1:-models}"
# Absolute directory holding this script, so the default root works on any machine.
SCRIPT_DIRECTORY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Second argument = repository root; defaults to the parent of scripts/.
ROOT="${2:-$(cd "$SCRIPT_DIRECTORY/.." && pwd)}"
# Downloaded parts live in a staging directory inside the repository.
PARTS_DIRECTORY="$ROOT/zenodo_parts"
# How many files aria2 downloads at the same time.
ARIA2_JOBS="${ARIA2_JOBS:-4}"
# How many TCP connections aria2 opens per file.
ARIA2_CONNECTIONS="${ARIA2_CONNECTIONS:-16}"

# Pick the record ids, archive prefix and output directory for the group.
case "$GROUP" in
    models)
        RECORD_IDS="16076914"
        PREFIX="models.tar.xz"
        TARGET_DIRECTORY="."    # "." means the repository root itself
        ;;
    past_experiments)
        RECORD_IDS="16076914"
        PREFIX="past_experiments.tar.xz"
        TARGET_DIRECTORY="."
        ;;
    data)
        # The data archive is split over all three records.
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
        # `>&2` writes the message to stderr; exit 2 ends the script as an error.
        echo "unknown group: $GROUP (use models, past_experiments, data or reviewer)" >&2
        exit 2
        ;;
esac

mkdir -p "$ROOT" "$PARTS_DIRECTORY" "$ROOT/$TARGET_DIRECTORY"
# Turn the user-supplied path into an absolute one, so the `cd` below is safe.
ROOT="$(cd "$ROOT" && pwd)"  # normalize so the cd below cannot break later paths
PARTS_DIRECTORY="$ROOT/zenodo_parts"
cd "$ROOT"

# -----------------------------------------------------------------------------
# aria2 is not part of every Colab image, so install it on demand.
# Line by line:
#
# `command -v aria2c`
#     Asks the shell whether a program named aria2c exists in PATH. It prints
#     the full path and exits with 0 when found, exits with 1 when not found,
#     and changes nothing on disk. It is the portable "is this installed?" test
#     (`which` is an external program and is not available everywhere).
#
# `>/dev/null 2>&1`
#     Silences that test. `>/dev/null` sends stdout to the null device; `2>&1`
#     then points stderr (fd 2) at wherever stdout currently points, i.e. also
#     /dev/null. The order is what makes it work: writing `2>&1 >/dev/null`
#     would send stderr to the terminal first and only then redirect stdout.
#     Without this, a missing program would print "command not found" on every
#     run just to answer a question.
#
# `!` in front of the command
#     Inverts the exit status: 0 becomes 1 and 1 becomes 0. So the block runs
#     exactly when the program is ABSENT, which is the only time we install.
#     (This is also why `set -e` does not abort here: a failing command inside
#     an `if` condition is a normal, expected outcome.)
#
# `apt-get update -qq`
#     Refreshes the package index before installing. `-qq` means "very quiet":
#     only warnings and errors are printed, which keeps the download log
#     readable. Skipping this leaves apt with a possibly stale index.
#
# `apt-get install -y -qq aria2`
#     Installs the Debian package `aria2`, which provides the binary `aria2c`
#     used below. `-y` answers "yes" to the confirmation prompt, so the script
#     never blocks waiting for input. No `sudo` is needed because Colab runs as
#     root; on a normal Debian/Ubuntu desktop it would have to be `sudo apt-get`.
#     On macOS this block does not apply at all (`brew install aria2` instead).
#
# Because of `set -e`, a failed install aborts the run right here, with apt's
# own error message, rather than failing later with "aria2c: command not found".
# The check also keeps re-runs fast: no apt index refresh when it is installed.
# -----------------------------------------------------------------------------
if ! command -v aria2c >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq aria2
fi
if [ "$GROUP" = "reviewer" ] && ! command -v unzip >/dev/null 2>&1; then
    apt-get update -qq
    apt-get install -y -qq unzip
fi

# =============================================================================
# STEP 1 OF 4: build the download manifest (this block is Python, not shell)
#
# What it produces, in $PARTS_DIRECTORY (<repo>/zenodo_parts):
#   manifest.json        name -> {size, checksum, url}, used by the size check
#   aria2_input.txt      the "answer file" read by `aria2c --input-file`
#   parts_manifest.tsv   name<TAB>size<TAB>url<TAB>checksum, in archive order
#
# Why this step exists, and why Python:
#   * the authoritative file list lives on Zenodo, so it must be fetched over
#     HTTPS and parsed as JSON - awkward in shell, three lines in Python;
#   * the SAME part can be published by two records, so the lists must be
#     merged, de-duplicated by file name, and the two MD5 values compared;
#   * the free space must be compared with a size estimate BEFORE 93 GB of
#     traffic starts, so the user gets a clear "not enough space" instead of a
#     disk-full failure in the middle of part 40.
#
# Concrete example, GROUP=data:
#   RECORD_IDS="16076914 16257786 17284940"
#   record 16076914 lists data.tar.xz.partaa ... partbc
#   record 16257786 lists data.tar.xz.partae (again) and partbd, partbf ... partcx
#   record 17284940 lists data.tar.xz.partbe and partcy ... partdk
#   -> after merging: 89 unique names, partae stored once because both records
#      publish the same md5; sorted by name they run aa, ab, ... dk
#   -> parts_manifest.tsv gets 89 rows, 93.16 GB in total
#   If partae had a different md5 in the two records, the run stops with
#   "conflict: data.tar.xz.partae differs between records" instead of
#   downloading two incompatible halves of one xz stream.
#
# How the shell passes its variables in:
#   `python3 -` means "run Python on stdin", i.e. on the heredoc below. The
#   words after the dash are the program's own arguments: $PREFIX becomes
#   sys.argv[1], $PARTS_DIRECTORY sys.argv[2], and $RECORD_IDS is deliberately
#   UNQUOTED so that "16076914 16257786 17284940" splits on spaces into three
#   separate arguments (sys.argv[3], sys.argv[4], sys.argv[5]).
#   The heredoc marker is quoted ('PYTHON') so the shell does NOT expand $...,
#   f-strings and $ signs inside the code below stay Python.
# =============================================================================
python3 - "$PREFIX" "$PARTS_DIRECTORY" $RECORD_IDS <<'PYTHON'
# json: parse the API answer. os: join paths, read environment variables.
# shutil: ask the OS for free disk space. sys: arguments and exit codes.
# urllib.request: perform the HTTPS GET without any third-party library.
import json
import os
import shutil
import sys
import urllib.request

# sys.argv is the list of words after `python3 -`, in the same order.
# Example with GROUP=data and a staging dir under /content:
#   sys.argv[1] == "data.tar.xz"
#   sys.argv[2] == "/content/MultiModalSpectralTransformer/zenodo_parts"
#   sys.argv[3:] == ["16076914", "16257786", "17284940"]
prefix, parts_directory = sys.argv[1], sys.argv[2]
record_ids = sys.argv[3:]

# Key = archive file name, value = that file's Zenodo API entry.
# Example after the three records are merged (data group):
#   {"data.tar.xz.partaa": {...}, "data.tar.xz.partab": {...}, ... 89 items}
# A plain dict is used so a repeated name simply overwrites nothing new: the
# merge below compares checksums first.
manifest = {}
# One iteration per record id, e.g. record 16076914 first.
for record_id in record_ids:
    # Example URL: https://zenodo.org/api/records/16076914
    api_url = f"https://zenodo.org/api/records/{record_id}"
    # Fetch the machine-readable file list of one record.
    # `with` closes the connection automatically, even on error; timeout=60
    # aborts instead of hanging forever if Zenodo does not answer.
    with urllib.request.urlopen(api_url, timeout=60) as response:
        # The JSON body has a "files" list. One entry looks like:
        #   {"key": "data.tar.xz.partaa", "size": 1048576000,
        #    "checksum": "md5:<hex>",
        #    "links": {"self": "https://zenodo.org/api/records/.../files/..."}}
        files = json.load(response)["files"]
    # One iteration per file of that record, e.g. 29 data parts for 16076914.
    for entry in files:
        # Keep only the requested archive: "data.tar.xz.partaa" matches the
        # prefix "data.tar.xz", while "models.tar.xz.partaa" and
        # "Reviewer_Experiment_Global_Impact.zip" are skipped here.
        if not entry["key"].startswith(prefix):
            continue
        # First time: None. Second record: the entry already collected.
        # Example: partae -> None on record 16076914, dict on record 16257786.
        known = manifest.get(entry["key"])
        # A part may be listed by several records; only identical copies are
        # acceptable. Same name + different md5 would mean the archive was
        # rebuilt in between, so the two halves would not concatenate.
        if known is not None and known["checksum"] != entry["checksum"]:
            sys.exit(f"conflict: {entry['key']} differs between records")
        # Store (or overwrite with the identical duplicate) the entry.
        manifest[entry["key"]] = entry

# Sort by file name, because xz needs the parts in the order they were cut.
# Example: partaa, partab, ... partaz, partba, ... partdk - the two-letter
# suffix is always the same length, so plain string sorting gives exactly the
# original byte order (unlike "part1, part10, part2" numeric-style names).
entries = sorted(manifest.values(), key=lambda entry: entry["key"])
# Example: PREFIX="data.tar.xz." (typo) matches nothing -> clear error instead
# of an empty download and a confusing tar failure later.
if not entries:
    sys.exit(f"no files matched prefix {prefix}")

# Sum over all parts, e.g. 93.16e9 bytes for the 89 data parts.
total_bytes = sum(entry["size"] for entry in entries)
# Free bytes of the filesystem holding the staging directory, e.g. 108e9 on a
# Colab temporary drive or 563e9 on a 1 TB data disk.
free_bytes = shutil.disk_usage(parts_directory).free
# One status line, e.g.
#   89 files, 93.16 GB compressed, 563.4 GB free
print(f"{len(entries)} files, {total_bytes / 1e9:.2f} GB compressed, "
      f"{free_bytes / 1e9:.1f} GB free")

# Space needed, computed from measured values instead of a blind multiplier.
# The tables below are `du -sh` results of a full, checksum-verified extraction
# on a 1 TB data disk; a single factor cannot describe every group because text
# CSV archives expand far more than binary model weights:
#   data.tar.xz              93.16 GB compressed -> 258 GiB extracted
#   past_experiments.tar.xz   8.29 GB compressed -> 110 GiB extracted
#   models.tar.xz            11.08 GB compressed ->  13 GiB extracted
#   Reviewer_..._Impact.zip   1.80 GB compressed -> 3.41 GiB extracted
MEASURED_EXPANDED_GIB = {
    "data.tar.xz": 260,              # 258 GiB measured, rounded up
    "past_experiments.tar.xz": 112,  # 110 GiB measured, rounded up
    "models.tar.xz": 14,             #  13 GiB measured, rounded up
    # 3.41 GiB (3,664,137,508 bytes) measured, rounded up
    "Reviewer_Experiment_Global_Impact": 4,
}
GIB = 2 ** 30
# Every group in the `case` above has a measured entry, so a plain lookup is
# enough; an unknown prefix would raise KeyError before downloading anything.
extracted_bytes = MEASURED_EXPANDED_GIB[prefix] * GIB
# The parts and the extracted tree coexist until tar has consumed the last one.
# Example, data group: (93.16 GB + 260 GiB) * 1.05 = ~391 GB.
# Example, models group: (11.08 GB + 14 GiB) * 1.05 = ~27 GB.
# Example, reviewer zip: (1.80 GB + 4 GiB) * 1.05 = ~6.4 GB.
required_bytes = int((total_bytes + extracted_bytes) * 1.05)
# SKIP_SPACE_CHECK=1 is the documented escape hatch for users who know that the
# extracted tree lives on another disk, e.g.
#   SKIP_SPACE_CHECK=1 bash scripts/download_zenodo_group.sh data \
#       /content/MultiModalSpectralTransformer
if os.environ.get("SKIP_SPACE_CHECK") != "1" and free_bytes < required_bytes:
    # Example message: "not enough free space: 19.9 GB available, about
    # 391.0 GB recommended (re-run with SKIP_SPACE_CHECK=1 to override)"
    sys.exit(
        f"not enough free space: {free_bytes / 1e9:.1f} GB available, "
        f"about {required_bytes / 1e9:.1f} GB recommended "
        f"(re-run with SKIP_SPACE_CHECK=1 to override)"
    )

# manifest.json: name -> full API entry, read again by the size check later.
# Kept as JSON (not TSV) because it stores nested fields (links, checksum).
with open(os.path.join(parts_directory, "manifest.json"), "w") as handle:
    json.dump({entry["key"]: entry for entry in entries}, handle)

# aria2_input.txt: the answer file for `aria2c --input-file`.
# Format is one URL line, then indented option lines, then a blank separator:
#   https://zenodo.org/api/records/16076914/files/data.tar.xz.partaa/content
#     out=data.tar.xz.partaa
#     checksum=md5=<hex>
#   https://.../data.tar.xz.partab/content
#     out=data.tar.xz.partab
#     checksum=md5=<hex>
with open(os.path.join(parts_directory, "aria2_input.txt"), "w") as handle:
    # One URL per part; aria2 writes the result under the name in `out=`.
    for entry in entries:
        handle.write(entry["links"]["self"] + "\n")
        # `out=` fixes the local file name, so parts never get renamed to
        # "data.tar.xz.partaa.1" and stay mergeable in order.
        handle.write(f"  out={entry['key']}\n")
        # The API returns "md5:<digest>", aria2 expects "md5=<digest>".
        # Example: "md5:1a2b..." -> "md5=1a2b..." (a ':' would be rejected).
        handle.write(f"  checksum={entry['checksum'].replace(':', '=')}\n")

# name, size, url and checksum of every part, in archive order.
# This is the file the shell loops over: one line per part, for example
#   data.tar.xz.partaa<TAB>1048576000<TAB>https://.../content<TAB>md5=<hex>
# "<TAB>" is a literal tab character, because the shell reads it with
# IFS=$'\t' and would split on spaces otherwise (URLs contain no tabs).
with open(os.path.join(parts_directory, "parts_manifest.tsv"), "w") as handle:
    for entry in entries:
        # "w" truncates the file: every run starts from a fresh manifest.
        handle.write(
            f"{entry['key']}\t{entry['size']}\t"
            f"{entry['links']['self']}\t{entry['checksum'].replace(':', '=')}\n"
        )
PYTHON

# Download every part in parallel (16 connections per file, per-part MD5 check
# done by aria2 itself through the `checksum=` option in aria2_input.txt).
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

# Any part that is missing or has the wrong number of bytes aborts the run,
# so a truncated download is never merged silently.
missing = []
for name, entry in manifest.items():
    path = os.path.join(parts_directory, name)
    if not os.path.exists(path) or os.path.getsize(path) != entry["size"]:
        missing.append(name)
if missing:
    sys.exit("incomplete download: " + ", ".join(missing))
print(f"all {len(manifest)} files present with the expected sizes")
PYTHON

# Turn the first TSV column (the file names) into an array.
mapfile -t PART_NAMES < <(cut -f1 parts_manifest.tsv)

if [ "$GROUP" = "reviewer" ]; then
    # A single zip: no concatenation, no xz, just unzip it.
    echo "extracting ${PART_NAMES[0]} into $TARGET_DIRECTORY"
    unzip -oq "${PART_NAMES[0]}" -d "$ROOT/$TARGET_DIRECTORY"
else
    # Feed all parts to tar in name order, which is archive order.
    echo "streaming ${#PART_NAMES[@]} parts through xz into the repository root"
    if command -v pv >/dev/null 2>&1; then
        # pv is optional: it only draws a throughput meter in the middle.
        cat "${PART_NAMES[@]}" | pv | tar -xJf - -C "$ROOT"
    else
        cat "${PART_NAMES[@]}" | tar -xJf - -C "$ROOT"
    fi
fi

# Extraction succeeded: only now remove the downloaded parts
rm -f "${PART_NAMES[@]}" ./*.aria2

# Remove bookkeeping files; the staging directory itself stays for the next run.
rm -f "$PARTS_DIRECTORY"/aria2_input.txt "$PARTS_DIRECTORY"/manifest.json \
    "$PARTS_DIRECTORY"/parts_manifest.tsv "$PARTS_DIRECTORY"/parts_order.txt \
    "$PARTS_DIRECTORY"/*.aria2
echo "done: $GROUP extracted into $ROOT/$TARGET_DIRECTORY"
# Show one line of `df -h`, i.e. the free space left after the extraction.
df -h "$ROOT" | tail -n 1
