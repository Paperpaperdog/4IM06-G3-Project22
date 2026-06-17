#!/usr/bin/env bash
# Stage results/ files under a size limit (default 50MB). Respects .gitignore unless
# -f is passed through to git add for explicitly listed force-add patterns.
#
#   bash scripts/git_add_results.sh              # dry-run (list only)
#   bash scripts/git_add_results.sh --add        # git add matching files
#   MAX_MB=10 bash scripts/git_add_results.sh --add
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/results"
MAX_MB="${MAX_MB:-50}"
MAX_BYTES=$((MAX_MB * 1024 * 1024))
DO_ADD=0
if [[ "${1:-}" == "--add" ]]; then
  DO_ADD=1
fi

if [[ ! -d "$RESULTS" ]]; then
  echo "No results/ directory at $RESULTS" >&2
  exit 1
fi

added=0
skipped_size=0
skipped_ignore=0

while IFS= read -r -d '' file; do
  rel="${file#"$ROOT/"}"
  size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file")
  size_h="${size}B"
  if (( size >= 1048576 )); then
    size_h="$(awk "BEGIN {printf \"%.1fMB\", $size/1048576}")"
  elif (( size >= 1024 )); then
    size_h="$(awk "BEGIN {printf \"%.1fKB\", $size/1024}")"
  fi
  if (( size > MAX_BYTES )); then
    echo "SKIP size ${size} > ${MAX_MB}MB: $rel"
    skipped_size=$((skipped_size + 1))
    continue
  fi
  if git check-ignore -q "$rel" 2>/dev/null; then
    # Still allow force-add for checkpoints if under size limit
    case "$rel" in
      results/*/checkpoints/*.pt|results/*/*/checkpoints/*.pt)
        if (( DO_ADD )); then
          git add -f "$rel"
        fi
        echo "ADD (force) $rel ($size_h)"
        added=$((added + 1))
        ;;
      *)
        echo "SKIP ignored: $rel"
        skipped_ignore=$((skipped_ignore + 1))
        ;;
    esac
  else
    if (( DO_ADD )); then
      git add "$rel"
    fi
    echo "ADD $rel ($size_h)"
    added=$((added + 1))
  fi
done < <(find "$RESULTS" -type f ! -name '.gitkeep' -print0)

echo "---"
echo "Would add / added: $added | skipped size: $skipped_size | skipped ignore: $skipped_ignore | limit: ${MAX_MB}MB"
if (( DO_ADD == 0 )); then
  echo "Dry-run only. Re-run with: bash scripts/git_add_results.sh --add"
fi
