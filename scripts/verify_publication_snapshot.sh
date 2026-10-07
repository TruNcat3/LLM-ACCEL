#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
case "${1:---head}" in
    --head) snapshot_tree="$(git rev-parse 'HEAD^{tree}')" ;;
    --index) snapshot_tree="$(git write-tree)" ;;
    *) echo "usage: $0 [--head|--index]" >&2; exit 2 ;;
esac
[[ $# -le 1 ]] || { echo "Expected at most one snapshot selector" >&2; exit 2; }

# Export Git objects, never the worktree: ignored or unstaged files cannot
# accidentally satisfy a release check. All temporary build outputs stay here.
snapshot_root="$(mktemp -d "${TMPDIR:-/tmp}/cowave-publication-snapshot.XXXXXX")"
trap 'rm -rf -- "${snapshot_root}"' EXIT
git archive "${snapshot_tree}" | tar -x -C "${snapshot_root}"
git -C "${snapshot_root}" init -q
git -C "${snapshot_root}" add -f -- .
git -C "${snapshot_root}" -c user.name='CoWave snapshot check' \
    -c user.email='snapshot@localhost' -c commit.gpgsign=false \
    commit -qm 'Temporary publication snapshot'

printf 'publication_snapshot_tree=%s\n' "${snapshot_tree}"
QWEN3B_PLAN_BUILD_DIR="${snapshot_root}/.snapshot-build" \
    VERIFY_ROOT_CHECKSUMS=1 make -C "${snapshot_root}" test_publication_release
printf 'PUBLICATION SNAPSHOT PASS tree=%s\n' "${snapshot_tree}"
