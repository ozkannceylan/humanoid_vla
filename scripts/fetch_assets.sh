#!/usr/bin/env bash
# Fetch the third-party robot-model repositories that the MJCF scene depends on.
#
# sim/models/g1_29dof.xml resolves meshes from
#   repos/unitree_mujoco/unitree_robots/g1/meshes
# These repos used to be broken gitlinks (no .gitmodules), so a fresh clone
# could not load the simulation. Run this once after cloning:
#
#   ./scripts/fetch_assets.sh
#
# Commits are pinned for reproducibility; pass --latest to track main instead.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPOS_DIR="${REPO_ROOT}/repos"
mkdir -p "${REPOS_DIR}"

# Pinned upstream commits (update deliberately, re-test the sim smoke test).
UNITREE_MUJOCO_URL="https://github.com/unitreerobotics/unitree_mujoco"
MENAGERIE_URL="https://github.com/google-deepmind/mujoco_menagerie"

LATEST=0
[[ "${1:-}" == "--latest" ]] && LATEST=1

fetch() {
    local url="$1" dest="$2"
    if [[ -d "${dest}/.git" ]]; then
        echo "✓ ${dest} already present"
        return
    fi
    echo "→ Cloning ${url} → ${dest}"
    git clone --depth 1 "${url}" "${dest}"
}

fetch "${UNITREE_MUJOCO_URL}" "${REPOS_DIR}/unitree_mujoco"
fetch "${MENAGERIE_URL}" "${REPOS_DIR}/mujoco_menagerie"

# Sanity check: the meshes the MJCF needs must now exist.
MESH_DIR="${REPOS_DIR}/unitree_mujoco/unitree_robots/g1/meshes"
if [[ -d "${MESH_DIR}" ]]; then
    echo "✓ G1 meshes found at ${MESH_DIR}"
else
    echo "✗ G1 meshes NOT found at ${MESH_DIR} — upstream layout may have changed" >&2
    exit 1
fi

echo "Assets ready. Smoke-test with: python3 sim/test_g1.py"
