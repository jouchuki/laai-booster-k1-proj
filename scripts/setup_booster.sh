#!/usr/bin/env bash
# Reuse Booster repos at the project root or in the older booster/ subfolder.
# New checkouts go at the project root at the commits used by CI and the cluster.
#
# Linux and macOS usage (from the repo root):
#   scripts/setup_booster.sh            # clone both repos
#   scripts/setup_booster.sh assets     # only booster_assets (what CI needs)
#
# To bump a version: change the commit below, run the script, check tests pass, commit.
# Once the team forks booster_deploy, point BOOSTER_DEPLOY_REPO at the fork.
set -euo pipefail

BOOSTER_ASSETS_REPO="${BOOSTER_ASSETS_REPO:-https://github.com/BoosterRobotics/booster_assets.git}"
BOOSTER_ASSETS_COMMIT="12a97516f57469078707afb587a5f84c1cb110d1"   # 2026-09-26

BOOSTER_DEPLOY_REPO="${BOOSTER_DEPLOY_REPO:-https://github.com/BoosterRobotics/booster_deploy.git}"
BOOSTER_DEPLOY_COMMIT="7bb1462e2ac742eba5c01659b815e2e99fbe617f"   # 2026-09-13

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/booster_paths.sh"

fetch() {  # fetch <dir> <repo> <commit>
  local name="$1" repo="$2" commit="$3" dir relative
  dir="$(booster_checkout_path "$REPO_ROOT" "$name")"
  relative="${dir#"$REPO_ROOT/"}"
  if [ -e "$dir/.git" ]; then
    local current
    current="$(git -C "$dir" rev-parse HEAD)"
    if [ "$current" = "$commit" ]; then
      echo "ok      $relative at pinned commit ${commit:0:7}"
    else
      echo "WARNING $relative is at ${current:0:7}, pinned is ${commit:0:7}."
      echo "        Left as is. To switch:"
      echo "        git -C \"$relative\" fetch origin $commit && git -C \"$relative\" checkout $commit"
    fi
  elif [ -e "$dir" ] || [ -L "$dir" ]; then
    echo "ERROR: $dir exists but is not a Git checkout; left untouched." >&2
    return 1
  else
    echo "clone   $relative @ ${commit:0:7}"
    git init -q "$dir"
    git -C "$dir" remote add origin "$repo"
    git -C "$dir" fetch -q --depth 1 origin "$commit"
    git -C "$dir" checkout -q FETCH_HEAD
  fi

  # uv's editable source is a fixed top-level path. Reuse nested checkouts via a
  # relative symlink instead of moving them or downloading another copy.
  if [ "$dir" != "$REPO_ROOT/$name" ]; then
    if [ -e "$REPO_ROOT/$name" ] || [ -L "$REPO_ROOT/$name" ]; then
      echo "ERROR: cannot link $name; that path already exists. Left untouched." >&2
      return 1
    fi
    ln -s "booster/$name" "$REPO_ROOT/$name"
    echo "link    $name -> booster/$name (existing checkout)"
  fi
}

fetch booster_assets "$BOOSTER_ASSETS_REPO" "$BOOSTER_ASSETS_COMMIT"
if [ "${1:-all}" != "assets" ]; then
  fetch booster_deploy "$BOOSTER_DEPLOY_REPO" "$BOOSTER_DEPLOY_COMMIT"
fi

echo "Done. Next, from the repository root:"
echo "  Linux/macOS local setup: uv sync --group sim --group train"
echo "  Linux GPU cluster:      uv sync --group train"
echo "See docs/commands.md for platform-specific viewer and training commands."
