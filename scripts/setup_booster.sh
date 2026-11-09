#!/usr/bin/env bash
# Clone Booster's repos into booster/ at pinned commits, so everyone on the team
# (and CI, and the cluster) uses exactly the same robot model and deploy code.
#
# Usage (from the repo root):
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

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/booster"

fetch() {  # fetch <dir> <repo> <commit>
  local dir="$ROOT/booster/$1" repo="$2" commit="$3"
  if [ -d "$dir/.git" ]; then
    local current
    current="$(git -C "$dir" rev-parse HEAD)"
    if [ "$current" = "$commit" ]; then
      echo "ok      booster/$1 at pinned commit ${commit:0:7}"
    else
      echo "WARNING booster/$1 is at ${current:0:7}, pinned is ${commit:0:7}."
      echo "        Left as is. To switch: git -C booster/$1 fetch origin $commit && git -C booster/$1 checkout $commit"
    fi
    return
  fi
  echo "clone   booster/$1 @ ${commit:0:7}"
  git init -q "$dir"
  git -C "$dir" remote add origin "$repo"
  git -C "$dir" fetch -q --depth 1 origin "$commit"
  git -C "$dir" checkout -q FETCH_HEAD
}

fetch booster_assets "$BOOSTER_ASSETS_REPO" "$BOOSTER_ASSETS_COMMIT"
if [ "${1:-all}" != "assets" ]; then
  fetch booster_deploy "$BOOSTER_DEPLOY_REPO" "$BOOSTER_DEPLOY_COMMIT"
fi

echo "Done. Next: uv sync --group sim   (Mac)   or   uv sync --group train   (cluster)"
