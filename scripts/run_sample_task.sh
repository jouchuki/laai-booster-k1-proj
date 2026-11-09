#!/usr/bin/env bash
# Linux/macOS host: run Booster's pretrained MuJoCo task from any directory.
# Usage: bash scripts/run_sample_task.sh [task] [deploy.py options]
# Default task: k1_fight. See docs/commands.md for platform-specific commands.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/scripts/booster_paths.sh"
DEPLOY_DIR="$(booster_checkout_path "$REPO_ROOT" booster_deploy)"
TASK="${1:-k1_fight}"
if [ "$#" -gt 0 ]; then
    shift
fi

case "$(uname -s)" in
    Linux) PYTHON_CMD=python ;;
    Darwin) PYTHON_CMD=mjpython ;;
    *)
        echo "Unsupported platform: this viewer wrapper supports Linux and macOS." >&2
        exit 1
        ;;
esac

if [ ! -f "$DEPLOY_DIR/scripts/deploy.py" ]; then
    echo "Missing pinned booster_deploy checkout. From the repository root, run:" >&2
    echo "  bash scripts/setup_booster.sh" >&2
    exit 1
fi

cd "$REPO_ROOT"
if [ ! -d "$REPO_ROOT/booster_assets" ]; then
    echo "Missing top-level booster_assets dependency. From the repository root, run:" >&2
    echo "  bash scripts/setup_booster.sh assets" >&2
    exit 1
fi
cd "$DEPLOY_DIR"
exec uv run --group sim "$PYTHON_CMD" scripts/deploy.py --task "$TASK" --mujoco "$@"
