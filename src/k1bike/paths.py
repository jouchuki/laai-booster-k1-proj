"""Paths to models and project folders, so nothing hard-codes a teammate's home folder."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
# The balance bike we use (USDZ-derived, measured wheelbase); see models/bike/README.md.
BIKE_XML = MODELS_DIR / "bike" / "bike.xml"
# The first placeholder bike made of simple shapes, kept for comparison.
PRIMITIVE_BIKE_XML = MODELS_DIR / "bike_primitive" / "bike.xml"
# The K1's solved seated pose on BIKE_XML (written by scripts/solve_seated_pose.py).
SEATED_POSE_JSON = MODELS_DIR / "k1_on_bike" / "seated_pose.json"
# Raw training output (gitignored): logs/<experiment>/<date>_<run-name>/.
LOGS_DIR = REPO_ROOT / "logs"


def booster_assets_dir() -> Path:
    """Folder of the cloned booster_assets repo (run scripts/setup_booster.sh first)."""
    from booster_assets import BOOSTER_ASSETS_DIR

    return Path(BOOSTER_ASSETS_DIR)


def k1_xml() -> Path:
    """MuJoCo model of the 22-DoF Booster K1 (see models/k1/README.md)."""
    return booster_assets_dir() / "robots" / "K1" / "K1_22dof.xml"
