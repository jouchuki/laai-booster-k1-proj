"""Paths to models and project folders, so nothing hard-codes a teammate's home folder."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_DIR = REPO_ROOT / "assets"
BIKE_XML = ASSETS_DIR / "bike" / "balance_bike.xml"
SKETCHFAB_BIKE_XML = ASSETS_DIR / "bike" / "sketchfab_mujoco" / "bike.xml"
REAL_BIKE_XML = ASSETS_DIR / "bike" / "real_bike" / "bike.xml"


def booster_assets_dir() -> Path:
    """Folder of the cloned booster_assets repo (run scripts/setup_booster.sh first)."""
    from booster_assets import BOOSTER_ASSETS_DIR

    return Path(BOOSTER_ASSETS_DIR)


def k1_xml() -> Path:
    """MuJoCo model of the 22-DoF Booster K1."""
    return booster_assets_dir() / "robots" / "K1" / "K1_22dof.xml"
