"""Paths to models and project folders, so nothing hard-codes a teammate's home folder."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"

# Bike models: models/<name>/bike.xml. Any bike with the standard part names (frame, fork,
# saddle, seat, left_grip, right_grip, steer, root) works in every scene.
# Textured scan of the real bike, calibrated to its measurements; see its README.
PUKY_BIKE_XML = MODELS_DIR / "puky_bike" / "bike.xml"
# The earlier bike (green in the viewer), from an internet USDZ model (56.5 cm wheelbase).
YELLOW_BIKE_XML = MODELS_DIR / "yellow_bike" / "bike.xml"
# The first placeholder bike made of simple shapes, kept for comparison.
PRIMITIVE_BIKE_XML = MODELS_DIR / "primitive_bike" / "bike.xml"
BIKES = {
    "puky_bike": PUKY_BIKE_XML,
    "yellow_bike": YELLOW_BIKE_XML,
    "primitive_bike": PRIMITIVE_BIKE_XML,
}
# The bike we use in real life: the default for every scene, the viewer and the seated K1.
DEFAULT_BIKE = "puky_bike"
DEFAULT_BIKE_XML = BIKES[DEFAULT_BIKE]

# Solved seated poses, one per bike: models/k1_on_bike/<bike name>.json
# (written by scripts/solve_seated_pose.py --bike <bike name>).
SEATED_POSES_DIR = MODELS_DIR / "k1_on_bike"
# Raw training output (gitignored): logs/<experiment>/<date>_<run-name>/.
LOGS_DIR = REPO_ROOT / "logs"


def bike_name(bike_xml: Path) -> str:
    """Name of a bike model: its folder under models/ (models/puky_bike/bike.xml -> puky_bike)."""
    return Path(bike_xml).resolve().parent.name


def seated_pose_json(bike_xml: Path) -> Path:
    """File holding the K1's solved seated pose on this bike."""
    return SEATED_POSES_DIR / f"{bike_name(bike_xml)}.json"


def booster_assets_dir() -> Path:
    """Folder of the cloned booster_assets repo (run scripts/setup_booster.sh first)."""
    from booster_assets import BOOSTER_ASSETS_DIR

    return Path(BOOSTER_ASSETS_DIR)


def k1_xml() -> Path:
    """MuJoCo model of the 22-DoF Booster K1 (see models/k1/README.md)."""
    return booster_assets_dir() / "robots" / "K1" / "K1_22dof.xml"
