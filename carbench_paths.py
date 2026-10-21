"""Filesystem locations for the DrivAerNet++ data used by CarBench.

Nothing here is hard-coded to one machine: every location resolves from an
environment variable, falling back to ``./data/DrivAerNet`` relative to the
repository root. Set the data root once, after downloading the dataset from the
Harvard Dataverse:

    export CARBENCH_DATA_ROOT=/path/to/DrivAerNet

Individual locations can be overridden if your layout differs:

    CARBENCH_SURFACE_DIR   directory of per-design surface VTK files
    CARBENCH_SPLITS_DIR    directory holding {train,val,test}_design_ids.txt
    CARBENCH_PRESSURE_STATS  .npz of training-split pressure mean/std
    CARBENCH_ABUPT_STAR_CHECKPOINT  AB-UPT* checkpoint from the original
        authors; not redistributed with CarBench (see checkpoints/README.md)
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))

DATA_ROOT = os.environ.get("CARBENCH_DATA_ROOT", os.path.join(_HERE, "data", "DrivAerNet"))
SURFACE_DIR = os.environ.get("CARBENCH_SURFACE_DIR", os.path.join(DATA_ROOT, "PressureVTK", "Dataverse"))
SPLITS_DIR = os.environ.get("CARBENCH_SPLITS_DIR", os.path.join(DATA_ROOT, "train_val_test_splits"))
TRAIN_IDS = os.path.join(SPLITS_DIR, "train_design_ids.txt")
VAL_IDS = os.path.join(SPLITS_DIR, "val_design_ids.txt")
TEST_IDS = os.path.join(SPLITS_DIR, "test_design_ids.txt")
PRESSURE_STATS = os.environ.get("CARBENCH_PRESSURE_STATS", os.path.join(DATA_ROOT, "pressure_stats.npz"))
ABUPT_STAR_CHECKPOINT = os.environ.get("CARBENCH_ABUPT_STAR_CHECKPOINT", "")


def require(path, what):
    """Fail with an actionable message instead of a bare FileNotFoundError."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{what} not found at {path}. Download DrivAerNet++ from the Harvard "
            f"Dataverse and set CARBENCH_DATA_ROOT; see README.md > Dataset.")
    return path
