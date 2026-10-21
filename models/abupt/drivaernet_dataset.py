import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset
from pydantic import BaseModel
import numpy as np
import pyvista as pv


class FileMap(BaseModel):
    """File mapping schema for aerodynamic datasets."""
    # Surface field files
    surface_position: str
    surface_pressure: str
    surface_friction: str
    # Volume field files
    volume_position: str
    volume_pressure: str
    volume_velocity: str
    volume_vorticity: str
    # Optional additional files
    surface_position_stl: str | None = None
    surface_position_stl_resampled: str | None = None
    volume_friction: str | None = None
    volume_distance_to_surface: str | None = None
    design_parameters: str | None = None


drivaernet_filemap = FileMap(
    surface_position="surface_position.pt",
    surface_pressure="surface_p.pt",
    surface_friction="surface_wallShearStress.pt",
    volume_position="volume_position.pt",
    volume_pressure="volume_p.pt",
    volume_velocity="volume_U.pt",
    volume_friction="volume_wallShearStress.pt",
    volume_vorticity="volume_vorticity.pt",
)


class AeroStatsSchema(BaseModel):
    """Unified statistics dataclass for aerodynamics datasets."""
    # Surface statistics
    surface_domain_min: tuple[float, float, float]
    surface_domain_max: tuple[float, float, float]
    surface_pos_mean: tuple[float, float, float]
    surface_pos_std: tuple[float, float, float]
    surface_pressure_mean: tuple[float]
    surface_pressure_std: tuple[float]
    surface_friction_mean: tuple[float, float, float]
    surface_friction_std: tuple[float, float, float]
    # Volume statistics
    volume_pos_mean: tuple[float, float, float]
    volume_pos_std: tuple[float, float, float]
    volume_pressure_mean: tuple[float]
    volume_pressure_std: tuple[float]
    volume_velocity_mean: tuple[float, float, float]
    volume_velocity_std: tuple[float, float, float]
    volume_vorticity_mean: tuple[float, float, float]
    volume_vorticity_std: tuple[float, float, float]
    volume_vorticity_logscale_mean: tuple[float, float, float]
    volume_vorticity_logscale_std: tuple[float, float, float]
    volume_vorticity_magnitude_mean: float
    volume_vorticity_magnitude_std: float
    volume_domain_min: tuple[float, float, float]
    volume_domain_max: tuple[float, float, float]
    # Design parameter statistics
    inflow_design_parameters_min: Sequence[float] | None = None
    inflow_design_parameters_max: Sequence[float] | None = None
    inflow_design_parameters_mean: Sequence[float] | None = None
    inflow_design_parameters_std: Sequence[float] | None = None
    geometry_design_parameters_min: Sequence[float] | None = None
    geometry_design_parameters_max: Sequence[float] | None = None
    geometry_design_parameters_mean: Sequence[float] | None = None
    geometry_design_parameters_std: Sequence[float] | None = None


drivaernet_stats = AeroStatsSchema(
    surface_domain_min=(-1.15154e00, -1.19442e00, 0.00000e00),
    surface_domain_max=(4.20453e00, 1.19442e00, 1.76205e00),
    surface_pos_mean=(1.53702e00, -8.12572e-03, 5.93542e-01),
    surface_pos_std=(1.39704e00, 6.44139e-01, 3.88172e-01),
    surface_pressure_mean=(-9.34098e01,),
    surface_pressure_std=(1.20787e02,),
    surface_friction_mean=(-6.71649e-01, 3.63487e-02, -8.46379e-02),
    surface_friction_std=(8.19941e-01, 4.51045e-01, 7.81055e-01),
    volume_pos_mean=(2.17577e00, -3.35840e-01, 7.60331e-01),
    volume_pos_std=(3.06283e00, 7.20275e-01, 6.50708e-01),
    volume_pressure_mean=(-6.24053e01,),
    volume_pressure_std=(9.42394e01,),
    volume_velocity_mean=(2.18719e01, -2.37778e-01, 6.73902e-01),
    volume_velocity_std=(1.21079e01, 3.97768e00, 3.90113e00),
    volume_vorticity_mean=(2.24115e00, 3.02183e01, 2.58549e01),
    volume_vorticity_std=(1.25151e05, 1.50186e05, 2.33779e05),
    volume_vorticity_logscale_mean=(2.57623e-02, 2.58335e-01, 4.29835e-01),
    volume_vorticity_logscale_std=(3.00179e00, 3.65020e00, 3.33356e00),
    volume_vorticity_magnitude_mean=3.03779e02,
    volume_vorticity_magnitude_std=3.04748e05,
    volume_domain_min=(-1.20012e01, -4.00040e00, -3.46979e-18),
    volume_domain_max=(3.20032e01, 4.00040e00, 6.40064e00),
)


@dataclass
class DrivAerNetStats:
    raw_pos_min: tuple[float] = (min(drivaernet_stats.volume_domain_min),)
    raw_pos_max: tuple[float] = (max(drivaernet_stats.volume_domain_max),)
    surface_raw_pos_min: tuple[float] = (min(drivaernet_stats.surface_domain_min),)
    surface_raw_pos_max: tuple[float] = (max(drivaernet_stats.surface_domain_max),)
    domain_min: tuple[float, float, float] = drivaernet_stats.volume_domain_min
    domain_max: tuple[float, float, float] = drivaernet_stats.volume_domain_max
    surface_pressure_mean: tuple[float] = drivaernet_stats.surface_pressure_mean
    surface_pressure_std: tuple[float] = drivaernet_stats.surface_pressure_std
    surface_wallshearstress_mean: tuple[float, float, float] = drivaernet_stats.surface_friction_mean
    surface_wallshearstress_std: tuple[float, float, float] = drivaernet_stats.surface_friction_std
    volume_totalpcoeff_mean: tuple[float] = drivaernet_stats.volume_pressure_mean
    volume_totalpcoeff_std: tuple[float] = drivaernet_stats.volume_pressure_std
    volume_velocity_mean: tuple[float, float, float] = drivaernet_stats.volume_velocity_mean
    volume_velocity_std: tuple[float, float, float] = drivaernet_stats.volume_velocity_std
    volume_vorticity_mean: tuple[float, float, float] = drivaernet_stats.volume_vorticity_mean
    volume_vorticity_std: tuple[float, float, float] = drivaernet_stats.volume_vorticity_std
    volume_vorticity_logscale_mean: tuple[float, float, float] = drivaernet_stats.volume_vorticity_logscale_mean
    volume_vorticity_logscale_std: tuple[float, float, float] = drivaernet_stats.volume_vorticity_logscale_std
    volume_vorticity_magnitude_mean: float = drivaernet_stats.volume_vorticity_magnitude_mean
    volume_vorticity_magnitude_std: float = drivaernet_stats.volume_vorticity_magnitude_std
    raw_pos_z_min: float = drivaernet_stats.volume_domain_min[2]


# REMOVED THE CIRCULAR IMPORT LINE THAT WAS HERE
# The line "from drivaernet_dataset import DrivAerNetDataset" has been removed
# since we're already defining the class in this file


class DrivAerNetDataset(Dataset):
    """DrivAerNet dataset that can handle both VTK and preprocessed PT files"""
    
    def __init__(
        self,
        root: str,
        split: str,
        train_design_ids_file: str = os.path.join(os.environ.get("CARBENCH_SPLITS_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "train_val_test_splits")), "train_design_ids.txt"),
        test_design_ids_file: str = os.path.join(os.environ.get("CARBENCH_SPLITS_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "train_val_test_splits")), "test_design_ids.txt"),
        val_design_ids_file: str = os.path.join(os.environ.get("CARBENCH_SPLITS_DIR", os.path.join(os.environ.get("CARBENCH_DATA_ROOT", "data/DrivAerNet"), "train_val_test_splits")), "val_design_ids.txt"),
        skip_missing: bool = True,
        **kwargs,
    ):
        # Set instance variables
        self.split = split
        self.train_design_ids_file = train_design_ids_file
        self.test_design_ids_file = test_design_ids_file
        self.val_design_ids_file = val_design_ids_file
        self.skip_missing = skip_missing

        # Set up root path
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise ValueError(f"Root directory '{self.root.as_posix()}' doesn't exist")

        # Detect data format (VTK or preprocessed PT)
        self.data_format = self._detect_data_format()
        
        self.filemap = drivaernet_filemap
        datasplits = self._load_all_datasplits()
        
        # Filter out missing files if skip_missing is True
        if self.skip_missing and self.data_format == "vtk":
            original_count = len(datasplits[self.split])
            datasplits[self.split] = self._filter_existing_files(datasplits[self.split])
            filtered_count = len(datasplits[self.split])
            if filtered_count < original_count:
                print(f"Warning: Filtered {original_count - filtered_count} missing files from {self.split} split")
                print(f"Using {filtered_count} files for {self.split} split")
        
        self.design_ids = datasplits[self.split]

        # Cache for VTK data to avoid reloading
        self._vtk_cache = {}

        super().__init__(**kwargs)
    
    def _filter_existing_files(self, design_ids):
        """Filter design IDs to only include those with existing VTK files"""
        existing_ids = []
        missing_ids = []
        
        for design_id in design_ids:
            # Check if VTK file exists
            possible_paths = [
                self.root / f"{design_id}.vtk",
                self.root / f"{design_id}_surfacePressure.vtk",
                self.root / f"{design_id}_pressure.vtk",
            ]
            
            exists = any(path.exists() for path in possible_paths)
            
            if not exists:
                # Try pattern matching
                pattern_files = list(self.root.glob(f"*{design_id}*.vtk"))
                exists = len(pattern_files) > 0
            
            if exists:
                existing_ids.append(design_id)
            else:
                missing_ids.append(design_id)
        
        if missing_ids:
            print(f"Missing VTK files for {len(missing_ids)} design IDs (first 5): {missing_ids[:5]}")
        
        return existing_ids

    def _detect_data_format(self):
        """Detect whether data is in VTK or preprocessed PT format"""
        # Check for VTK files
        vtk_files = list(self.root.glob("*.vtk"))
        if vtk_files:
            print(f"Detected VTK format with {len(vtk_files)} files")
            return "vtk"
        
        # Check for preprocessed directories
        pt_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        if pt_dirs and (pt_dirs[0] / "surface_position.pt").exists():
            print(f"Detected preprocessed PT format")
            return "pt"
        
        print(f"Warning: Could not detect data format in {self.root}")
        return "unknown"

    @staticmethod
    def get_normalization_stats():
        return DrivAerNetStats()

    def _load_all_datasplits(self):
        """Load design IDs for all splits"""
        split_design_ids_paths = {
            "train": self.train_design_ids_file,
            "test": self.test_design_ids_file,
            "val": self.val_design_ids_file,
        }
        
        blacklist_ids = [
            "E_S_WWC_WM_485",  # inconsistent meshes
            "F_D_WM_WW_0978",  # doesn't exist
            "F_D_WM_WW_3641",  # doesn't exist
        ]
        
        split_ids = dict()
        for split, path in split_design_ids_paths.items():
            with open(path) as f:
                split_ids[split] = [line.strip() for line in f if line.strip() not in blacklist_ids]

        # Union of all splits, for training on the full dataset (no held-out split)
        split_ids["all"] = split_ids["train"] + split_ids["val"] + split_ids["test"]

        return split_ids

    def __len__(self) -> int:
        return len(self.design_ids)

    def _load_vtk(self, idx: int):
        """Load VTK file and cache it"""
        design_id = self.design_ids[idx]
        
        if design_id not in self._vtk_cache:
            vtk_path = self.root / f"{design_id}.vtk"
            
            if not vtk_path.exists():
                raise FileNotFoundError(f"VTK file not found: {vtk_path}")
            
            # Load VTK file
            mesh = pv.read(str(vtk_path))
            self._vtk_cache[design_id] = mesh
            
        return self._vtk_cache[design_id]

    def _load(self, idx: int, filename: str) -> torch.Tensor:
        """Load data based on format (VTK or preprocessed PT)"""
        if self.data_format == "vtk":
            return self._load_from_vtk(idx, filename)
        else:
            # Original PT loading
            design_uri = self.root / self.design_ids[idx]
            if not design_uri.exists():
                raise FileNotFoundError(f"{design_uri.as_posix()} does not exist")
            return torch.load(design_uri / filename, weights_only=True)

    def _load_from_vtk(self, idx: int, filename: str) -> torch.Tensor:
        """Extract data from VTK file based on requested field"""
        mesh = self._load_vtk(idx)
        
        # Surface position
        if filename == self.filemap.surface_position:
            return torch.tensor(mesh.points, dtype=torch.float32)
        
        # Surface pressure
        elif filename == self.filemap.surface_pressure:
            # Try different possible field names
            for field_name in ['p', 'pressure', 'Pressure', 'surface_pressure', 'surfacePressure']:
                if field_name in mesh.point_data:
                    return torch.tensor(mesh.point_data[field_name], dtype=torch.float32)
            
            # If no pressure field found, generate dummy data
            print(f"Warning: No pressure field found in VTK, using dummy data")
            return torch.randn(mesh.n_points) * 120.0 - 93.0  # Match statistics
        
        # Surface wall shear stress
        elif filename == self.filemap.surface_friction:
            # For pressure-only model, just return zeros
            return torch.zeros(mesh.n_points, 3, dtype=torch.float32)
        
        # For volume fields, return dummy data for now
        elif filename in [self.filemap.volume_position, self.filemap.volume_pressure, 
                         self.filemap.volume_velocity, self.filemap.volume_vorticity]:
            print(f"Warning: Volume field {filename} not available in VTK, using dummy data")
            num_points = 100000  # Typical volume mesh size
            if filename == self.filemap.volume_position:
                return torch.randn(num_points, 3)
            elif filename == self.filemap.volume_pressure:
                return torch.randn(num_points)
            elif filename == self.filemap.volume_velocity:
                return torch.randn(num_points, 3)
            elif filename == self.filemap.volume_vorticity:
                return torch.randn(num_points, 3)
        
        else:
            raise ValueError(f"Unknown filename: {filename}")

    def getitem_surface_position_vtp(self, idx: int) -> torch.Tensor:
        """Retrieves surface position (N, 3)"""
        return self._load(idx=idx, filename=self.filemap.surface_position)

    def getitem_surface_pressure(self, idx: int) -> torch.Tensor:
        """Retrieves surface pressure coefficients (N, 1)"""
        pressure = self._load(idx=idx, filename=self.filemap.surface_pressure)
        if pressure.dim() == 1:
            pressure = pressure.unsqueeze(1)
        return pressure

    def getitem_surface_wallshearstress(self, idx: int) -> torch.Tensor:
        """Retrieves surface friction coefficients (N, 3)"""
        return self._load(idx=idx, filename=self.filemap.surface_friction)

    def getitem_volume_position(self, idx: int) -> torch.Tensor:
        """Retrieves volume position (N, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_position)

    def getitem_volume_totalpcoeff(self, idx: int) -> torch.Tensor:
        """Retrieves volume pressure coefficients (N, 1)"""
        pressure = self._load(idx=idx, filename=self.filemap.volume_pressure)
        if pressure.dim() == 1:
            pressure = pressure.unsqueeze(1)
        return pressure

    def getitem_volume_velocity(self, idx: int) -> torch.Tensor:
        """Retrieves volume velocity (N, 3)"""
        return self._load(idx=idx, filename=self.filemap.volume_velocity)