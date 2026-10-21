"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Data loading utilities for the Transolver model on DrivAerNet++ dataset.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import pyvista as pv
from sklearn.neighbors import NearestNeighbors
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_interior_points(surface_mesh, n_points):
    """Generate interior points for SDF computation from surface mesh"""
    bounds = surface_mesh.bounds
    x_min, x_max = bounds[0], bounds[1]
    y_min, y_max = bounds[2], bounds[3] 
    z_min, z_max = bounds[4], bounds[5]
    
    # Expand bounds slightly
    margin = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range
    z_min -= margin * z_range
    z_max += margin * z_range
    
    # Generate random points in expanded bounding box
    points = np.random.uniform(
        low=[x_min, y_min, z_min],
        high=[x_max, y_max, z_max],
        size=(n_points, 3)
    ).astype(np.float32)
    
    return points


def compute_sdf_polydata(query_points, surface_mesh):
    """Compute SDF for query points given a surface mesh"""
    
    # Get surface points and compute normals
    surface_points = surface_mesh.points.astype(np.float64)
    
    try:
        surface_with_normals = surface_mesh.compute_normals(
            point_normals=True,
            auto_orient_normals=True,
            consistent_normals=True
        )
        surface_normals = surface_with_normals.point_data['Normals']
    except:
        # Fallback normal estimation
        surface_normals = np.zeros_like(surface_points)
        surface_normals[:, 2] = 1.0  # Z-up default
    
    # Find nearest surface points
    nbrs = NearestNeighbors(n_neighbors=1).fit(surface_points)
    distances, indices = nbrs.kneighbors(query_points.astype(np.float64))
    
    nearest_points = surface_points[indices.flatten()]
    nearest_normals = surface_normals[indices.flatten()]
    
    # Compute direction vectors
    direction_vectors = query_points.astype(np.float64) - nearest_points
    direction_norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    direction_vectors_normalized = direction_vectors / (direction_norms + 1e-12)
    
    # Determine sign using dot product with normals
    dot_products = np.sum(direction_vectors_normalized * nearest_normals, axis=1)
    signed_distances = distances.flatten() * np.sign(dot_products)
    
    # Points very close to surface get zero distance
    on_surface_mask = distances.flatten() < 1e-6
    signed_distances[on_surface_mask] = 0.0
    
    return signed_distances.astype(np.float32), nearest_normals.astype(np.float32)


class TransolverDrivAerNetDataset(Dataset):
    """Dataset for PolyData surface meshes"""
    
    def __init__(self, root_dir: str, num_points: int, cache_dir=None, mode='train', use_sdf=False):
        self.root_dir = root_dir
        self.vtk_files = sorted([os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.vtk')])
        self.num_points = num_points
        self.mode = mode  # 'train' for mixed sampling, 'eval' for surface-only
        self.use_sdf = use_sdf  # Flag to control SDF computation and interior points
        self.cache_dir = cache_dir if cache_dir else os.path.join(root_dir, "transolver_polydata_cache")
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        logger.info(f"PolyData dataset: {len(self.vtk_files)} files, {num_points} points per sample, mode: {mode}, use_sdf: {use_sdf}")
    
    def __len__(self):
        return len(self.vtk_files)
    
    def _get_cache_path(self, vtk_file_path):
        sdf_suffix = '_sdf' if self.use_sdf else '_nosdf'
        base_name = os.path.basename(vtk_file_path).replace('.vtk', f'_poly_n{self.num_points}_{self.mode}{sdf_suffix}.pt')
        return os.path.join(self.cache_dir, base_name)
    
    def __getitem__(self, idx):
        vtk_file_path = self.vtk_files[idx]
        filename = os.path.basename(vtk_file_path)
        cache_path = self._get_cache_path(vtk_file_path)
        
        if os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, map_location='cpu')
                return data['features'], data['pressure']
            except Exception as e:
                logger.warning(f"Cache failed for {filename}: {e}")
        
        try:
            mesh = pv.read(vtk_file_path)
            
            # Check if pressure data exists
            if 'p' not in mesh.point_data:
                available_data = list(mesh.point_data.keys())
                # Try common pressure field names
                pressure_candidates = ['pressure', 'P', 'Pressure', 'p_static', 'static_pressure']
                pressure_key = None
                for candidate in pressure_candidates:
                    if candidate in mesh.point_data:
                        pressure_key = candidate
                        break
                
                if pressure_key is None:
                    logger.error(f"No pressure data in {filename}. Available: {available_data}")
                    return torch.zeros(self.num_points, 7), torch.zeros(self.num_points)
            else:
                pressure_key = 'p'
            
            # Sample surface points
            surface_points = mesh.points.astype(np.float32)
            surface_pressure = mesh.point_data[pressure_key].astype(np.float32)
            
            # When use_sdf is False, only use surface points (no interior points)
            if not self.use_sdf:
                # Surface-only mode: sample or pad from surface points only
                if len(surface_points) < self.num_points:
                    # Pad with repetition if not enough points
                    np.random.seed(idx)
                    n_needed = self.num_points - len(surface_points)
                    repeat_indices = np.random.choice(len(surface_points), n_needed, replace=True)
                    points = np.vstack([surface_points, surface_points[repeat_indices]])
                    pressure = np.concatenate([surface_pressure, surface_pressure[repeat_indices]])
                else:
                    # Sample from surface points
                    np.random.seed(idx)
                    surface_indices = np.random.choice(len(surface_points), self.num_points, replace=False)
                    points = surface_points[surface_indices]
                    pressure = surface_pressure[surface_indices]
                
                # Compute surface normals (no interior points, no SDF)
                try:
                    surface_with_normals = mesh.compute_normals(
                        point_normals=True,
                        auto_orient_normals=True,
                        consistent_normals=True
                    )
                    all_normals = surface_with_normals.point_data['Normals'].astype(np.float32)
                    
                    # Get normals for the sampled/padded points
                    if len(surface_points) < self.num_points:
                        # Use same indices for normals as we did for points
                        normals = np.vstack([all_normals, all_normals[repeat_indices]])
                    else:
                        normals = all_normals[surface_indices]
                except:
                    # Fallback if normal computation fails
                    logger.warning(f"Failed to compute normals for {filename}, using z-up default")
                    normals = np.zeros((len(points), 3), dtype=np.float32)
                    normals[:, 2] = 1.0  # Z-up default
                
                # Features are [x, y, z, nx, ny, nz] (6D) - surface geometry with orientation
                features = np.concatenate([points, normals], axis=1).astype(np.float32)
                
            elif len(surface_points) < self.num_points:
                # Not enough surface points, pad with zeros or generate interior based on mode
                points = np.zeros((self.num_points, 3), dtype=np.float32)
                pressure = np.zeros(self.num_points, dtype=np.float32)
                points[:len(surface_points)] = surface_points
                pressure[:len(surface_points)] = surface_pressure
                
                if self.mode == 'train':
                    # Generate interior points for remaining slots in training mode
                    n_interior = self.num_points - len(surface_points)
                    if n_interior > 0:
                        np.random.seed(idx)
                        interior_points = generate_interior_points(mesh, n_interior)
                        points[len(surface_points):] = interior_points
                        # Interior pressure assumed to be mean of surface pressure
                        pressure[len(surface_points):] = surface_pressure.mean()
                else:
                    # In evaluation mode, only use surface points - pad with repetition if needed
                    n_needed = self.num_points - len(surface_points)
                    if n_needed > 0:
                        np.random.seed(idx)
                        repeat_indices = np.random.choice(len(surface_points), n_needed, replace=True)
                        points[len(surface_points):] = surface_points[repeat_indices]
                        pressure[len(surface_points):] = surface_pressure[repeat_indices]
                
                n_surface_used = len(surface_points)
                
            else:
                # Enough surface points available
                if self.mode == 'train':
                    # Balanced training mode: 10k surface + 10k interior for balanced SDF diversity
                    # This provides balanced geometric information via SDF
                    n_surface_sample = self.num_points // 2  # 10k surface points (half)
                    n_interior_sample = self.num_points - n_surface_sample  # 10k interior points (half)
                    
                    np.random.seed(idx)
                    
                    # Sample surface points
                    surface_indices = np.random.choice(len(surface_points), n_surface_sample, replace=False)
                    sampled_surface_points = surface_points[surface_indices]
                    sampled_surface_pressure = surface_pressure[surface_indices]
                    
                    # Generate interior points
                    interior_points = generate_interior_points(mesh, n_interior_sample)
                    interior_pressure = np.full(n_interior_sample, surface_pressure.mean(), dtype=np.float32)
                    
                    # Combine
                    points = np.vstack([sampled_surface_points, interior_points])
                    pressure = np.concatenate([sampled_surface_pressure, interior_pressure])
                    
                    n_surface_used = n_surface_sample
                else:
                    # Evaluation mode: sample only surface points
                    np.random.seed(idx)
                    surface_indices = np.random.choice(len(surface_points), self.num_points, replace=False)
                    points = surface_points[surface_indices]
                    pressure = surface_pressure[surface_indices]
                    
                    n_surface_used = self.num_points
            
                # Compute SDF and normals only when use_sdf is True
                sdf, normals = compute_sdf_polydata(points, mesh)
                
                # Create features [x, y, z, sdf, nx, ny, nz]
                features = np.concatenate([
                    points,  # [N, 3]
                    sdf.reshape(-1, 1),  # [N, 1]
                    normals  # [N, 3]
                ], axis=1)  # [N, 7]
            
            features_tensor = torch.tensor(features, dtype=torch.float32)
            pressure_tensor = torch.tensor(pressure, dtype=torch.float32)
            
            # ================================================================
            # RETURN RAW PRESSURE VALUES - normalization handled by trainer
            # ================================================================
            # pressure_tensor = (pressure_tensor - PRESSURE_MEAN) / PRESSURE_STD  # Moved to trainer
            
            # Save to cache
            torch.save({
                'features': features_tensor,
                'pressure': pressure_tensor  # Raw pressure - trainer will normalize
            }, cache_path)
            
            if self.use_sdf:
                logger.info(f"Processed {filename}: SDF[{sdf.min():.6f}, {sdf.max():.6f}], "
                           f"pressure[{pressure.min():.1f}, {pressure.max():.1f}] Pa -> "
                           f"normalized[{pressure_tensor.min():.3f}, {pressure_tensor.max():.3f}], surface pts: {n_surface_used}")
            else:
                logger.info(f"Processed {filename}: Surface+Normals mode (6D), "
                           f"pressure[{pressure.min():.1f}, {pressure.max():.1f}] Pa -> "
                           f"normalized[{pressure_tensor.min():.3f}, {pressure_tensor.max():.3f}], points: {len(points)}")
            
            return features_tensor, pressure_tensor
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            feature_dim = 6 if not self.use_sdf else 7  # 6D [x,y,z,nx,ny,nz] or 7D [x,y,z,sdf,nx,ny,nz]
            return torch.zeros(self.num_points, feature_dim), torch.zeros(self.num_points)


def create_subset(dataset, ids_file):
    """Create subset based on design IDs"""
    if not os.path.exists(ids_file):
        raise FileNotFoundError(f"IDs file not found: {ids_file}")
    
    with open(ids_file, 'r') as file:
        subset_ids = [id_.strip() for id_ in file.readlines() if id_.strip()]
    
    subset_indices = []
    for i, vtk_file in enumerate(dataset.vtk_files):
        file_basename = os.path.basename(vtk_file)
        if any(id_ in file_basename for id_ in subset_ids):
            subset_indices.append(i)
    
    logger.info(f"Subset: {len(subset_indices)} samples")
    return Subset(dataset, subset_indices)


def get_dataloaders(dataset_path: str, subset_dir: str, num_points: int, 
                    batch_size: int, world_size: int = 1, rank: int = 0,
                    cache_dir: str = None, num_workers: int = 4, mode: str = 'train',
                    use_sdf: bool = False):
    """
    Get dataloaders for PolyData meshes
    
    Args:
        use_sdf: If True, use 7D features [x,y,z,sdf,nx,ny,nz] with interior points.
                 If False (default), use 6D features [x,y,z,nx,ny,nz] with surface points only.
    """
    
    # For training and validation, use mixed sampling only if use_sdf is True
    # For testing/evaluation, use the specified mode
    train_mode = 'train' if use_sdf else 'eval'
    val_mode = 'train' if use_sdf else 'eval'
    test_mode = mode  # This allows surface-only evaluation while keeping mixed training
    
    train_dataset = TransolverDrivAerNetDataset(
        root_dir=dataset_path,
        num_points=num_points,
        cache_dir=cache_dir,
        mode=train_mode,
        use_sdf=use_sdf
    )
    
    val_dataset = TransolverDrivAerNetDataset(
        root_dir=dataset_path,
        num_points=num_points,
        cache_dir=cache_dir,
        mode=val_mode,
        use_sdf=use_sdf
    )
    
    test_dataset = TransolverDrivAerNetDataset(
        root_dir=dataset_path,
        num_points=num_points,
        cache_dir=cache_dir,
        mode=test_mode,
        use_sdf=use_sdf
    )
    
    train_subset = create_subset(train_dataset, os.path.join(subset_dir, 'train_design_ids.txt'))
    val_subset = create_subset(val_dataset, os.path.join(subset_dir, 'val_design_ids.txt'))
    test_subset = create_subset(test_dataset, os.path.join(subset_dir, 'test_design_ids.txt'))
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_subset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_subset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_subset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=shuffle_train,
                             sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                           sampler=val_sampler, num_workers=num_workers, pin_memory=True)
    
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False,
                            sampler=test_sampler, num_workers=num_workers, pin_memory=True)
    
    feature_info = "7D [x,y,z,sdf,nx,ny,nz] with interior points" if use_sdf else "6D [x,y,z,nx,ny,nz] surface+normals"
    logger.info(f"DataLoaders: Train {len(train_subset)} (mode: {train_mode}), Val {len(val_subset)} (mode: {val_mode}), Test {len(test_subset)} (mode: {test_mode})")
    logger.info(f"Feature mode: {feature_info}")
    
    return train_loader, val_loader, test_loader


# Normalization constants
PRESSURE_MEAN = -94.5
PRESSURE_STD = 117.25