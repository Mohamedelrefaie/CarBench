from torch.utils.data import Dataset

from collators import ConcatSparseTensorCollator, FieldDecoderCollator, MultiStageCollator
from preprocessors import (
    MomentNormalizationPreprocessor,
    PointSamplingPreprocessor,
    PositionNormalizationPreprocessor,
    ReplaceKeyPreprocessor,
    SupernodeSamplingPreprocessor,
    AnchorPointSamplingPreprocessor,
)


class SurfaceOnlyAbuptCollator(MultiStageCollator):
    def __init__(
        self,
        num_geometry_points: int,
        num_surface_anchor_points: int,
        num_geometry_supernodes: int,
        dataset: Dataset,
        use_query_positions: bool = False,
        seed: int | None = None,
    ):
        stats = dataset.get_normalization_stats()
        super().__init__(
            preprocessors=[
                # normalize positions (these are the stats for the surface positions)
                PositionNormalizationPreprocessor(
                    items={"surface_position_vtp"},
                    raw_pos_min=stats.surface_raw_pos_min,
                    raw_pos_max=stats.surface_raw_pos_max,
                    scale=1000,
                ),
                # normalize surface pressures
                MomentNormalizationPreprocessor(
                    item="surface_pressure",
                    mean=stats.surface_pressure_mean,
                    std=stats.surface_pressure_std,
                ),
                # normalize surface wallshearstress
                MomentNormalizationPreprocessor(
                    item="surface_wallshearstress",
                    mean=stats.surface_wallshearstress_mean,
                    std=stats.surface_wallshearstress_std,
                ),
                # duplicate surface positions for geometry/surface
                ReplaceKeyPreprocessor(
                    source_key="surface_position_vtp",
                    target_keys={"geometry_position", "surface_position"},
                ),
                # preprocess geometry data
                PointSamplingPreprocessor(
                    items={"geometry_position"},
                    num_points=num_geometry_points,
                    seed=None if seed is None else seed + 1,
                ),
                SupernodeSamplingPreprocessor(
                    item="geometry_position",
                    num_supernodes=num_geometry_supernodes,
                    supernode_idx_key="geometry_supernode_idx",
                    seed=None if seed is None else seed + 2,
                ),
                # subsample surface data
                AnchorPointSamplingPreprocessor(
                    items={"surface_position", "surface_pressure", "surface_wallshearstress"},
                    num_points=num_surface_anchor_points,
                    keep_queries=use_query_positions,
                    to_prefix_and_postfix=lambda item: item.split("_"),
                    to_prefix_midfix_postfix=lambda item: item.split("_") if len(item.split("_")) == 3 else [None] * 3,
                    seed=None if seed is None else seed + 3,
                ),
            ],
            collators=[
                # collate geometry positions (remains sparse for supernode_pooling)
                ConcatSparseTensorCollator(
                    items=["geometry_position"],
                    create_batch_idx=True,
                    batch_idx_key="geometry_batch_idx",
                ),
                ConcatSparseTensorCollator(items=["geometry_supernode_idx"], create_batch_idx=False),
                # collate surface data
                FieldDecoderCollator(
                    position_item="surface_anchor_position",
                    target_items=["surface_anchor_pressure", "surface_anchor_wallshearstress"],
                ),
                # collate auxiliary data
                FieldDecoderCollator(
                    position_item="surface_query_position",
                    target_items=["surface_query_pressure", "surface_query_wallshearstress"],
                    optional=True,
                ),
            ],
            postprocessors=[],
            dataset=dataset,
        )

    def preprocess_inputs_only(self, samples):
        batch = self(samples)
        for key in list(batch.keys()):
            if "pressure" in key or "wallshear" in key or "totalpcoeff" in key or "velocity" in key or "vorticity" in key:
                batch.pop(key)
        return batch
