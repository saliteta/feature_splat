import warnings

from .cuda._torch_impl import accumulate
from .cuda._wrapper import (
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    persp_proj,
    quat_scale_to_covar_preci,
    rasterize_to_indices_in_range,
    rasterize_to_pixels,
    spherical_harmonics,
    world_to_cam,
    _RasterizeToPixels
)
from .rendering import (
    rasterization,
    rasterization_inria_wrapper,
    rasterization_legacy_wrapper,
)
from .version import __version__



all = [
    "rasterization",
    "rasterization_legacy_wrapper",
    "rasterization_inria_wrapper",
    "spherical_harmonics",
    "isect_offset_encode",
    "isect_tiles",
    "persp_proj",
    "fully_fused_projection",
    "quat_scale_to_covar_preci",
    "rasterize_to_pixels",
    "world_to_cam",
    "accumulate",
    "rasterize_to_indices_in_range",
    "__version__",
    # deprecated
    "rasterize_gaussians",
    "project_gaussians",
    "map_gaussian_to_intersects",
    "bin_and_sort_gaussians",
    "compute_cumulative_intersects",
    "compute_cov2d_bounds",
    "get_tile_bin_edges",
]
