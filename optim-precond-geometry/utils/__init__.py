# Utils package for optimization geometry analysis
from .hvp import power_iteration, compute_hvp_reverse_over_reverse
from .metrics import compute_cosine_similarity
from .logging import CSVLogger, WandbLogger
from .geometry import (
    OptimizationGeometryTracker,
    GradientSubspaceTracker,
    GradientSimilarityTracker,
    lanczos_iteration,
    compute_layer_gradient_norms,
    compute_layer_effective_ranks,
    compute_effective_rank,
    compute_spectral_density,
    categorize_layers
)

__all__ = [
    'power_iteration',
    'compute_hvp_reverse_over_reverse',
    'compute_cosine_similarity',
    'CSVLogger',
    'WandbLogger',
    'OptimizationGeometryTracker',
    'GradientSubspaceTracker',
    'GradientSimilarityTracker',
    'lanczos_iteration',
    'compute_layer_gradient_norms',
    'compute_layer_effective_ranks',
    'compute_effective_rank',
    'compute_spectral_density',
    'categorize_layers'
]
