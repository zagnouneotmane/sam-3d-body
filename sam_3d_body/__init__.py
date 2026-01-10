# Copyright (c) Meta Platforms, Inc. and affiliates.
try:
    from ._version import version as __version__
except ImportError:
    # Fallback if _version.py doesn't exist (e.g., before installation)
    __version__ = "0.1.0"

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .build_models import load_sam_3d_body, load_sam_3d_body_hf

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
]
