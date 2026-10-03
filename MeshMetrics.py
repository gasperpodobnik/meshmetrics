"""Deprecated alias for the ``meshmetrics`` package.

Kept as a single module (not a package directory) so it can coexist with
``meshmetrics/`` on case-insensitive filesystems (Windows, macOS).
"""

import sys
import warnings

import meshmetrics
from meshmetrics.metrics import *  # noqa: F401,F403  (names the old package exposed)
from meshmetrics import *  # noqa: F401,F403
from meshmetrics import metrics, utils

warnings.warn(
    "`import MeshMetrics` is deprecated, use `import meshmetrics` instead.",
    DeprecationWarning,
    stacklevel=2,
)

# keep `from MeshMetrics.utils import ...` / `MeshMetrics.metrics` working
sys.modules[__name__ + ".metrics"] = metrics
sys.modules[__name__ + ".utils"] = utils
