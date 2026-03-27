"""
Public API for the methods package.

    from methods import METHOD_REGISTRY, ExperimentConfig, BaseMethod

Concrete classes (SpectralMethod, GCN, GAT, SGC) are not re-exported here;
access them via METHOD_REGISTRY or import directly from their submodules.
"""

from methods.base import BaseMethod, ExperimentConfig
from methods.registry import METHOD_REGISTRY

__all__ = [
    "BaseMethod",
    "ExperimentConfig",
    "METHOD_REGISTRY",
]
