"""Reliable neural optimization proxies with exact repair and certificates."""

from reliable_proxy.domain import DispatchInstance
from reliable_proxy.generator import GeneratorConfig, generate_instance
from reliable_proxy.model import NumpyMLPProxy, TrainingConfig
from reliable_proxy.oracle import ExactSolution, solve_exact
from reliable_proxy.proxy import ProxySolveResult, ReliableOptimizationProxy

__all__ = [
    "DispatchInstance",
    "ExactSolution",
    "GeneratorConfig",
    "NumpyMLPProxy",
    "ProxySolveResult",
    "ReliableOptimizationProxy",
    "TrainingConfig",
    "generate_instance",
    "solve_exact",
]

__version__ = "0.1.0"
