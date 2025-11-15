"""
OmniGraphDiff - Hierarchical Graph-Driven Generative Multi-Omics Integration

A production-grade framework for multi-omics data integration using graph neural networks
and deep generative models (VAE and Diffusion).
"""

__version__ = "0.1.0"
__author__ = "OmniGraphDiff Team"
__license__ = "MIT"

# Import C++ backend
try:
    import omnigraph_cpp
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False
    import warnings
    warnings.warn(
        "C++ backend (omnigraph_cpp) not available. "
        "Some operations will fall back to slower Python implementations. "
        "To build the C++ backend, run: cd cpp_backend && mkdir build && "
        "cd build && cmake .. && make && cd ../.."
    )

# Core modules
from . import models
from . import losses
from . import data
from . import training
from . import evaluation
from . import visualization
from . import utils

# Convenient imports
from .models import OmniGraphVAE, OmniGraphDiffusion
from .data import MultiOmicsDataset, build_graphs
from .training import Trainer
from .utils import set_seed, load_config

__all__ = [
    "__version__",
    "BACKEND_AVAILABLE",
    # Models
    "OmniGraphVAE",
    "OmniGraphDiffusion",
    # Data
    "MultiOmicsDataset",
    "build_graphs",
    # Training
    "Trainer",
    # Utils
    "set_seed",
    "load_config",
    # Submodules
    "models",
    "losses",
    "data",
    "training",
    "evaluation",
    "visualization",
    "utils",
]


def get_build_info():
    """Get build and environment information."""
    import torch
    import sys

    info = {
        "omnigraphdiff_version": __version__,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpp_backend_available": BACKEND_AVAILABLE,
    }

    if BACKEND_AVAILABLE:
        info["cpp_build_info"] = omnigraph_cpp.get_build_info()

    return info


def print_system_info():
    """Print system and build information."""
    import torch

    print("=" * 60)
    print("OmniGraphDiff System Information")
    print("=" * 60)
    print(f"OmniGraphDiff version: {__version__}")
    print(f"Python version: {get_build_info()['python_version'].split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"C++ backend available: {BACKEND_AVAILABLE}")

    if BACKEND_AVAILABLE:
        cpp_info = omnigraph_cpp.get_build_info()
        print("\nC++ Backend Information:")
        for key, value in cpp_info.items():
            print(f"  {key}: {value}")

    print("=" * 60)
