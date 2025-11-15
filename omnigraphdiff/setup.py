"""
OmniGraphDiff Setup
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = [
    "torch>=2.0.0",
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "pyyaml>=6.0",
    "tqdm>=4.62.0",
    "lifelines>=0.27.0",  # Survival analysis
    "umap-learn>=0.5.0",
    "matplotlib>=3.4.0",
    "seaborn>=0.11.0",
    "tensorboard>=2.8.0",
    "h5py>=3.6.0",
]

setup(
    name="omnigraphdiff",
    version="0.1.0",
    author="OmniGraphDiff Team",
    author_email="",
    description="Hierarchical Graph-Driven Generative Multi-Omics Integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dxpython/OmniOmicsR/tree/main/omnigraphdiff",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
