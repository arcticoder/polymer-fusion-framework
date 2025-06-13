"""
Polymer Fusion Framework Setup
===============================

Setup script for the comprehensive polymer-enhanced fusion research framework.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Polymer Fusion Framework - A comprehensive framework for polymer-enhanced fusion research"

# Read requirements
def read_requirements():
    try:
        with open("polymer-induced-fusion/requirements.txt", "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "numpy>=1.21.0",
            "scipy>=1.7.0", 
            "matplotlib>=3.4.0",
            "pandas>=1.3.0",
            "sympy>=1.8",
            "jupyter>=1.0.0"
        ]

setup(
    name="polymer-fusion-framework",
    version="1.0.0",
    author="Polymer Fusion Research Team",
    author_email="",
    description="Comprehensive framework for polymer-enhanced fusion research",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/polymer-fusion-framework/polymer-fusion-framework",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0",
            "ipywidgets>=7.6",
        ],
        "hts": [
            "scipy>=1.7.0",
            "matplotlib>=3.4.0",
            "numpy>=1.21.0",
        ],
        "latex": [
            "pylatex>=1.4.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "polymer-fusion=polymer_induced_fusion.main:main",
            "hts-analyze=polymer_induced_fusion.hts_materials_simulation:main",
            "reactor-design=polymer_induced_fusion.plan_a_step5_reactor_design:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "*.md",
            "*.txt", 
            "*.json",
            "*.png",
            "*.pdf",
            "*.tex",
            "*.bib",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/polymer-fusion-framework/polymer-fusion-framework/issues",
        "Source": "https://github.com/polymer-fusion-framework/polymer-fusion-framework",
        "Documentation": "https://polymer-fusion-framework.readthedocs.io/",
    },
)
