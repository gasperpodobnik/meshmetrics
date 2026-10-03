"""PyPI package definition."""

from setuptools import setup

setup(
    name="MeshMetrics",
    version="0.2.2",
    description=(
        "Library to compute distance-based performance metrics for image segmentation tasks."
    ),
    url="https://github.com/gasperpodobnik/MeshMetrics",
    author="Gasper Podobnik",
    license="Apache License, Version 2.0",
    packages=["MeshMetrics"],
    install_requires=[
        "numpy",
        "SimpleITK",
        "SimpleITKUtilities",
        "vtk",
    ],
)
