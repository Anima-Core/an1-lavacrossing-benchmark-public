#!/usr/bin/env python3
"""
Setup script for One-Shot LavaCrossing Benchmark
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lavacrossing-benchmark",
    version="1.0.0",
    author="Anima-Core",
    description="A rigorous benchmark for testing one-shot catastrophe avoidance in AI agents",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Anima-Core/an1-lavacrossing-benchmark-public",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "lavacrossing-benchmark=an1_lavacrossing_oneshot_benchmark:main",
            "lavacrossing-test=test_lavacrossing_benchmark:main",
        ],
    },
)