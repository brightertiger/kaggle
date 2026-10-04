#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="talkingdata-fraud-detection",
    version="1.0.0",
    author="Ujjwal Singh Rao",
    description="A comprehensive machine learning solution for detecting fraudulent ad clicks in mobile advertising",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brightertiger/kaggle/tree/main/talking",
    packages=find_packages(),
    py_modules=["main"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.11",
    install_requires=[line.strip() for line in
                      (this_directory / "requirements.txt").read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "talkingdata-fraud=main:main",
        ],
    },
)
