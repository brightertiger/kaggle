#!/usr/bin/env python3

from setuptools import setup, find_packages
from pathlib import Path

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="talkingdata-fraud-detection",
    version="1.0.0",
    author="Ujjwal Sharma",
    author_email="ujjwalsharma@example.com",
    description="A comprehensive machine learning solution for detecting fraudulent ad clicks in mobile advertising",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ujjwal-sharma/talkingdata-fraud-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "lightgbm>=3.2.0",
        "pyarrow>=5.0.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
    ],
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
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
