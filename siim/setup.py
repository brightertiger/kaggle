from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).resolve().parent

with open(HERE / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(HERE / "requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="siim-melanoma-classifier",
    version="1.0.0",
    author="Ujjwal Singh Rao",
    description="A deep learning pipeline for SIIM-ISIC melanoma classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brightertiger/kaggle/tree/main/siim",
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
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
    },
    entry_points={
        "console_scripts": [
            "siim-train=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
