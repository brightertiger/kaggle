from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with open(ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(ROOT / "requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="whale-identification",
    version="1.0.0",
    author="Ujjwal Rao",
    description="Advanced whale identification using deep learning and metric learning techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brightertiger/kaggle/tree/main/whale",
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
        "Topic :: Scientific/Engineering :: Image Recognition",
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
        "jupyter": [
            "jupyter>=1.0",
            "ipywidgets>=7.6",
        ],
    },
    entry_points={
        "console_scripts": [
            "whale-train=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
)
