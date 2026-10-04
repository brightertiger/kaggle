from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

with open(ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(ROOT / "requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="spooky-author-identification",
    version="1.0.0",
    author="Ujjwal Singh Rao",
    description="A comprehensive machine learning pipeline for author identification using text analysis techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brightertiger/kaggle/tree/main/spooky",
    packages=find_packages(),
    py_modules=["main"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pytest>=7.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "spooky-author=main:main",
        ],
    },
)
