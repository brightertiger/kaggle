from setuptools import setup, find_packages
from pathlib import Path

ROOT = Path(__file__).resolve().parent

with open(ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(ROOT / "requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="rsna-intracranial-hemorrhage",
    version="1.0.0",
    author="Ujjwal Singh Rao",
    description="Deep Learning Pipeline for RSNA Intracranial Hemorrhage Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brightertiger/kaggle/tree/main/rsna",
    packages=find_packages(),
    py_modules=["main"],
    classifiers=[
        "Development Status :: 4 - Beta",
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
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "rsna-hemorrhage=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
