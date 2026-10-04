from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent

with open(ROOT / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(ROOT / "requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tweet-sentiment-analysis",
    version="1.0.0",
    author="Ujjwal Singh Rao",
    description="Tweet Sentiment Extraction with RoBERTa span prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brightertiger/kaggle/tree/main/tweet",
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
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
    },
    entry_points={
        "console_scripts": [
            "tweet-sentiment=main:main",
        ],
    },
)
