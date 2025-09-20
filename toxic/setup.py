from setuptools import setup, find_packages

setup(
    name="toxic-comment-classification",
    version="1.0.0",
    description="Multi-class toxic comment classification using ensemble methods",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.0.0",
        "keras>=2.6.0",
        "tensorflow>=2.6.0",
        "nltk>=3.6.0",
        "regex>=2021.8.3",
        "dask>=2021.6.0",
        "keras-contrib>=2.0.8",
        "fasttext>=0.9.2"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
