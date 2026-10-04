from pathlib import Path

from setuptools import find_packages, setup

setup(
    name='toxic-comment-classification',
    version='1.0.0',
    description='Multi-label toxic comment classification with GRU and sparse text ensembles',
    author='Ujjwal Singh Rao',
    packages=find_packages(),
    python_requires='>=3.11',
    install_requires=Path(__file__).with_name('requirements.txt').read_text().splitlines(),
    classifiers=[
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
    ],
)
