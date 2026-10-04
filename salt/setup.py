from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).resolve().parent
setup(
    name='salt-identification',
    version='1.0.0',
    author='Ujjwal Singh Rao',
    description='U-Net segmentation for the TGS Salt Identification Challenge',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/brightertiger/kaggle/tree/main/salt',
    packages=find_packages(where=str(ROOT)),
    package_dir={'': str(ROOT)},
    py_modules=['main'],
    python_requires='>=3.11',
    install_requires=(ROOT / 'requirements.txt').read_text().splitlines(),
    entry_points={'console_scripts': ['salt-segmentation=main:main']},
)
