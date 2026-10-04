from pathlib import Path
from setuptools import setup, find_packages

ROOT = Path(__file__).resolve().parent
requirements = [line.strip() for line in (ROOT / 'requirements.txt').read_text().splitlines()
                if line.strip() and not line.startswith('#')]
use_requirements = [line.strip() for line in (ROOT / 'requirements-use.txt').read_text().splitlines()
                    if line.strip() and not line.startswith('#')]

setup(
    name='jigsawml', version='1.0.0', author='Ujjwal Singh Rao',
    description='Jigsaw Multilingual Toxic Comment Classification Solution',
    long_description=(ROOT / 'README.md').read_text(encoding='utf-8'),
    long_description_content_type='text/markdown',
    url='https://github.com/brightertiger/kaggle/tree/main/jigsawml',
    packages=find_packages(), py_modules=['main'], python_requires='>=3.11',
    install_requires=requirements, extras_require={'use': use_requirements},
    entry_points={'console_scripts': ['jigsawml=main:main']},
)
