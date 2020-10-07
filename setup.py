#!/usr/bin/env python3

from pathlib import Path

import setuptools

project_dir = Path(__file__).parent

setuptools.setup(
    name="pymodulon",
    version="0.0.2",
    description="Python package for analyzing and visualizing iModulons",
    long_description=project_dir.joinpath("README.rst").read_text(encoding="utf-8"),
    keywords=("python iModulon RNA-seq transcriptomics ICA regulation"),
    author="Anand V. Sastry",
    author_email="avsastry@eng.ucsd.edu",
    maintainer="Anand V. Sastry",
    maintainer_email="avsastry@eng.ucsd.edu",
    url="https://github.com/SBRG/pymodulon",
    packages=setuptools.find_packages(),
    python_requires=">3.6",
    include_package_data=True,
    install_requires=project_dir.joinpath("requirements.txt").read_text().split("\n"),
    license="MIT",
    classifiers=[
        "Development Status :: 1 - Planning",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    platforms="GNU/Linux, Mac OS X > 10.7, Microsoft Windows >= 7",
)
