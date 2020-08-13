#!/usr/bin/env python3

from pathlib import Path

import setuptools

project_dir = Path(__file__).parent

setuptools.setup(
    name="pymodulon",
    version="0.0.1",
    description="Python package for analyzing and visualizing iModulons",
    # Allow UTF-8 characters in README with encoding argument.
    long_description=project_dir.joinpath("README.md").read_text(encoding="utf-8"),
    keywords=("python iModulon RNA-seq transcriptomics ICA regulation"),
    author="Anand V. Sastry",
    author_email="avsastry@eng.ucsd.edu",
    maintainer="Anand V. Sastry",
    maintainer_email="avsastry@eng.ucsd.edu",
    url="https://github.com/SBRG/pymodulon",
    packages=setuptools.find_packages("src"),
    package_dir={"": "src"},
    # pip 9.0+ will inspect this field when installing to help users install a
    # compatible version of the library for their Python version.
    python_requires=">3.6",
    # There are some peculiarities on how to include package data for source
    # distributions using setuptools. You also need to add entries for package
    # data to MANIFEST.in.
    # See https://stackoverflow.com/questions/7522250/
    include_package_data=True,
    # This is a trick to avoid duplicating dependencies between both setup.py and
    # requirements.txt.
    # requirements.txt must be included in MANIFEST.in for this to work.
    # It does not work for all types of dependencies (e.g. VCS dependencies).
    # For VCS dependencies, use pip >= 19 and the PEP 508 syntax.
    #   Example: 'requests @ git+https://github.com/requests/requests.git@branch_or_tag'
    #   See: https://github.com/pypa/pip/issues/6162
    install_requires=project_dir.joinpath("requirements.txt").read_text().split("\n"),
    zip_safe=False,
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
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics"
    ],
    platforms="GNU/Linux, Mac OS X > 10.7, Microsoft Windows >= 7",
)
