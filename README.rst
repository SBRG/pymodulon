======================================================================
**PyModulon** - Python package for analyzing and visualizing iModulons
======================================================================

.. image:: https://img.shields.io/pypi/v/pymodulon
    :target: https://pypi.org/project/pymodulon
    :alt: PyPI

.. image:: https://img.shields.io/pypi/pyversions/pymodulon
    :target: https://pypi.org/project/pymodulon
    :alt: PyPI - Python Version

.. image:: https://img.shields.io/github/license/sbrg/pymodulon
    :target: https://opensource.org/licenses/MIT
    :alt: MIT License

.. image:: https://img.shields.io/readthedocs/pymodulon
    :target: https://pymodulon.readthedocs.io/en/latest/
    :alt: Documentation Status

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black code style

.. image:: https://anaconda.org/conda-forge/pymodulon/badges/installer/conda.svg
    :target: https://conda.anaconda.org/conda-forge
    :alt: Conda installation

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

What is an iModulon?
~~~~~~~~~~~~~~~~~~~~
To learn about iModulons, how they are computed, and what they can tell you, see our `about page <https://imodulondb.org/about.html>`_.

Installation
~~~~~~~~~~~~

You can install PyModulon from `PyPI <https://pypi.org/project/pymodulon/>`_ using ``pip`` as follows::

        pip install pymodulon

Alternatively, you can install using `Conda <http://anaconda.org/>`_::

        conda install -c avsastry -c conda-forge pymodulon

We recommend installing through a conda environment::

	conda create -n pymodulon -c conda-forge pymodulon
	conda activate pymodulon

(Optional) Install other dependencies::

	conda install -c anaconda graphviz
	conda install -c bioconda meme blast

Documentation
-------------
The documentation for **PyModulon** can be found `here <http://pymodulon.readthedocs.io/>`_.

Development
-----------
If you would like to assist in development, please install `pre-commit <https://pre-commit.com/>`_ to ensure code style and consistency.

Since **PyModulon** is currently under development, the recommended method to
install **PyModulon** is to use the editable ``pip`` installation within a `virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_ or `conda
environment <https://docs.conda.io/en/latest/>`_.

First initialize the conda environment::

    conda create -n pymodulon-dev python=3.8
    conda activate pymodulon-dev

(Optional) Install other dependencies::

	conda install -c anaconda graphviz
	conda install -c bioconda meme blast

Next, download the github repository::

	git clone https://github.com/SBRG/pymodulon.git

Then install with ``pip`` using the ``-e`` flag::

	python -m pip install -e .

This method of installation will automatically update your
package each time you pull from this repository.

To update your code, run the following from your local **PyModulon** folder::

	git pull


Cite
----
Please cite the original *E. coli* iModulon paper: `Sastry et al. Nature Communications. 2019. <https://www.nature.com/articles/s41467-019-13483-w>`_

If you use organism-specific datasets in your work, please cite these datasets:

* *Escherichia coli*: `Sastry et al. Nature Communications. 2019. <https://www.nature.com/articles/s41467-019-13483-w>`_
* *Bacillus subtilis*: `Rychel et al. Nature Communications. 2020.  <https://www.nature.com/articles/s41467-020-20153-9>`_
* *Staphylococcus aureus*: `Poudel et al. PNAS. 2020. <https://www.pnas.org/content/117/29/17228.abstract>`_
