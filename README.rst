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
--------------------
To learn about iModulons, how they are computed, and what they can tell you, see our `about page <https://imodulondb.org/about.html>`_.

Installation
------------

With Docker
~~~~~~~~~~~

The easiest way to get started with PyModulon is using the Docker container. 

1. Install `Docker <https://docs.docker.com/get-docker/>`_
2. Open terminal and navigate to your work folder
3. Run the following commands to start a Jupyter notebook server::

	docker run -p 8888:8888 -v "${PWD}":/home/jovyan/work sbrg/pymodulon
	
4. Copy the URL from terminal to connect to the Jupyter notebook
5. Navigate to the ``work`` folder, which has your current directory mounted.
6. To close the notebook, press ``Ctrl+C`` in terminal

With Pip
~~~~~~~~

You can install PyModulon from `PyPI <https://pypi.org/project/pymodulon/>`_ using ``pip`` as follows::

        python -m pip install pymodulon
	
With Conda
~~~~~~~~~~

Alternatively, you can install using `Conda <http://anaconda.org/>`_::

        conda install -c conda-forge pymodulon

We recommend installing through a conda environment::

	conda create -n pymodulon -c conda-forge pymodulon
	conda activate pymodulon

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Some features of PyModulon require additional dependencies. Follow the links below for installation instructions.

1. `GraphViz <https://graphviz.org/download/>`_
2. `NCBI BLAST <https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download>`_
3. `MEME Suite <https://meme-suite.org/meme/doc/install.html?man_type=web>`_

This step is not necessary if you use the Docker container.

Documentation
-------------
The documentation for **PyModulon** can be found `here <http://pymodulon.readthedocs.io/>`_.

Development
-----------
If you would like to assist in development, please install `pre-commit <https://pre-commit.com/>`_ to ensure code style and consistency.

We recommend using an editable pip installation for development::

	git clone https://github.com/SBRG/pymodulon.git
	cd pymodulon
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
