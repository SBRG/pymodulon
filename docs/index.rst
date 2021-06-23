PyModulon: Analyzing iModulons in Python
========================================

.. TODO: Make and add logo

|PyVer| |PyPi| |Conda| |Docker| |Docs| |Style| |Lic|

What is an iModulon?
--------------------
Gene expression datasets are highly complex. Differential expression analysis is the
most common analytical tool used to analyze these datasets, but often result in
large sets of differentially genes, many of which have little to no functional
annotation. Recent studies of large transcriptomic datasets have introduced
Independent Component Analysis (ICA) as a scalable alternative that produces
easily-interpretable results :cite:`Sastry2019`.

When applied to a gene
expression dataset, ICA extracts **i**\ndependently **modul**\ated groups of genes,
called **iModulons**. In bacteria, iModulons recapitulate known regulation with
remarkable accuracy, and often encode complete metabolic pathways [cite].

In
addition, ICA simultaneously computes iModulon activities, which represents the
effect of the associated transcriptional regulator on gene expression.

For more
information on iModulons, see the `iModulonDB About Page <https://imodulondb
.org/about.html>`_.

What is PyModulon?
------------------
**PyModulon** contains modules for the analysis, visualization, and
dissemination of
iModulons. **PyModulon** is designed with `Jupyter <https://jupyter.org/>`_ notebooks
in mind. In addition, it enables users to easily create their own
`iModulonDB <https://imodulondb.org/>`_ pages for any analyzed dataset.

Citation
~~~~~~~~
The pymodulon manuscript is currently in preparation. For now, please cite
:cite:`Sastry2019`. If you use the custom `iModulonDB <https://imodulondb.org/>`_
pages, please cite :cite:`Rychel2020b`.

Installation
------------

With Docker
~~~~~~~~~~~

The easiest way to get started with PyModulon is using the Docker container.

1. Install `Docker <https://docs.docker.com/get-docker/>`_
2. Open terminal and navigate to your work folder
3. Run the following commands to start a `Jupyter Notebook <https://jupyter.org/>`_ server::

    docker run -p 8888:8888 -v "${PWD}":/home/jovyan/work sbrg/pymodulon

4. Copy the URL from terminal to connect to the Jupyter notebook
5. Navigate to the ``work`` folder, which has your current directory mounted.
6. To close the notebook, press ``Ctrl+C`` in terminal. All changes made to files in your current directory are saved to your local machine.

With Pip
~~~~~~~~

You can install PyModulon from `PyPI <https://pypi.org/project/pymodulon/>`_ using ``pip`` as follows::

        python -m pip install pymodulon

With Conda
~~~~~~~~~~

Alternatively, you can install using `Conda <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_::

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


Example Workflow
----------------

We have provided an example workflow `here <https://github.com/avsastry/modulome-workflow>`_ illustrating how to compile and process all publicly available RNA-seq data for *Bacillus subtilis*, and subsequently compute and characterize its iModulons. This tutorial was designed to be easily applied to any other bacteria.

.. toctree::
   :numbered:
   :maxdepth: 1
   :caption: Tutorials

   tutorials/introduction_to_the_ica_data_object.ipynb
   tutorials/plotting_functions.ipynb
   tutorials/inferring_imodulon_activities_for_new_data.ipynb
   tutorials/gene_enrichment_analysis.ipynb
   tutorials/searching_for_motifs.ipynb
   tutorials/imodulon_thresholds.ipynb
   tutorials/comparing_imodulons.ipynb
   tutorials/creating_the_gene_table.ipynb
   tutorials/additional_functions.ipynb
   tutorials/creating_an_imodulondb_dashboard.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   autoapi/index
   zreferences

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. |PyVer| image:: https://img.shields.io/pypi/pyversions/pymodulon?logo=Python&style=plastic
    :target: https://www.python.org/downloads/
    :alt: PyPI - Python Version

.. |PyPI| image:: https://img.shields.io/pypi/v/pymodulon?logo=PyPi&style=plastic
    :target: https://pypi.org/project/pymodulon/
    :alt: PyPI

.. |Conda| image:: https://img.shields.io/conda/vn/conda-forge/pymodulon?logo=anaconda&style=plastic
    :target: https://conda.anaconda.org/conda-forge
    :alt: Conda installation

.. |Docker| image:: https://img.shields.io/docker/v/sbrg/pymodulon?label=Docker&logo=Docker&sort=semver&style=plastic
    :target: https://hub.docker.com/r/sbrg/pymodulon
    :alt: Docker container

.. |Docs| image:: https://img.shields.io/readthedocs/pymodulon?logo=Read%20The%20Docs&style=plastic
    :target: https://pymodulon.readthedocs.io/en/latest/
    :alt: Documentation Status

.. |Style| image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=plastic
    :target: https://github.com/psf/black
    :alt: Black code style

.. |Lic| image:: https://img.shields.io/github/license/sbrg/pymodulon?logo=license&style=plastic
    :target: https://opensource.org/licenses/MIT
    :alt: MIT License
