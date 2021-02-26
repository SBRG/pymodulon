PyModulon: Analyzing iModulons in Python
========================================

.. TODO: Make and add logo

|PyVer| |PyPiVer| |Black| |Precom|

What is an iModulon?
--------------------
Gene expression datasets are highly complex. Differential expression analysis is the
most common analytical tool used to analyze these datasets, but often result in
large sets of differentially genes, many of which have little to no functional
annotation. Recent studies of large transcriptomic datasets have introduced
Independent Component Analysis (ICA) as a scalable alternative that produces
easily-interpretable results (see :cite:`Sastry2019`). When applied to a gene
expression dataset, ICA extracts **i**\ndependently **modul**\ated groups of genes,
called **iModulons**. In bacteria, iModulons recapitulate known regulation with
remarkable accuracy, and often encode complete metabolic pathways [cite]. In
addition, ICA simultaneously computes iModulon activities, which represents the
effect of the associated transcriptional regulator on gene expression. For more
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

Installation and Setup
----------------------
We recommend you install **PyModulon** using ``pip`` within a `virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_::

   pip install pymodulon

The motif module requires installing `MEME <https://meme-suite.org/meme/>`_. We
recommend that you use the `docker container for MEME <https://hub.docker
.com/r/memesuite/memesuite>`_.

Tutorials
---------

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


API
---

For documentation on a specific method or function,
search the :ref:`/autoapi/index.rst`

.. toctree::
   :maxdepth: 1
   :caption: Contents

   autoapi/index

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources

   zreferences

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |PyVer| image:: https://img.shields.io/pypi/pyversions/pymodulon?logo=Python&style=plastic
    :target: https://www.python.org/downloads/
    :alt: Supported Python Versions

.. |PyPiVer| image:: https://img.shields.io/pypi/v/pymodulon?logo=PyPi&style=plastic
    :target: https://pypi.org/project/pymodulon/
    :alt: Current PyPi version

.. |Black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Black

.. |Precom| image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit
