pymodulon - Python package for analyzing and visualizing iModulons
==================================================================

|PyPI|

What is an iModulon?
~~~~~~~~~~~~~~~~~~~~
To learn about iModulons, how they are computed, and what they can tell you, see our `about page <https://imodulondb.org/about.html>`_.

Installation
~~~~~~~~~~~~

Since pymodulon is currently under development, the recommended method to
install **pymodulon** is to use the editable ``pip`` installation. It is
recommended to do this inside a `virtual environment
<http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_ or in a `conda
environment <https://docs.conda.io/en/latest/>`_. This is because we require
Python 3.8 for certain functionalities.

To create the conda environment::

	conda create -n pymodulon python=3.8
	conda activate pymodulon

(Optional) Install graphviz::

	conda install -c anaconda graphviz

Next, download the github repository::

	git clone https://github.com/SBRG/pymodulon.git

Then install with ``pip`` using the ``-e`` flag::

	python -m pip install -e .

This method of installation will automatically update your
package each time you pull from this repository.

To update your code, run the following from your local pymodulon folder::

	git pull

.. The recommended method is to install **pymodulon** is to use ``pip`` to
.. `install pymodulon from PyPI <https://pypi.python.org/pypi/pymodulon>`_. It is
.. recommended to do this inside a `virtual environment
.. <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_)::

.. 	pip install pymodulon

.. in the ``pymodulon`` source directory. For additional information, please refer to the
.. the `detailed installation instructions <INSTALL.rst>`_.

Development
-----------
If you would like to assist in development, please install `pre-commit <https://pre-commit.com/>`_ to ensure code style and consistency.

Cite
----
Please cite the original *E. coli* iModulon paper: `Sastry et al. Nature Communications. 2019. <https://www.nature.com/articles/s41467-019-13483-w>`_

If you use organism-specific datasets in your work, please cite these datasets:

* *Escherichia coli*: `Sastry et al. Nature Communications. 2019. <https://www.nature.com/articles/s41467-019-13483-w>`_
* *Bacillus subtilis*: `Rychel et al. Nature Communications. 2020.  <https://www.nature.com/articles/s41467-020-20153-9>`_
* *Staphylococcus aureus*: `Poudel et al. PNAS. 2020. <https://www.pnas.org/content/117/29/17228.abstract>`_

.. |PyPI| image:: https://badge.fury.io/py/pymodulon.svg
    :target: https://pypi.python.org/pypi/pymodulon
