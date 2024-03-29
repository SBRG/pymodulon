# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup ---------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute.

import os
import sys


BASE_PATH = os.path.dirname(os.path.dirname(__file__))
SRC_PATH = os.path.join(BASE_PATH, "src")
sys.path.insert(0, SRC_PATH)

# This import has to be here below inserting SRC path.
from pymodulon import __version__ as release  # noqa: E402


# -- General configuration ----------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx_rtd_theme",
    "sphinxcontrib.bibtex",
]

# Automated documention of Python Code (autoapi)
autoapi_type = "python"
autoapi_dirs = [SRC_PATH]

# Move typehints from signature to description
autodoc_typehints = "description"

# Automated section labeling (autosectionlabel)
autosectionlabel_prefix_document = True

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "**.ipynb_checkpoints", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

bibtex_bibfiles = ["zreferences.bib"]

# Ensure libraries are loaded for docs

autodoc_mock_imports = [
    "adjusttext",
    "biopython",
    "graphviz",
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "seaborn",
    "statsmodels",
    "tqdm",
    "matplotlib_venn",
]

# Remove docstrings for undocumented functions
autodoc_default_flags = [
    "members",
    "private-members",
    "special-members",
    # 'undoc-members',
    "show-inheritance",
]

# -- Project information ------------------------------------------------------

project = "pymodulon"
copyright = "2020, Anand Sastry"
author = "Anand Sastry"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
version = ".".join(release.split(".")[:2])

# -- Options for HTML output --------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# A list of paths that contain extra files not directly related to the
# documentation, such as robots.txt or .htaccess. Relative paths are taken as
# relative to the configuration directory. They are copied to the output
# directory. They will overwrite any existing file of the same name.
html_extra_path = ["robots.txt"]

# -- Options for linkcheck --------------------------------------------------
linkcheck_ignore = [
    r"^https://doi.org/+",  # Always redirects
]

# -- NBSphinx -----------------------------------------------------------------

# Execute notebooks before conversion: 'always', 'never', 'auto' (default)
nbsphinx_execute = "always"
nbsphinx_allow_errors = False
nbsphinx_execute_arguments = [
    "--Application.log_level=CRITICAL",
]
nbsphinx_timeout = 180

# -- Intersphinx --------------------------------------------------------------

# Refer to the Python documentation for other libraries.
intersphinx_mapping = {
    "http://docs.python.org/": None,
    "https://matplotlib.org/stable/": None,
    "https://numpy.org/doc/stable/": None,
    "https://pandas.pydata.org/pandas-docs/stable/": None,
    "http://docs.scipy.org/doc/scipy/reference": None,
}
intersphinx_cache_limit = 10  # days to keep the cached inventories
