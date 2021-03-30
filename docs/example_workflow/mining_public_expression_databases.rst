Mining public expression databases
==================================

We have provided a docker container `here<https://hub.docker.com/repository/docker/avsastry/get-all-rnaseq>`_ that finds all publicly available RNA-seq data for a given organism and collates the metadata into a single file. `Docker <https://docs.docker.com/get-docker/>`_ must be installed to run the following commands.

For example, the following code finds all RNA-seq data for *Bacillus subtilis* and saves the data to ``Bacillus_subtilis.tsv``

.. code-block:: bash

   docker run --rm -it avsastry/get-all-rnaseq "Bacillus subtilis" > Bacillus_subtilis.tsv
   
More generally, this can be run for any organism:

.. code-block:: bash

   docker run --rm -it avsastry/get-all-rnaseq "<organism name>" > <filename>
