Mining public expression databases
==================================

We have provided a docker container that finds all publicly available RNA-seq data for a given organism and collates the metadata into a single file.

For example, the following code finds all RNA-seq data for *Bacillus subtilis* and saves the data to ``Bacillus_subtilis.tsv``

.. code-block:: bash

   docker run --rm -it avsastry/get-all-rnaseq:latest "Bacillus subtilis" > Bacillus_subtilis.tsv
   
More generally, this can be run for any organism:

.. code-block:: bash

   docker run --rm -it avsastry/get-all-rnaseq:latest "<organism name>" > <filename>
