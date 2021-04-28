"""
Core functions for the IcaData object
"""

import copy
import logging
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm_notebook as tqdm

from pymodulon.enrichment import (
    compute_annotation_enrichment,
    compute_regulon_enrichment,
    compute_trn_enrichment,
    contingency,
)
from pymodulon.util import _check_dict, _check_table, compute_threshold


class IcaData(object):
    """
    Class representation of all iModulon-related data
    """

    def __init__(
        self,
        M,
        A,
        X=None,
        log_tpm=None,
        gene_table=None,
        sample_table=None,
        imodulon_table=None,
        trn=None,
        dagostino_cutoff=None,
        optimize_cutoff=False,
        thresholds=None,
        threshold_method="dagostino",
        motif_info=None,
        dataset_table=None,
        splash_table=None,
        gene_links=None,
        tf_links=None,
        link_database="External Database",
    ):
        """
        Initialize IcaData object

        Parameters
        ----------
        M : str or ~pandas.DataFrame
            `M` matrix from ICA
        A : str or ~pandas.DataFrame
            `A` matrix from ICA
        X : str or ~pandas.DataFrame, optional
            log-TPM expression matrix, centered to reference condition(s) (
            default: None)
        log_tpm : str or ~pandas.DataFrame, optional
            Raw `log-TPM` expression matrix (without centering) (default: None)
        gene_table : str or ~pandas.DataFrame, optional
            Table containing genome annotation (default: None)
        sample_table : str or ~pandas.DataFrame, optional
            Table containing sample metadata (default: None)
        imodulon_table : str or ~pandas.DataFrame, optional
            Table containing iModulon names, enrichments, and annotations
            (default: None)
        trn : str or ~pandas.DataFrame, optional
            Table mapping transcriptional regulators to target genes (
            default: None)
        dagostino_cutoff : int, optional
            Cutoff value to use for the D'agostino test for iModulon gene
            thresholds. This option will be ignored if optimize_cutoff is True,
            if threshold_method is "kmeans", or if custom thresholds are
            provided.
        optimize_cutoff : bool
            If true, optimize the D'agostino cutoff for iModulon threshold using
            the TRN (if provided). This option will be ignored if
            threshold_method is "kmeans" or if custom thresholds are
            provided. (default: False)
        thresholds : dict, optional
            Dictionary mapping custom thresholds to iModulons (default: None). If
            numerical thresholds are supplied, this supercedes all related options (
            i.e. `threshold_method`, `optimize_cutoff`, `dagostino_cutoff`).
        threshold_method : str
            Either "dagostino" (default with TRN) or "kmeans" (default if no TRN
            provided)
        dataset_table : dict, optional
            Dictionary of general dataset information for the details box on the
            dataset page of iModulonDB (default: None)
        splash_table : dict, optional
            Dictionary of general information for the splash page
            link to this dataset, as well as folder names for where its data
            is stored in iModulonDB (default: None)
        gene_links : dict, optional
            dictionary of genes to links in an external database (default: None)
        tf_links : dict, optional
            Dictionary of TFs (from the TRN) to links in a database (default:
            None)
        link_database : str, optional
            Name of the database for the gene_links dictionary (default:
            "External Database")
        """

        #########################
        # Load M and A matrices #
        #########################

        M = _check_table(M, "M")
        A = _check_table(A, "A")

        # Convert column names of M to int if possible
        try:
            M.columns = M.columns.astype(int)
        except TypeError:
            pass

        # Check that M and A matrices have identical iModulon names
        if M.columns.tolist() != A.index.tolist():
            raise ValueError("M and A matrices have different iModulon names")

        # Ensure that M and A matrices have unique indices/columns
        if M.index.duplicated().any():
            raise ValueError("M matrix contains duplicate gene names")
        if A.columns.duplicated().any():
            raise ValueError("A matrix contains duplicate sample names")
        if M.columns.duplicated().any():
            raise ValueError("M and A matrices contain " "duplicate iModulon names")

        # Store M and A
        self._m = M
        self._a = A

        #################
        # Load X matrix #
        #################

        # Check X matrix
        if X is None:
            self._x = None
        else:
            self.X = X

        if log_tpm is None:
            self._log_tpm = None
        else:
            self.log_tpm = log_tpm

        ####################
        # Load data tables #
        ####################

        # Initialize sample and gene names
        self._gene_names = M.index.tolist()
        self.gene_table = gene_table
        self._sample_names = A.columns.tolist()
        self.sample_table = sample_table
        self._imodulon_names = M.columns.tolist()
        self.imodulon_table = imodulon_table

        ############
        # Load TRN #
        ############

        self.trn = trn

        # Initialize thresholds either with or without optimization
        self._cutoff_optimized = False
        if thresholds is not None:
            # Throw a warning if user was expecting d'agostino optimization
            if optimize_cutoff:
                logging.warning(
                    "Using manually input thresholds. D'agostino "
                    "optimization will not be performed."
                )
            self.thresholds = thresholds
            self._dagostino_cutoff = None

        # Use kmeans if TRN is empty, or kmeans is selected
        elif self.trn.empty or threshold_method == "kmeans":
            # Throw a warning if user was expecting d'agostino optimization
            if optimize_cutoff:
                logging.warning(
                    "Using Kmeans threshold method. D'agostino "
                    "optimization will not be performed."
                )
            self.compute_kmeans_thresholds()
            self._dagostino_cutoff = None

        # Else use D'agostino method
        elif threshold_method == "dagostino":
            if optimize_cutoff:
                logging.warning(
                    "Optimizing iModulon thresholds, may take 2-3 minutes..."
                )
                # this function sets self.dagostino_cutoff internally
                self.reoptimize_thresholds(progress=False, plot=False)
                # also sets an attribute to tell us if we've done
                # this optimization; only reasonable to try it
                # again if the user uploads a new TRN
            elif dagostino_cutoff is None:
                logging.warning(
                    "Using the default dagostino_cutoff of 550. This may "
                    "not be optimal for your dataset. Use "
                    "ica_data.reoptimize_thresholds() to find the optimal "
                    "threshold."
                )
                self._dagostino_cutoff = 550
                self.recompute_thresholds(self.dagostino_cutoff)
            else:
                self._dagostino_cutoff = dagostino_cutoff
                self.recompute_thresholds(self.dagostino_cutoff)
        # Capture improper threshold methods
        else:
            raise ValueError('Threshold method must either be "dagostino" or "kmeans"')

        ##############
        # Motif Info #
        ##############

        self.motif_info = motif_info

        ##############################
        # Load iModulonDB Properties #
        ##############################

        # initialize links
        self.dataset_table = dataset_table
        self.splash_table = splash_table
        self.link_database = link_database
        self.gene_links = gene_links
        self.tf_links = tf_links

        # Initialize COG colors
        if "COG" in self.gene_table.columns:
            cogs = sorted(self.gene_table.COG.unique())
            self.cog_colors = dict(
                zip(
                    cogs,
                    [
                        "red",
                        "pink",
                        "y",
                        "orchid",
                        "mediumvioletred",
                        "green",
                        "lightgray",
                        "lightgreen",
                        "slategray",
                        "blue",
                        "saddlebrown",
                        "turquoise",
                        "lightskyblue",
                        "c",
                        "skyblue",
                        "lightblue",
                        "fuchsia",
                        "dodgerblue",
                        "lime",
                        "sandybrown",
                        "black",
                        "goldenrod",
                        "chocolate",
                        "orange",
                    ],
                )
            )

        # Initialize motif info
        self._motif_info = {}
        if motif_info is not None:
            for k1, v1 in motif_info.items():
                params = {}
                for k2, v2 in v1.items():
                    try:
                        params[k2] = pd.read_json(v2, orient="table")
                    except ValueError:
                        params[k2] = v2
                self._motif_info[k1] = MotifInfo(**params)

    @property
    def M(self):
        """Get M matrix"""
        return self._m

    @property
    def M_binarized(self):
        """Get binarized version of M matrix based on current thresholds"""
        m_binarized = pd.DataFrame().reindex_like(self.M)
        m_binarized[:] = 0
        for imodulon in m_binarized.columns:
            in_imodulon = abs(self.M[imodulon]) > self.thresholds[imodulon]
            m_binarized.loc[in_imodulon, imodulon] = 1
        return m_binarized

    @property
    def A(self):
        """Get A matrix"""
        return self._a

    @property
    def X(self):
        """Get X matrix"""
        return self._x

    @X.setter
    def X(self, x_matrix):
        x = _check_table(x_matrix, "X")

        # Check that gene and sample names conform to M and A matrices
        if x.columns.tolist() != self.A.columns.tolist():
            raise ValueError("X and A matrices have different sample names")
        if x.index.tolist() != self.M.index.tolist():
            raise ValueError("X and M matrices have different gene names")

        # Set x matrix
        self._x = x

    @property
    def log_tpm(self):
        """Get 'log_tpm' matrix"""
        return self._log_tpm

    @log_tpm.setter
    def log_tpm(self, lt_matrix):
        log_tpm = _check_table(lt_matrix, "log-TPM")

        # Check that gene and sample names conform to M and A matrices
        if log_tpm.columns.tolist() != self.A.columns.tolist():
            raise ValueError("log-TPM and A matrices have different sample " "names")
        if log_tpm.index.tolist() != self.M.index.tolist():
            raise ValueError("log-TPM and M matrices have different gene names")

        # Set log-tpm matrix
        self._log_tpm = log_tpm

    # Gene, sample and iModulon name properties
    @property
    def imodulon_names(self):
        """Get iModulon names"""
        return self._imodulon_table.index.tolist()

    @imodulon_names.setter
    def imodulon_names(self, new_names):
        self._update_imodulon_names(new_names)

    @property
    def sample_names(self):
        """Get sample names"""
        return self._sample_table.index.tolist()

    @property
    def gene_names(self):
        """Get gene names"""
        return self._gene_table.index.tolist()

    # Gene, sample and iModulon tables
    @property
    def gene_table(self):
        """Get gene table"""
        return self._gene_table

    @gene_table.setter
    def gene_table(self, new_table):
        table = _check_table(new_table, "gene", self._gene_names)
        self._gene_table = table

        # Update gene names
        names = table.index
        self._gene_names = names
        self._m.index = names
        if self._x is not None:
            self._x.index = names

    @property
    def sample_table(self):
        """Get sample table"""
        return self._sample_table

    @sample_table.setter
    def sample_table(self, new_table):
        table = _check_table(new_table, "sample", self._sample_names)
        self._sample_table = table

        # Update sample names
        names = table.index
        self._sample_names = names
        self._a.columns = names
        if self._x is not None:
            self._x.columns = names

    @property
    def imodulon_table(self):
        """Get table of iModulons"""
        return self._imodulon_table

    @imodulon_table.setter
    def imodulon_table(self, new_table):
        table = _check_table(new_table, "imodulon", self._imodulon_names)
        self._imodulon_table = table

    # TRN
    @property
    def trn(self):
        """Get table with TRN information"""
        return self._trn

    @trn.setter
    def trn(self, new_trn):
        self._trn = _check_table(new_trn, "TRN", index_col=None)
        if not self._trn.empty:
            # Check that regulator and gene_id columns are filled in
            if self._trn.regulator.isnull().any():
                raise ValueError('Null value detected in "regulator" column ' "of TRN")
            if self._trn.gene_id.isnull().any():
                raise ValueError('Null value detected in "gene_id" column ' "of TRN")

            # Make sure regulators do not contain / or + characters
            for reg in self._trn.regulator.unique():
                if "+" in reg or "/" in reg:
                    logging.warning(
                        "The characters '+' and '/' are used for combining "
                        "regulons and cannot be in regulator names. These "
                        "characters will be replaced with ';'"
                    )
                    break

            self._trn.regulator = [
                re.sub("\\+", ";", re.sub("/", ";", reg)) for reg in self._trn.regulator
            ]

            # Only include genes that are in S/X matrix
            extra_genes = set(self._trn.gene_id) - set(self.gene_names)
            if len(extra_genes) > 0:
                logging.warning(
                    "The following genes are in the TRN but not in "
                    "your M "
                    "matrix: {}".format(extra_genes)
                )
            self._trn = self._trn[self._trn.gene_id.isin(self.gene_names)]

            # Save regulator information to gene table
            reg_dict = {}
            for name, group in self._trn.groupby("gene_id"):
                reg_dict[name] = ",".join(group.regulator)
            self._gene_table["regulator"] = pd.Series(reg_dict).reindex(self.gene_names)

        # mark that our cutoffs are no longer optimized since the TRN
        self._cutoff_optimized = False

    # Motif information
    @property
    def motif_info(self):
        """Get motif info"""
        return self._motif_info

    @motif_info.setter
    def motif_info(self, info):
        self._motif_info = info

    def _update_imodulon_names(self, new_names):
        """
        Iterates and updates iModulon names

        Parameters
        ----------
        new_names: list
            New names to update iModulons

        Returns
        -------
        None: None
        """
        name_series = pd.Series(new_names, index=self.imodulon_names)

        # Check if new names are duplicates
        dups = name_series[name_series.duplicated(keep=False)]
        if len(dups) > 0:
            seen = {}
            # For duplicated names, add a "-1" or "-2" etc.
            for key, val in dups.items():
                if val in seen.keys():
                    name_series[key] = val + "-" + str(seen[val])
                    seen[val] += 1
                else:
                    name_series[key] = val + "-1"
                    seen[val] = 2

                logging.warning(
                    "Duplicate iModulon names detected. iModulon {} will "
                    "be renamed to {}".format(key, name_series[key])
                )

        # Update thresholds and motif info
        for old_name, new_name in name_series.items():
            self._thresholds[new_name] = self._thresholds.pop(old_name)
            try:
                self._motif_info[new_name] = self._motif_info.pop(old_name)
            except KeyError:
                pass

        # Update iModulon names
        final_names = name_series.values.tolist()
        self._imodulon_names = final_names
        self._a.index = final_names
        self._m.columns = final_names
        self._imodulon_table.index = final_names

    def rename_imodulons(self, name_dict=None, column=None) -> None:
        """
        Rename an iModulon.

        Parameters
        ----------
        name_dict : dict
            Dictionary mapping old iModulon names to new names
            (e.g. {old_name:new_name}) (default: None)
        column : str
            Uses a column from the iModulon table to rename iModulons (
            default: None)

        Returns
        -------
        None: None
        """

        # Rename using the column parameter if given
        if column is not None:
            raise DeprecationWarning(
                "column paramter will be removed soon. Please "
                "use 'ica_data.imodulon_names = "
                "ica_data.imodulon_table[column]'"
            )
        else:
            new_names = [
                name_dict[name] if name in name_dict.keys() else name
                for name in self.imodulon_names
            ]

        self._update_imodulon_names(new_names)

    # Show enriched
    def view_imodulon(self, imodulon):
        """
        View genes in an iModulon and show relevant information about each gene.

        Parameters
        ----------
        imodulon : int or str
            Name of iModulon

        Returns
        -------
        final_rows: ~pandas.DataFrame
            Table showing iModulon gene information

        """

        # Find genes in iModulon
        in_imodulon = abs(self.M[imodulon]) > self.thresholds[imodulon]

        # Get gene weights information
        gene_weights = self.M.loc[in_imodulon, imodulon]
        gene_weights.name = "gene_weight"
        gene_rows = self.gene_table.loc[in_imodulon]
        final_rows = pd.concat([gene_weights, gene_rows], axis=1)

        return final_rows

    def find_single_gene_imodulons(self, save=False):
        """
        A simple function that returns the names of all likely single-gene
        iModulons. Checks if the largest iModulon gene weight is more
        than twice the weight of the second highest iModulon gene weight.

        Parameters
        ----------
        save : bool
            If true, save output to imodulon_table (default: False)

        Returns
        -------
        single_genes_imodulons: list
            List of single-gene iModulons
        """

        single_genes_imodulons = []
        for imodulon in self.imodulon_names:
            sorted_weights = abs(self.M[imodulon]).sort_values(ascending=False)
            if sorted_weights.iloc[0] > 2 * sorted_weights.iloc[1]:
                single_genes_imodulons.append(imodulon)
                if save:
                    self.imodulon_table.loc[imodulon, "single_gene"] = True
        return single_genes_imodulons

    ###############
    # Enrichments #
    ###############

    def _update_imodulon_table(self, enrichment):
        """
        Update iModulon table given new iModulon enrichments

        Parameters
        ----------
        enrichment : ~pandas.Series or ~pandas.DataFrame iModulon enrichment

        Returns
        -------
        None: None
        """
        if isinstance(enrichment, pd.Series):
            enrichment = pd.DataFrame(enrichment)
        keep_rows = self.imodulon_table[
            ~self.imodulon_table.index.isin(enrichment.index)
        ]
        keep_cols = self.imodulon_table.loc[
            enrichment.index, set(self.imodulon_table.columns) - set(enrichment.columns)
        ]
        df_top_enrich = pd.concat([enrichment, keep_cols], axis=1)
        new_table = pd.concat([keep_rows, df_top_enrich], sort=False)

        # Reorder columns
        col_order = enrichment.columns.tolist() + keep_cols.columns.tolist()
        new_table = new_table[col_order]

        # Reorder rows
        new_table = new_table.reindex(self.imodulon_names)

        self.imodulon_table = new_table

    def compute_regulon_enrichment(
        self, imodulon, regulator, save=False, evidence=None
    ):
        """
        Compare an iModulon against a regulon. (Note: q-values cannot be
        computed for single enrichments)

        Parameters
        ----------
        imodulon: int or str
            Name of 'iModulon'
        regulator: str
            TF name, or complex regulon, where "/" uses genes in any regulon
            and "+" uses genes in all regulons
        save: bool
            If true, save enrichment score to the imodulon_table (default: True)
        evidence: list or str
            'Evidence' level of TRN interactions to include during TRN
            enrichment (default: None)

        Returns
        -------
        enrich: ~pandas.Series
            Table containing enrichment statistics
        """

        if evidence is not None:
            if isinstance(evidence, str):
                evidences_to_use = [evidence]
            else:
                evidences_to_use = evidence

            if "evidence" in self.trn.columns:
                trn_to_use = self.trn[self.trn["evidence"].isin(evidences_to_use)]
            else:
                logging.warning(
                    'TRN does not contain an "evidence" column. Ignoring '
                    "evidence argument."
                )
                trn_to_use = self.trn
        else:
            trn_to_use = self.trn

        imod_genes = self.view_imodulon(imodulon).index
        enrich = compute_regulon_enrichment(
            set(imod_genes), regulator, set(self.gene_names), trn_to_use
        )
        enrich.rename({"gene_set_size": "imodulon_size"}, inplace=True)
        if save:
            table = self.imodulon_table
            for key, value in enrich.items():
                table.loc[imodulon, key] = value
                table.loc[imodulon, "regulator"] = enrich.name
            self.imodulon_table = table
        return enrich

    def compute_trn_enrichment(
        self,
        imodulons=None,
        fdr=1e-5,
        max_regs=1,
        save=False,
        method="both",
        force=False,
        evidence=None,
    ) -> pd.DataFrame:

        """
        Compare iModulons against all regulons in the TRN

        Parameters
        ----------
        imodulons: int or str
            Name of iModulon(s). If none given, compute enrichments for all
            'iModulons' (default: None)
        fdr : float
            False detection rate (default: 1e-5)
        max_regs : int
            Maximum number of regulators to include in complex regulon (
            default: 1)
        save : bool
            Save regulons with highest enrichment scores to the
            imodulon_table (default: False)
        method : str
            How to combine multiple regulators  (default: 'both').
            'or' computes enrichment against union of regulons,
            'and' computes enrichment against intersection of regulons, and
            'both' performs both tests
        force : bool
            If false, prevents computation of >2 regulators (default: False)
        evidence: list or str
            Evidence level of TRN interactions to include during TRN enrichment

        Returns
        -------
        df_enriched: ~pandas.DataFrame
            Table of statistically significant enrichments
        """

        enrichments = []

        if imodulons is None:
            imodulon_list = self.imodulon_names
        elif isinstance(imodulons, str) or isinstance(imodulons, int):
            imodulon_list = [imodulons]
        else:
            imodulon_list = imodulons

        if evidence is not None:
            if isinstance(evidence, str):
                evidences_to_use = [evidence]
            else:
                evidences_to_use = evidence
            trn_to_use = self.trn[self.trn["evidence"].isin(evidences_to_use)]
        else:
            trn_to_use = self.trn

        for imodulon in imodulon_list:
            gene_list = self.view_imodulon(imodulon).index
            df_enriched = compute_trn_enrichment(
                set(gene_list),
                set(self.gene_names),
                trn_to_use,
                max_regs=max_regs,
                fdr=fdr,
                method=method,
                force=force,
            )
            df_enriched["imodulon"] = imodulon
            enrichments.append(df_enriched)

        df_enriched = pd.concat(enrichments, axis=0, sort=True)

        # Set regulator as column instead of axis
        df_enriched.index.name = "regulator"
        df_enriched.reset_index(inplace=True)

        # Reorder columns
        df_enriched.rename({"gene_set_size": "imodulon_size"}, inplace=True, axis=1)
        col_order = [
            "imodulon",
            "regulator",
            "pvalue",
            "qvalue",
            "precision",
            "recall",
            "f1score",
            "TP",
            "regulon_size",
            "imodulon_size",
            "n_regs",
        ]
        df_enriched = df_enriched[col_order]

        # Sort by q-value
        df_enriched.sort_values(["imodulon", "qvalue", "n_regs"])

        if save:
            df_top_enrich = df_enriched.drop_duplicates("imodulon")
            self._update_imodulon_table(df_top_enrich.set_index("imodulon"))

        return df_enriched

    def compute_annotation_enrichment(
        self, annotation, column, imodulons=None, fdr=0.1
    ) -> pd.DataFrame:
        """
        Compare iModulons against a gene annotation table

        Parameters
        ----------
        annotation : ~pandas.DataFrame
            Table containing two columns: the gene locus tag, and its
            appropriate annotation
        column : str
            Name of the column containing the annotation
        imodulons : list or str or int
            Name of iModulon(s). If none given, compute enrichments for all
            iModulons (default: None)
        fdr : float
            False detection rate (default: 0.1)

        Returns
        -------
        DF_enriched: ~pandas.DataFrame
            Table of statistically significant enrichments
        """

        # TODO: write test function
        # TODO: Figure out save function

        enrichments = []

        if imodulons is None:
            imodulon_list = self.imodulon_names
        elif isinstance(imodulons, str):
            imodulon_list = [imodulons]
        else:
            imodulon_list = imodulons

        for imodulon in imodulon_list:
            gene_list = self.view_imodulon(imodulon).index
            df_enriched = compute_annotation_enrichment(
                set(gene_list),
                set(self.gene_names),
                column=column,
                annotation=annotation,
                fdr=fdr,
            )
            df_enriched["imodulon"] = imodulon
            enrichments.append(df_enriched)

        DF_enriched = pd.concat(enrichments, axis=0, sort=True)

        # Set annotation name as column instead of axis
        DF_enriched.index.name = column
        DF_enriched.reset_index(inplace=True)

        # Rename column
        DF_enriched.rename({"gene_set_size": "imodulon_size"}, inplace=True, axis=1)

        enrich_col = DF_enriched.columns[0]
        col_order = [
            "imodulon",
            enrich_col,
            "pvalue",
            "qvalue",
            "precision",
            "recall",
            "f1score",
            "TP",
            "target_set_size",
            "imodulon_size",
        ]
        DF_enriched = DF_enriched[col_order]
        return DF_enriched

    ######################################
    # Threshold properties and functions #
    ######################################

    @property
    def dagostino_cutoff(self):
        """Get D'agostino cutoff"""
        return self._dagostino_cutoff

    @property
    def cutoff_optimized(self):
        return self._cutoff_optimized

    @property
    def thresholds(self):
        """Get thresholds"""
        return self._thresholds

    @thresholds.setter
    def thresholds(self, new_thresholds):
        """Set thresholds"""
        new_thresh_len = len(new_thresholds)
        imod_names_len = len(self._imodulon_names)
        if new_thresh_len != imod_names_len:
            raise ValueError(
                "new_threshold has {:d} elements, but should "
                "have {:d} elements".format(new_thresh_len, imod_names_len)
            )

        # fix json peculiarity of saving int dict keys as string
        thresh_copy = new_thresholds.copy()
        for key in thresh_copy.keys():
            # Could this be replaced with a try/except clause?
            if isinstance(key, str) and all([char.isdigit() for char in key]):
                new_thresholds.update({int(key): new_thresholds.pop(key)})

        self._thresholds = new_thresholds
        self._cutoff_optimized = False

    def change_threshold(self, imodulon, value):
        """
        Set threshold for an iModulon

        Parameters
        ----------
        imodulon : int or str
            Name of iModulon
        value : float
            New threshold

        Returns
        -------
        None: None
        """

        self._thresholds[imodulon] = value
        self._cutoff_optimized = False

    def recompute_thresholds(self, dagostino_cutoff):
        """
        Re-computes iModulon thresholds using a new D'Agostino cutoff

        Parameters
        ----------
        dagostino_cutoff : int
            New D'agostino cutoff statistic

        Returns
        -------
        None: None
        """
        self._update_thresholds(dagostino_cutoff)
        self._cutoff_optimized = False

    def _update_thresholds(self, dagostino_cutoff: int):
        self._thresholds = {
            k: compute_threshold(self._m[k], dagostino_cutoff)
            for k in self._imodulon_names
        }
        self._dagostino_cutoff = dagostino_cutoff

    def reoptimize_thresholds(self, progress=True, plot=True):
        """
        Re-optimizes the D'Agostino statistic cutoff for defining iModulon
        thresholds if the TRN has been updated

        Parameters
        ----------
        progress : bool
            Show a progress bar (default: True)
        plot : bool
            Show the sensitivity analysis plot (default: True)
        Returns
        -------
        int
            New D'agostino cutoff
        """

        if self.trn.empty:
            raise ValueError(
                "D'agostino cutoff cannot be optimized if no TRN is "
                "provided. Use ica_data.compute_kmeans_thresholds() "
                "instead."
            )

        if not self._cutoff_optimized:
            new_cutoff = self._optimize_dagostino_cutoff(progress, plot)
            self._update_thresholds(new_cutoff)
            self._cutoff_optimized = True
        else:
            print(
                "Cutoff already optimized, and no new TRN data provided. "
                "Re-optimization will return same cutoff."
            )
        return self.dagostino_cutoff

    def _optimize_dagostino_cutoff(self, progress, plot):
        """
        Computes an abridged version of the TRN enrichments for the 20
        highest-weighted genes in order to determine a global minimum
        for the D'Agostino cutoff ultimately used to threshold and
        define the genes "in" an iModulon

        Parameters
        ----------
        progress : bool
            Show a progress bar (default: True)
        plot : bool
            Show the sensitivity analysis plot (default: True)
        Returns
        -------
        int
            New D'agostino cutoff
        """

        # prepare a DataFrame of the best single-TF enrichments for the
        # top 20 genes in each component
        top_enrichments = []
        all_genes = list(self.M.index)
        for imod in self.M.columns:

            genes_top20 = list(abs(self.M[imod]).sort_values().iloc[-20:].index)
            imod_enrichment_df = compute_trn_enrichment(
                set(genes_top20), set(all_genes), self.trn, max_regs=1
            )

            # compute_trn_enrichment is being hijacked a bit; we want
            # the index to be components, not the enriched TFs
            imod_enrichment_df["TF"] = imod_enrichment_df.index
            imod_enrichment_df["component"] = imod

            if not imod_enrichment_df.empty:
                # take the best single-TF enrichment row (by q-value)
                top_enrichment = imod_enrichment_df.sort_values(by="qvalue").iloc[0, :]
                top_enrichments.append(top_enrichment)

        # perform a sensitivity analysis to determine threshold effects
        # on precision/recall overall
        cutoffs_to_try = np.arange(50, 2000, 50)
        f1_scores = []

        if progress:
            iterator = tqdm(cutoffs_to_try)
        else:
            iterator = cutoffs_to_try

        for cutoff in iterator:
            cutoff_f1_scores = []
            for enrich_row in top_enrichments:
                # for this enrichment row, get all the genes regulated
                # by the regulator chosen above
                regulon_genes = list(
                    self.trn[self.trn["regulator"] == enrich_row["TF"]].gene_id
                )

                # compute the weighting threshold based on this cutoff to try
                thresh = compute_threshold(self.M[enrich_row["component"]], cutoff)
                component_genes = self.M[
                    abs(self.M[enrich_row["component"]]) > thresh
                ].index.tolist()

                # Compute the contingency table (aka confusion matrix)
                # for overlap between the regulon and iM genes
                ((tp, fp), (fn, tn)) = contingency(
                    set(regulon_genes), component_genes, set(all_genes)
                )

                # Calculate F1 score for one regulator-component pair
                # and add it to the running list for this cutoff
                precision = np.true_divide(tp, tp + fp) if tp > 0 else 0
                recall = np.true_divide(tp, tp + fn) if tp > 0 else 0
                f1_score = (
                    (2 * precision * recall) / (precision + recall) if tp > 0 else 0
                )
                cutoff_f1_scores.append(f1_score)

            # Get mean of F1 score for this potential cutoff
            f1_scores.append(np.mean(cutoff_f1_scores))

        # extract the best cutoff
        best_cutoff = cutoffs_to_try[np.argmax(f1_scores)]

        if plot:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_xlabel("D'agostino Test Statistic", fontsize=14)
            ax.set_ylabel("Mean F1 score")
            ax.plot(cutoffs_to_try, f1_scores)
            ax.scatter([best_cutoff], [max(f1_scores)], color="r")

        return int(best_cutoff)

    def _kmeans_cluster(self, imodulon):
        data = self.M[imodulon]
        model = KMeans(n_clusters=3, random_state=1)
        model.fit(abs(data).values.reshape(-1, 1))

        df = pd.DataFrame(abs(data))
        df["cluster"] = model.labels_

        # Get top two clusters
        counts = df.cluster.value_counts().sort_values(ascending=True)
        idx1 = counts.index[0]
        idx2 = counts.index[1]
        clust1 = df[df.cluster == idx1]
        clust2 = df[df.cluster == idx2]

        # Get midpoint between lowest iModulon gene and highest insignificant
        # gene
        threshold = np.mean([clust1[imodulon].min(), clust2[imodulon].max()])
        return threshold

    def compute_kmeans_thresholds(self):
        """
        Computes iModulon thresholds using K-means clustering

        Returns
        -------
        None: None
        """

        self._thresholds = {k: self._kmeans_cluster(k) for k in self._imodulon_names}

    def copy(self):
        """
        Make a deep copy of an IcaData object

        Returns
        -------
        IcaData: ~pymodulon.core.IcaData
            Copy of IcaData object
        """

        return copy.deepcopy(self)

    def imodulons_with(self, gene):
        """
        Lists the iModulons containing a gene

        Parameters
        ----------
        gene : str
            Gene name or locus tag

        Returns
        -------
        list
            List of iModulons containing the gene
        """

        # Check that gene exists
        if gene not in self.X.index:
            gene = self.name2num(gene)

        return self.M.columns[self.M_binarized.loc[gene] == 1].to_list()

    def name2num(self, gene):
        """
        Convert a gene name to the locus tag

        Parameters
        ----------
        gene : list or str
            Gene name or list of gene names

        Returns
        -------
        final_list : list or str
            Locus tag or list of locus tags
        """

        gene_table = self.gene_table
        if "gene_name" not in gene_table.columns:
            raise ValueError('Gene table does not contain "gene_name" column.')

        if isinstance(gene, str):
            gene_list = [gene]
        else:
            gene_list = gene

        final_list = []
        for g in gene_list:
            g_names = gene_table.gene_name.apply(lambda x: x.casefold())
            loci = gene_table[g_names == g.casefold()].index

            # Ensure only one locus maps to this gene
            if len(loci) == 0:
                raise ValueError("Gene does not exist: {}".format(g))
            elif len(loci) > 1:
                logging.warning(
                    "Found multiple genes named {}. Only "
                    "reporting first locus tag".format(g)
                )

            final_list.append(loci[0])

        # Return string if string was given as input
        if isinstance(gene, str):
            return final_list[0]
        else:
            return final_list

    def num2name(self, gene):
        """
        Get the name of a gene from its locus tag

        Parameters
        ----------
        gene : list or str
            Locus tag or list of locus tags

        Returns
        -------
        result : list or str
            Gene name or list of gene names
        """

        result = self.gene_table.loc[gene].gene_name
        if isinstance(gene, list):
            return result.tolist()
        else:
            return result

    #########################
    # iModulonDB Properties #
    #########################

    @property
    def dataset_table(self):
        return self._dataset_table

    @dataset_table.setter
    def dataset_table(self, new_dst):
        if new_dst is None:
            # count some statistics
            num_genes = self._m.shape[0]
            num_samps = self._a.shape[1]
            num_ims = self._m.shape[1]
            if ("project" in self.sample_table.columns) and (
                "condition" in self.sample_table.columns
            ):
                num_conds = len(self.sample_table.groupby(["condition", "project"]))
            else:
                num_conds = "Unknown"

            # initialize dataset_table
            self._dataset_table = pd.Series(
                {
                    "Title": "New Dataset",
                    "Organism": "New Organism",
                    "Strain": "Unknown Strain",
                    "Number of Samples": num_samps,
                    "Number of Unique Conditions": num_conds,
                    "Number of Genes": num_genes,
                    "Number of iModulons": num_ims,
                }
            )
        elif isinstance(new_dst, dict):
            self._dataset_table = new_dst
        elif isinstance(new_dst, str):
            self._dataset_table = _check_dict(new_dst)
        else:
            raise ValueError(
                "New dataset must be None, a filename, a dictionary, "
                "or a JSON string"
            )

    @property
    def splash_table(self):
        return self._splash_table

    @splash_table.setter
    def splash_table(self, new_splash):

        if new_splash is None:
            new_splash = dict()

        if isinstance(new_splash, str):
            new_splash = _check_dict(new_splash)

        self._splash_table = new_splash

        default_splash_table = {
            "large_title": "New Dataset",
            "subtitle": "Unpublished study",
            "author": "Pymodulon User",
            "organism_folder": "new_org",
            "dataset_folder": "new_dataset",
        }
        for k, v in default_splash_table.items():
            if k not in new_splash:  # use what is provided, default for
                # what isn't
                self._splash_table[k] = v

    @property
    def link_database(self):
        return self._link_database

    @link_database.setter
    def link_database(self, new_db):
        if isinstance(new_db, str):
            self._link_database = new_db
        else:
            raise ValueError("link_database must be a string.")

    @property
    def gene_links(self):
        return self._gene_links

    @gene_links.setter
    def gene_links(self, new_links):

        if new_links is None:
            new_links = dict()

        if isinstance(new_links, str):
            new_links = _check_dict(new_links)

        """
        # uncomment this to be warned for unused gene links
        for gene in new_links.keys():
            if not(gene in self._m.index):
                warnings.warn('The gene %s has a link
                but is not in the M matrix.'%(gene))
        """
        self._gene_links = new_links
        for gene in set(self._m.index) - set(new_links.keys()):
            self._gene_links[gene] = np.nan

    @property
    def tf_links(self):
        return self._tf_links

    @tf_links.setter
    def tf_links(self, new_links):

        if new_links is None:
            new_links = dict()

        if isinstance(new_links, str):
            new_links = _check_dict(new_links)

        if not self.trn.empty:
            for tf in new_links.keys():
                if not (tf in list(self.trn.regulator)):
                    print("%s has a TF link but is not in the TRN" % tf)

        self._tf_links = new_links


class MotifInfo:
    def __init__(self, motifs, sites, cmd, file, matches=None):
        self._motifs = motifs
        self._sites = sites
        self._cmd = cmd
        self._file = file
        self._matches = matches

    def __repr__(self):
        if len(self.motifs) == 1:
            motif_str = "motif"
        else:
            motif_str = "motifs"

        if len(self.sites) == 1:
            site_str = "site"
        else:
            site_str = "sites"
        return (
            f"<MotifInfo with {len(self.motifs)} {motif_str} across "
            f"{sum(self.sites.site_seq.notnull())} {site_str}>"
        )

    @property
    def motifs(self):
        return self._motifs

    @property
    def sites(self):
        return self._sites

    @property
    def cmd(self):
        return self._cmd

    @property
    def file(self):
        return self._file

    @property
    def matches(self):
        return self._matches

    @matches.setter
    def matches(self, matches):
        self._matches = matches
