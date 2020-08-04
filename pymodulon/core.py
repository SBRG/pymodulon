from typing import List, Union

from pymodulon.util import *
from pymodulon.enrichment import *
from pymodulon.util import _check_table


class IcaData(object):
    """ Class representation of all iModulon-related data
    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, s_matrix: Data, a_matrix: Data, x_matrix: Data = None,
                 gene_table: Data = None, sample_table: Data = None, imodulon_table: Data = None,
                 trn: Data = None, optimize_cutoff: bool = False, dagostino_cutoff: int = 550,
                 thresholds: Union[List[float]] = None):
        """

        :param s_matrix: S matrix from ICA
        :param a_matrix: A matrix from ICA
        :param x_matrix: log-TPM expression values (not normalized to reference)
        :param gene_table: Table containing relevant gene information
        :param sample_table: Table containing relevant sample metadata
        :param imodulon_table: Table containing iModulon names and enrichments
        :param trn: Table containing transcriptional regulatory network links
        :param optimize_cutoff: Indicates if the cutoff for iModulon thresholds should be optimized based on the
        provided TRN (if available). Optimizing thresholds may take a couple of minutes to complete.
        :param dagostino_cutoff: the cutoff value to use for the D'Agostino test for thresholding iModulon genes; this
        option will be overridden if optimize_cutoff is set to True
        :param thresholds: a list of pre-computed thresholds index-matched to the imodulons (columns of S); overrides
        all automatic optimization/computing of thresholds
        """

        #########################
        # Load S and A matrices #
        #########################

        # Type check S and A matrices
        if isinstance(s_matrix, str):
            s_matrix = pd.read_csv(s_matrix, index_col=0)
        elif not isinstance(s_matrix, pd.DataFrame):
            raise TypeError('s_matrix must be either a DataFrame or filename')

        if isinstance(a_matrix, str):
            a_matrix = pd.read_csv(a_matrix, index_col=0)
        elif not isinstance(a_matrix, pd.DataFrame):
            raise TypeError('a_matrix must be either a DataFrame or filename')

        # Convert column names of S to int if possible
        try:
            s_matrix.columns = s_matrix.columns.astype(int)
        except TypeError:
            pass

        # Check that S and A matrices have identical iModulon names
        if s_matrix.columns.tolist() != a_matrix.index.tolist():
            raise ValueError('S and A matrices have different iModulon names')

        # Ensure that S and A matrices have unique indices/columns
        if s_matrix.index.duplicated().any():
            raise ValueError('S matrix contains duplicate gene names')
        if a_matrix.columns.duplicated().any():
            raise ValueError('A matrix contains duplicate sample names')
        if s_matrix.columns.duplicated().any():
            raise ValueError('S and A matrices contain duplicate iModulon names')

        # Store S and A
        self._s = s_matrix
        self._a = a_matrix

        #################
        # Load X matrix #
        #################

        # Check X matrix
        if x_matrix is None:
            self._x = None
        else:
            self.X = x_matrix

        # Initialize sample and gene names
        self._gene_names = s_matrix.index.tolist()
        self.gene_table = gene_table
        self._sample_names = a_matrix.columns.tolist()
        self.sample_table = sample_table
        self._imodulon_names = s_matrix.columns.tolist()

        ############
        # Load TRN #
        ############
        if trn is None:
            df_trn = pd.DataFrame(columns=['regulator', 'regulator_id', 'gene_name', 'gene_id', 'effect', 'ev_level'])
        elif isinstance(trn, str):
            df_trn = pd.read_csv(trn)
        elif isinstance(trn, pd.DataFrame):
            df_trn = trn
        else:
            raise TypeError('TRN must either be a pandas DataFrame or filename')
        # Only include genes that are in S/X matrix
        self.trn = df_trn[df_trn.gene_id.isin(self.gene_names)]

        # Initialize thresholds either with or without optimization
        self.dagostino_cutoff = dagostino_cutoff
        self._cutoff_optimized = False
        if thresholds is None:
            if optimize_cutoff:
                if trn is None:
                    raise ValueError('Thresholds cannot be optimized if no TRN is provided.')
                else:
                    warn("Optimizing cutoff, may take 2-3 minutes...")
                    # this private function sets self.dagostino_cutoff internally
                    self._optimize_dagostino_cutoff()
                    # also set a private attribute to tell us if we've done this optimization; only reasonable to try
                    # it again if the user uploads a new TRN
                    self._cutoff_optimized = True
            self._thresholds = {k: compute_threshold(self._s[k], self.dagostino_cutoff) for k in self._imodulon_names}
        else:
            self.thresholds = thresholds

        ####################
        # Load data tables #
        ####################

        # this one is loaded here because it needs the thresholds
        self.imodulon_table = imodulon_table

    @property
    def S(self):
        """ Get S matrix """
        return self._s

    @property
    def A(self):
        """ Get A matrix """
        return self._a

    @property
    def X(self):
        """ Get X matrix """
        return self._x

    @X.setter
    def X(self, x_matrix):
        if isinstance(x_matrix, str):
            df = pd.read_csv(x_matrix, index_col=0)
        elif isinstance(x_matrix, pd.DataFrame):
            df = x_matrix
        else:
            raise TypeError('X must be a pandas DataFrame or filename')

        # Check that gene and sample names conform to S and A matrices
        if df.columns.tolist() != self.A.columns.tolist():
            raise ValueError('X and A matrices have different sample names')
        if df.index.tolist() != self.S.index.tolist():
            raise ValueError('X and S matrices have different gene names')

        # Set x_matrix
        self._x = df

    @X.deleter
    def X(self):
        # Delete X matrix
        del self._x

    # Thresholds property
    @property
    def thresholds(self):
        """ Get thresholds """
        return self._thresholds

    @thresholds.setter
    def thresholds(self, new_thresholds):
        """ Set thresholds """
        if len(new_thresholds) != len(self._imodulon_names):
            raise ValueError('new_threshold has {:d} elements, but should have {:d} elements'.format(len(
                new_thresholds), len(self._imodulon_names)))
        if isinstance(new_thresholds, dict):
            self._thresholds = new_thresholds
        elif isinstance(new_thresholds, list):
            self._thresholds = dict(zip(self._imodulon_names, new_thresholds))
        else:
            raise TypeError('new_thresholds must be list or dict')

    def update_threshold(self, imodulon: ImodName, value):
        """
        Set threshold for an iModulon
        :param imodulon: name of iModulon
        :param value: New threshold
        """
        self._thresholds[imodulon] = value

    # Gene, sample and iModulon name properties
    @property
    def imodulon_names(self):
        """ Get iModulon names """
        return self._imodulon_table.index.tolist()

    @property
    def sample_names(self) -> List:
        """ Get sample names """
        return self._sample_table.index.tolist()

    @property
    def gene_names(self) -> List:
        """ Get gene names """
        return self._gene_table.index.tolist()

    # Gene, sample and iModulon tables
    # TODO: Add checking
    @property
    def gene_table(self):
        return self._gene_table

    @gene_table.setter
    def gene_table(self, new_table):
        table = _check_table(new_table, self._gene_names, 'gene')
        self._gene_table = table

        # Update gene names
        names = table.index
        self._gene_names = names
        self._s.index = names
        if self._x is not None:
            self._x.index = names

    @property
    def sample_table(self):
        return self._sample_table

    @sample_table.setter
    def sample_table(self, new_table):
        table = _check_table(new_table, self._sample_names, 'sample')
        self._sample_table = table

        # Update sample names
        names = table.index
        self._sample_names = names
        self._a.columns = names
        if self._x is not None:
            self._x.columns = names

    @property
    def imodulon_table(self):
        return self._imodulon_table

    @imodulon_table.setter
    def imodulon_table(self, new_table):
        table = _check_table(new_table, self._imodulon_names, 'imodulon')
        self._imodulon_table = table
        self._update_imodulon_names(table.index)

    def _update_imodulon_names(self, new_names):
        # Update thresholds
        for old_name, new_name in zip(self._imodulon_names, new_names):
            self._thresholds[new_name] = self._thresholds.pop(old_name)

        # Update iModulon names
        self._imodulon_names = new_names
        self._a.index = new_names
        self._s.columns = new_names
        self._imodulon_table.index = new_names

    # Show enriched
    def view_imodulon(self, imodulon: ImodName):
        """
        View genes in an iModulon and relevant information about each gene
        :param imodulon: Name of iModulon
        :return: Pandas Dataframe showing iModulon gene information
        """
        # Find genes in iModulon
        in_imodulon = abs(self.S[imodulon]) > self.thresholds[imodulon]

        # Get gene weights information
        gene_weights = self.S.loc[in_imodulon, imodulon]
        gene_weights.name = 'gene_weight'
        gene_rows = self.gene_table.loc[in_imodulon]
        final_rows = pd.concat([gene_weights, gene_rows], axis=1)

        return final_rows

    def rename_imodulons(self, name_dict: Dict[ImodName, ImodName]) -> None:
        """
        Rename an iModulon
        :param name_dict: Dictionary mapping old iModulon names to new names (e.g. {old_name:new_name})
        """

        new_names = [name_dict[name] if name in name_dict.keys() else name for name in self.imodulon_names]
        self._update_imodulon_names(new_names)

    # TRN
    @property
    def trn(self):
        return self._trn

    @trn.setter
    def trn(self, new_trn):
        # Only include genes that are in S/X matrix
        self._trn = new_trn[new_trn.gene_id.isin(self.gene_names)]
        # make a note that our cutoffs are no longer optimized since the TRN has changed
        self._cutoff_optimized = False

    # Enrichments
    def compute_regulon_enrichment(self, imodulon: ImodName, regulator: str, save: bool = False):
        """
        Compare an iModulon against a regulon. (Note: q-values cannot be computed for single enrichments)
        :param imodulon: Name of iModulon
        :param regulator:
        :param save:
        :return: Pandas Series containing enrichment statistics
        """
        imod_genes = self.view_imodulon(imodulon).index
        enrich = compute_regulon_enrichment(imod_genes, regulator, self.gene_names, self.trn)
        enrich.rename({'gene_set_size': 'imodulon_size'}, inplace=True)
        if save:
            table = self.imodulon_table
            for key, value in enrich.items():
                table.loc[imodulon, key] = value
            self.imodulon_table = table
        return enrich

    def compute_trn_enrichment(self, imodulons: Union[ImodName, List[ImodName]] = None, fdr: float = 1e-5,
                               max_regs: int = 1, save: bool = False, method: str = 'both',
                               force: bool = False) -> pd.DataFrame:
        """
        Compare iModulons against all regulons
        :param imodulons: Name of iModulon(s). If none given, compute enrichments for all iModulons
        :param fdr: False detection rate (default: 1e-5)
        :param max_regs: Maximum number of regulators to include in complex regulon (default: 1)
        :param save: Save regulons with highest enrichment scores to the imodulon_table
        :param method: How to combine regulons. 'or' computes enrichment against union of regulons, \
    'and' computes enrichment against intersection of regulons, and 'both' performs both tests (default: 'both')
    intersection of regulons
        :param force: Allows computation of >2 regulators
        :return: Pandas Dataframe of statistically significant enrichments
        """
        enrichments = []

        if imodulons is None:
            imodulon_list = self.imodulon_names
        elif isinstance(imodulons, str):
            imodulon_list = [imodulons]
        else:
            imodulon_list = imodulons

        for imodulon in imodulon_list:
            gene_list = self.view_imodulon(imodulon).index
            df_enriched = compute_trn_enrichment(gene_list, self.gene_names, self.trn, max_regs=max_regs, fdr=fdr,
                                                 method=method, force=force)
            df_enriched['imodulon'] = imodulon
            enrichments.append(df_enriched)

        df_enriched = pd.concat(enrichments, axis=0)

        # Set regulator as column instead of axis
        df_enriched.index.name = 'regulator'
        df_enriched.reset_index(inplace=True)

        # Reorder columns
        df_enriched.rename({'gene_set_size': 'imodulon_size'}, inplace=True, axis=1)
        col_order = ['imodulon', 'regulator', 'pvalue', 'qvalue', 'precision', 'recall', 'f1score',
                     'TP', 'regulon_size', 'imodulon_size', 'n_regs']
        df_enriched = df_enriched[col_order]

        if save:
            df_top_enrich = df_enriched.sort_values(['imodulon', 'qvalue', 'n_regs']).drop_duplicates('imodulon')
            self._update_imodulon_table(df_top_enrich.set_index('imodulon'))

        return df_enriched

    def _update_imodulon_table(self, enrichment):
        """
        Update iModulon table given new iModulon enrichments
        :param enrichment: Pandas series or dataframe containing an iModulon enrichment
        """
        if isinstance(enrichment, pd.Series):
            enrichment = pd.DataFrame(enrichment)
        keep_rows = self.imodulon_table[~self.imodulon_table.index.isin(enrichment.index)]
        keep_cols = self.imodulon_table.loc[enrichment.index, set(self.imodulon_table.columns) -
                                            set(enrichment.columns)]
        df_top_enrich = pd.concat([enrichment, keep_cols], axis=1)
        new_table = pd.concat([keep_rows, df_top_enrich])
        self.imodulon_table = new_table.reindex(self.imodulon_names)

    def reoptimize_thresholds(self):
        """
        Re-optimizes the D'Agostino statistic cutoff for defining iModulon thresholds if the trn has been updated
        """
        if not self._cutoff_optimized:
            self._optimize_dagostino_cutoff()
            self._cutoff_optimized = True
            self._thresholds = {k: compute_threshold(self._s[k], self.dagostino_cutoff) for k in self._imodulon_names}
        else:
            print('Cutoff already optimized, and no new TRN data provided. Reoptimization will return same cutoff.')

    def _optimize_dagostino_cutoff(self):
        """
        Computes an abridged version of the TRN enrichments for the 20 highest-weighted genes in order to determine
        a global minimum for the D'Agostino cutoff ultimately used to threshold and define the genes "in" an iModulon
        """

        # prepare a DataFrame of the best single-TF enrichments for the top 20 genes in each component
        top_enrichments = []
        all_genes = list(self.S.index)
        for imod in self.S.columns:

            genes_top20 = list(abs(self.S[imod]).sort_values().iloc[-20:].index)
            imod_enrichment_df = compute_trn_enrichment(genes_top20, all_genes, self.trn, max_regs=1)

            # compute_trn_enrichment is being hijacked a bit; we want the index to be components, not the enriched TFs
            imod_enrichment_df['TF'] = imod_enrichment_df.index
            imod_enrichment_df['component'] = imod
            if not imod_enrichment_df.empty:
                # take the best single-TF enrichment row according to p-value
                top_enrichment = imod_enrichment_df.sort_values(by='pvalue', ascending=False).iloc[0, :]
                top_enrichments.append(top_enrichment)

        # perform a sensitivity analysis to determine threshold effects on precision/recall overall
        cutoffs_to_try = np.arange(300, 2000, 50)
        f1_scores = []
        for cutoff in cutoffs_to_try:
            cutoff_f1_scores = []
            for enrich_row in top_enrichments:

                # for this enrichment row, get all the genes regulated by the regulator chosen above
                regulon_genes = list(self.trn[self.trn['regulator'] == enrich_row['TF']].gene_id)

                # compute the weighting threshold based on this cutoff to try
                thresh = compute_threshold(self.S[enrich_row['component']], cutoff)
                component_genes = list(self.S[abs(self.S[enrich_row['component']]) > thresh].index)

                # Compute the contingency table (aka confusion matrix) for overlap between the regulon and iM genes
                ((tp, fp), (fn, tn)) = contingency(regulon_genes, component_genes, all_genes)

                # Calculate F1 score for one regulator-component pair and add it to the running list for this cutoff
                precision = np.true_divide(tp, tp + fp)
                recall = np.true_divide(tp, tp + fn)
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                cutoff_f1_scores.append(f1_score)

            # Get mean of F1 score for this potential cutoff
            f1_scores.append(np.mean(cutoff_f1_scores))

        # extract the best cutoff and set it as the cutoff to use
        self.dagostino_cutoff = cutoffs_to_try[np.argmax(f1_scores)]
