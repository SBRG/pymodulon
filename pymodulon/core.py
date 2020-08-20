from pymodulon.enrichment import *
from pymodulon.util import _check_table, compute_threshold, Data, ImodNameList
from typing import Optional, Mapping, Iterable
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm


class IcaData(object):
    """ Class representation of all iModulon-related data
    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, M: Data, A: Data, X: Optional[Data] = None,
                 gene_table: Optional[Data] = None,
                 sample_table: Optional[Data] = None,
                 imodulon_table: Optional[Data] = None,
                 trn: Optional[Data] = None,
                 optimize_cutoff: bool = False,
                 dagostino_cutoff: int = 550,
                 thresholds: Optional[Union[Mapping[ImodName, float], Iterable]] = None):
        """

        :param M: S matrix from ICA
        :param A: A matrix from ICA
        :param X: log-TPM expression values (not normalized to reference)
        :param gene_table: Table containing relevant gene information
        :param sample_table: Table containing relevant sample metadata
        :param imodulon_table: Table containing iModulon names and enrichments
        :param trn: Table containing transcriptional regulatory network links
        :param optimize_cutoff: Indicates if the cutoff for iModulon thresholds
            should be optimized based on the provided TRN (if available).
            Optimizing thresholds may take a couple of minutes to complete.
        :param dagostino_cutoff: the cutoff value to use for the D'Agostino
            test for thresholding iModulon genes; this option will be
            overridden if optimize_cutoff is set to True
        :param thresholds: a list of pre-computed thresholds index-matched to
            the imodulons (columns of S); overrides all automatic
            optimization/computing of thresholds
        """

        #########################
        # Load M and A matrices #
        #########################

        M = _check_table(M, 'M')
        A = _check_table(A, 'A')

        # Convert column names of M to int if possible
        try:
            M.columns = M.columns.astype(int)
        except TypeError:
            pass

        # Check that M and A matrices have identical iModulon names
        if M.columns.tolist() != A.index.tolist():
            raise ValueError('M and A matrices have different iModulon names')

        # Ensure that M and A matrices have unique indices/columns
        if M.index.duplicated().any():
            raise ValueError('M matrix contains duplicate gene names')
        if A.columns.duplicated().any():
            raise ValueError('A matrix contains duplicate sample names')
        if M.columns.duplicated().any():
            raise ValueError('M and A matrices contain '
                             'duplicate iModulon names')

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
        self._dagostino_cutoff = dagostino_cutoff
        self._cutoff_optimized = False
        if thresholds is None:
            if optimize_cutoff:
                if trn is None:
                    raise ValueError('Thresholds cannot be optimized '
                                     'if no TRN is provided.')
                else:
                    warn('Optimizing iModulon thresholds, '
                         'may take 2-3 minutes...')
                    # this function sets self.dagostino_cutoff internally
                    self.reoptimize_thresholds(progress=False, plot=False)
                    # also sets an attribute to tell us if we've done
                    # this optimization; only reasonable to try it
                    # again if the user uploads a new TRN
            else:
                self.recompute_thresholds(self.dagostino_cutoff)
        else:
            self.thresholds = thresholds


    @property
    def M(self):
        """ Get M matrix """
        return self._m

    @property
    def M_binarized(self):
        """ Get binarized version of M matrix based on current thresholds """
        m_binarized = pd.DataFrame().reindex_like(self.M)
        m_binarized[:] = 0
        for imodulon in m_binarized.columns:
            in_imodulon = abs(self.M[imodulon]) > self.thresholds[imodulon]
            m_binarized.loc[in_imodulon, imodulon] = 1
        return m_binarized

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
            sep = '\t' if x_matrix.endswith('.tsv') else ','
            df = pd.read_csv(x_matrix, index_col=0, sep=sep)
        elif isinstance(x_matrix, pd.DataFrame):
            df = x_matrix
        else:
            raise TypeError('X must be a pandas DataFrame or filename')

        # Check that gene and sample names conform to M and A matrices
        if df.columns.tolist() != self.A.columns.tolist():
            raise ValueError('X and A matrices have different sample names')
        if df.index.tolist() != self.M.index.tolist():
            raise ValueError('X and M matrices have different gene names')

        # Set x matrix
        self._x = df

    @X.deleter
    def X(self):
        # Delete X matrix
        del self._x

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
    @property
    def gene_table(self):
        return self._gene_table

    @gene_table.setter
    def gene_table(self, new_table):
        table = _check_table(new_table, 'gene', self._gene_names)
        self._gene_table = table

        # Update gene names
        names = table.index
        self._gene_names = names
        self._m.index = names
        if self._x is not None:
            self._x.index = names

    @property
    def sample_table(self):
        return self._sample_table

    @sample_table.setter
    def sample_table(self, new_table):
        table = _check_table(new_table, 'sample', self._sample_names)
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
        table = _check_table(new_table, 'imodulon', self._imodulon_names)
        self._imodulon_table = table

    def _update_imodulon_names(self, new_names):
        # Update thresholds
        for old_name, new_name in zip(self._imodulon_names, new_names):
            self._thresholds[new_name] = self._thresholds.pop(old_name)

        # Update iModulon names
        self._imodulon_names = new_names
        self._a.index = new_names
        self._m.columns = new_names
        self._imodulon_table.index = new_names

    def rename_imodulons(self, name_dict: Mapping[ImodName, ImodName] = None,
                         column=None) -> None:
        """
        Rename an iModulon.
        :param name_dict: Dictionary mapping old iModulon names to new
            names (e.g. {old_name:new_name})
        :param column: Uses a column from the iModulon table to rename iModulons
        """

        # Rename using the column parameter if given
        if column is not None:
            if column in self.imodulon_table.columns:
                new_names = self.imodulon_table[column]
                self._imodulon_table = self._imodulon_table.drop(column, axis=1)
            else:
                raise ValueError('{} is not a column in the iModulon table'.format(column))
        else:
            new_names = self.imodulon_names

        # Use dictionary to rename iModulons
        if name_dict is not None:
            new_names = [name_dict[name] if name in name_dict.keys() else name
                         for name in new_names]
        self._update_imodulon_names(new_names)

    # Show enriched
    def view_imodulon(self, imodulon: ImodName):
        """
        View genes in an iModulon and relevant information about each gene

        :param imodulon: Name of iModulon
        :return: Pandas Dataframe showing iModulon gene information
        """
        # Find genes in iModulon
        in_imodulon = abs(self.M[imodulon]) > self.thresholds[imodulon]

        # Get gene weights information
        gene_weights = self.M.loc[in_imodulon, imodulon]
        gene_weights.name = 'gene_weight'
        gene_rows = self.gene_table.loc[in_imodulon]
        final_rows = pd.concat([gene_weights, gene_rows], axis=1)

        return final_rows

    def find_single_gene_imodulons(self, save: bool = False) -> List[ImodName]:
        """
        A simple function that returns the names of all likely single-gene
        iModulons. Checks if the largest iModulon gene weight is more
        than twice the weight of the second highest iModulon gene weight.

        :return: List of the current single-gene iModulon names
        """
        single_genes_imodulons = []
        for imodulon in self.imodulon_names:
            sorted_weights = abs(self.M[imodulon]).sort_values(ascending=False)
            if sorted_weights.iloc[0] > 2 * sorted_weights.iloc[1]:
                single_genes_imodulons.append(imodulon)
                if save:
                    self.imodulon_table.loc[imodulon, 'single_gene'] = True
        return single_genes_imodulons

    # TRN
    @property
    def trn(self):
        return self._trn

    @trn.setter
    def trn(self, new_trn):
        if isinstance(new_trn, str):
            new_trn = pd.read_csv(new_trn).reset_index(drop=True)

        self._trn = _check_table(new_trn, 'TRN')
        if not self._trn.empty:
            # Only include genes that are in S/X matrix
            self._trn = new_trn[new_trn.gene_id.isin(self.gene_names)]

            # Save regulator information to gene table
            reg_dict = {}
            for name, group in self._trn.groupby('gene_id'):
                reg_dict[name] = ','.join(group.regulator)
            self._gene_table['regulator'] = pd.Series(reg_dict).reindex(self.gene_names)

        # make a note that our cutoffs are no longer optimized since the TRN has changed
        self._cutoff_optimized = False

    ###############
    # Enrichments #
    ###############

    def compute_regulon_enrichment(self, imodulon: ImodName, regulator: str,
                                   save: bool = False):
        """
        Compare an iModulon against a regulon. (Note: q-values cannot be
        computed for single enrichments)

        :param imodulon: Name of iModulon
        :param regulator: Complex regulon, where "/" uses genes in any
            regulon and "+" uses genes in all regulons
        :param save: Save enrichment score to the imodulon_table
        :return: Pandas Series containing enrichment statistics
        """
        imod_genes = self.view_imodulon(imodulon).index
        enrich = compute_regulon_enrichment(imod_genes, regulator,
                                            self.gene_names, self.trn)
        enrich.rename({'gene_set_size': 'imodulon_size'}, inplace=True)
        if save:
            table = self.imodulon_table
            for key, value in enrich.items():
                table.loc[imodulon, key] = value
                table.loc[imodulon, 'regulator'] = enrich.name
            self.imodulon_table = table
        return enrich

    def compute_trn_enrichment(self, imodulons: Optional[ImodNameList] = None,
                               fdr: float = 1e-5, max_regs: int = 1,
                               save: bool = False, method: str = 'both',
                               force: bool = False) -> pd.DataFrame:
        """
        Compare iModulons against all regulons

        :param imodulons: Name of iModulon(s). If none given, compute
            enrichments for all iModulons
        :param fdr: False detection rate (default: 1e-5)
        :param max_regs: Maximum number of regulators to include in
            complex regulon (default: 1)
        :param save: Save regulons with highest enrichment scores to
            the imodulon_table
        :param method: How to combine regulons.
            'or' computes enrichment against union of regulons,
            'and' computes enrichment against intersection of regulons, and
            'both' performs both tests (default: 'both')
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
            df_enriched = compute_trn_enrichment(gene_list, self.gene_names,
                                                 self.trn, max_regs=max_regs,
                                                 fdr=fdr, method=method,
                                                 force=force)
            df_enriched['imodulon'] = imodulon
            enrichments.append(df_enriched)

        df_enriched = pd.concat(enrichments, axis=0)

        # Set regulator as column instead of axis
        df_enriched.index.name = 'regulator'
        df_enriched.reset_index(inplace=True)

        # Reorder columns
        df_enriched.rename({'gene_set_size': 'imodulon_size'},
                           inplace=True, axis=1)
        col_order = ['imodulon', 'regulator', 'pvalue', 'qvalue', 'precision',
                     'recall', 'f1score', 'TP', 'regulon_size',
                     'imodulon_size', 'n_regs']
        df_enriched = df_enriched[col_order]

        if save:
            df_top_enrich = df_enriched.sort_values(
                ['imodulon', 'qvalue', 'n_regs']).drop_duplicates('imodulon')
            self._update_imodulon_table(df_top_enrich.set_index('imodulon'))

        return df_enriched

    def _update_imodulon_table(self, enrichment):
        """
        Update iModulon table given new iModulon enrichments

        :param enrichment: Pandas series or dataframe containing an
            iModulon enrichment
        """
        if isinstance(enrichment, pd.Series):
            enrichment = pd.DataFrame(enrichment)
        keep_rows = self.imodulon_table[~self.imodulon_table.index.isin(
            enrichment.index)]
        keep_cols = self.imodulon_table.loc[enrichment.index,
                                            set(self.imodulon_table.columns)
                                            - set(enrichment.columns)]
        df_top_enrich = pd.concat([enrichment, keep_cols], axis=1)
        new_table = pd.concat([keep_rows, df_top_enrich])

        # Reorder columns
        col_order = enrichment.columns.tolist() + keep_cols.columns.tolist()
        new_table = new_table[col_order]

        # Reorder rows
        new_table = new_table.reindex(self.imodulon_names)

        self.imodulon_table = new_table

    ######################################
    # Threshold properties and functions #
    ######################################

    @property
    def dagostino_cutoff(self):
        return self._dagostino_cutoff

    @property
    def thresholds(self):
        """ Get thresholds """
        return self._thresholds

    @thresholds.setter
    def thresholds(self, new_thresholds):
        """ Set thresholds """
        new_thresh_len = len(new_thresholds)
        imod_names_len = len(self._imodulon_names)
        if new_thresh_len != imod_names_len:
            raise ValueError('new_threshold has {:d} elements, but should '
                             'have {:d} elements'.format(new_thresh_len,
                                                         imod_names_len))
        if isinstance(new_thresholds, dict):
            # fix json peculiarity of saving int dict keys as string
            thresh_copy = new_thresholds.copy()
            for key in thresh_copy.keys():
                if isinstance(key, str) and all([char.isdigit()
                                                 for char in key]):
                    new_thresholds.update({int(key): new_thresholds.pop(key)})

            self._thresholds = new_thresholds
        elif isinstance(new_thresholds, list):
            self._thresholds = dict(zip(self._imodulon_names, new_thresholds))
        else:
            raise TypeError('new_thresholds must be list or dict')

    def change_threshold(self, imodulon: ImodName, value):
        """
        Set threshold for an iModulon

        :param imodulon: name of iModulon
        :param value: New threshold
        """
        self._thresholds[imodulon] = value
        self._cutoff_optimized = False

    def recompute_thresholds(self, dagostino_cutoff: int):
        """
        Re-computes iModulon thresholds using a new D'Agostino cutoff

        :param dagostino_cutoff: Value to use for the D'Agostino test
            to determine iModulon thresholds
        :return: None
        """
        self._update_thresholds(dagostino_cutoff)
        self._cutoff_optimized = False

    def _update_thresholds(self, dagostino_cutoff: int):
        self._thresholds = {k: compute_threshold(self._m[k], dagostino_cutoff)
                            for k in self._imodulon_names}
        self._dagostino_cutoff = dagostino_cutoff

    def reoptimize_thresholds(self, progress=True, plot=True):
        """
        Re-optimizes the D'Agostino statistic cutoff for defining iModulon
        thresholds if the trn has been updated
        :param progress: Show a progress bar (default: True)
        :param plot: Show the sensitivity analysis plot (default: True)
        """
        if not self._cutoff_optimized:
            self._optimize_dagostino_cutoff(progress, plot)
            self._cutoff_optimized = True
            self._update_thresholds(self.dagostino_cutoff)
        else:
            print('Cutoff already optimized, and no new TRN data provided. '
                  'Re-optimization will return same cutoff.')

    def _optimize_dagostino_cutoff(self, progress, plot):
        """
        Computes an abridged version of the TRN enrichments for the 20
        highest-weighted genes in order to determine a global minimum
        for the D'Agostino cutoff ultimately used to threshold and
        define the genes "in" an iModulon
        :param progress: Show a progress bar (default: True)
        :param plot: Show the sensitivity analysis plot (default: True)
        """

        # prepare a DataFrame of the best single-TF enrichments for the
        # top 20 genes in each component
        top_enrichments = []
        all_genes = list(self.M.index)
        for imod in self.M.columns:

            genes_top20 = list(
                abs(self.M[imod]).sort_values().iloc[-20:].index)
            imod_enrichment_df = compute_trn_enrichment(genes_top20, all_genes,
                                                        self.trn, max_regs=1)

            # compute_trn_enrichment is being hijacked a bit; we want
            # the index to be components, not the enriched TFs
            imod_enrichment_df['TF'] = imod_enrichment_df.index
            imod_enrichment_df['component'] = imod

            if not imod_enrichment_df.empty:
                # take the best single-TF enrichment row (by q-value)
                top_enrichment = imod_enrichment_df.sort_values(
                    by='qvalue').iloc[0, :]
                top_enrichments.append(top_enrichment)

        # perform a sensitivity analysis to determine threshold effects
        # on precision/recall overall
        cutoffs_to_try = np.arange(300, 2000, 50)
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
                regulon_genes = list(self.trn[self.trn['regulator']
                                              == enrich_row['TF']].gene_id)

                # compute the weighting threshold based on this cutoff to try
                thresh = compute_threshold(self.M[enrich_row['component']],
                                           cutoff)
                component_genes = self.M[abs(self.M[enrich_row['component']])
                                         > thresh].index.tolist()

                # Compute the contingency table (aka confusion matrix)
                # for overlap between the regulon and iM genes
                ((tp, fp), (fn, tn)) = contingency(regulon_genes,
                                                   component_genes,
                                                   all_genes)

                # Calculate F1 score for one regulator-component pair
                # and add it to the running list for this cutoff
                precision = np.true_divide(tp, tp + fp) if tp > 0 else 0
                recall = np.true_divide(tp, tp + fn) if tp > 0 else 0
                f1_score = (2 * precision * recall) / (precision + recall) \
                    if tp > 0 else 0
                cutoff_f1_scores.append(f1_score)

            # Get mean of F1 score for this potential cutoff
            f1_scores.append(np.mean(cutoff_f1_scores))

        # extract the best cutoff and set it as the cutoff to use
        best_cutoff = cutoffs_to_try[np.argmax(f1_scores)]
        self._dagostino_cutoff = best_cutoff

        if plot:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.set_xlabel("D'agostino Test Statistic", fontsize=14)
            ax.set_ylabel("Mean F1 score")
            ax.plot(cutoffs_to_try, f1_scores)
            ax.scatter([best_cutoff], [max(f1_scores)], color='r')

        return best_cutoff
