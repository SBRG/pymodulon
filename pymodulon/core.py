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
                 trn: Data = None, dagostino_cutoff: int = 550):
        """

        :param s_matrix: S matrix from ICA
        :param a_matrix: A matrix from ICA
        :param x_matrix: log-TPM expression values (not normalized to reference)
        :param gene_table: Table containing relevant gene information
        :param sample_table: Table containing relevant sample metadata
        :param imodulon_table: Table containing iModulon names and enrichments
        :param trn: Table containing transcriptional regulatory network links
        :param dagostino_cutoff: Cut-off value for iModulon threshold calculation (default: 550)
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

        # Convert column names of S to ints if possible
        try:
            s_matrix.columns = s_matrix.columns.astype(int)
        except TypeError:
            pass

        # Check that S and A matrices have identical iModulon names
        if s_matrix.columns.tolist() != a_matrix.index.tolist():
            raise ValueError('S and A matrices have different iModulon names')

        # Initialize sample and gene names
        self._gene_names = s_matrix.index.tolist()
        self._sample_names = a_matrix.columns.tolist()
        self._imod_names = s_matrix.columns.tolist()

        # Store S and A
        self._s = s_matrix
        self._a = a_matrix

        # Initialize thresholds
        self._thresholds = {k: compute_threshold(self._s[k], dagostino_cutoff) for k in self._imod_names}

        #################
        # Load X matrix #
        #################

        # Check X matrix
        if x_matrix is None:
            self._x = None
        else:
            self.X = x_matrix

        ####################
        # Load data tables #
        ####################

        self.gene_table = gene_table
        self.sample_table = sample_table
        self.imodulon_table = imodulon_table

        # Set TRN
        if trn is None:
            trn = pd.DataFrame()
            # TODO: Add TF info to gene table
        self.trn = trn

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
        if df.columns.tolist() != self.sample_names:
            raise ValueError('X and A matrices have different sample names')
        if df.index.tolist() != self.gene_names:
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
        if len(new_thresholds) != len(self._imod_names):
            raise ValueError('new_threshold has {:d} elements, but should have {:d} elements'.format(len(
                new_thresholds), len(self._imod_names)))
        if isinstance(new_thresholds, dict):
            self._thresholds = new_thresholds
        elif isinstance(new_thresholds, list):
            self._thresholds = dict(zip(self._imod_names, new_thresholds))
        else:
            raise TypeError('new_thresholds must be list or dict')

    def update_threshold(self, imodulon: ImodName, value):
        """ Set threshold for an iModulon
            name: Name of iModulon
            value: New threshold
        """
        self._thresholds[imodulon] = value

    # Gene, sample and iModulon name properties

    @property
    def imodulon_names(self):
        """ Get iModulon names """
        return self._imod_names

    @imodulon_names.setter
    def imodulon_names(self, new_names):

        # Check length of new_names
        if len(new_names) != len(self._imod_names):
            raise ValueError('new_names has {:d} elements, but should contain {:d} elements'.format(
                len(new_names), len(self._imod_names)))

        # Check for duplicates in new_names
        if len(new_names) != len(set(new_names)):
            raise ValueError('new_names contains duplicate names')

        self._imod_names = new_names

        # Rename S and A matrices
        self._s.columns = new_names
        self._a.index = new_names

        # Update threshold dict
        convert_dict = dict(zip(new_names, self._imod_names))
        self._thresholds = {new_names: self._thresholds[convert_dict[x]] for x in new_names}

        # TODO: Update imod_table

    @property
    def sample_names(self):
        return self._sample_names

    @property
    def gene_names(self):
        return self._gene_names

    # Gene, sample and iModulon tables
    # TODO: Add checking
    @property
    def gene_table(self):
        return self._gene_table

    @gene_table.setter
    def gene_table(self, new_table):
        self._gene_table = _check_table(new_table, self.gene_names, 'gene')

    @property
    def sample_table(self):
        return self._sample_table

    @sample_table.setter
    def sample_table(self, new_table):
        self._sample_table = _check_table(new_table, self.sample_names, 'sample')

    @property
    def imodulon_table(self):
        return self._imod_table

    @imodulon_table.setter
    def imodulon_table(self, new_table):
        self._imod_table = _check_table(new_table, self.imodulon_names, 'imodulon')

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
        gene_rows = self.gene_table.loc[in_imodulon]
        final_rows = pd.concat([gene_weights, gene_rows])

        return final_rows

    # TRN
    @property
    def trn(self):
        return self._trn

    @trn.setter
    def trn(self, new_trn):
        self._trn = new_trn

    # Enrichments
    def compute_regulon_enrichment(self, imodulon: ImodName, regulator: str):
        """
        Compare an iModulon against a regulon
        :param imodulon:
        :param regulator:
        :return: Pandas Series containing enrichment statistics
        """
        imod_genes = self.view_imodulon(imodulon)
        compute_regulon_enrichment(imod_genes, regulator, self.gene_names, self.trn)
        return

    def compute_trn_enrichment(self, fdr: float = 1e-5, max_regs: int = 1, save: bool = False) -> pd.DataFrame:
        """
        Compare iModulons against all regulons
        :param fdr: False detection rate (default: 1e-5)
        :param max_regs: Maximum number of regulators to include in complex regulon (default: 1)
        :param save: Save regulons with highest enrichment scores to the imodulon_table
        :return: Pandas Dataframe of statistically significant enrichments
        """
        pass
