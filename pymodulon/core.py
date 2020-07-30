from pymodulon.util import *


class IcaData(object):
    """ Class representation of all iModulon-related data
    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, s_matrix: pd.DataFrame, a_matrix: pd.DataFrame,
                 x_matrix=None, imodulons=None,
                 gene_table=None, sample_table=None,
                 imodulon_table=None, trn=None,
                 dagostino_cutoff=550):
        # TODO: Add type hinting
        """
        Required Args:
            s_matrix: S matrix from ICA
            a_matrix: A matrix from ICA
        Optional Args:
            x_matrix: log-TPM expression values (not normalized to reference)
            gene_table: Table containing relevant gene information
            sample_table: Table containing relevant sample metadata
            imodulons: List of iModulon names
            dagostino_cutoff: Cut-off value for iModulon threshold calculation (default: 550)
            TODO: Fill in
        """

        # Check S and A matrices
        if isinstance(s_matrix, pd.DataFrame) and isinstance(a_matrix, pd.DataFrame):

            # Convert column names of S to ints if necessary
            try:
                s_matrix.columns = s_matrix.columns.astype(int)
            except TypeError:
                pass

            if s_matrix.columns.tolist() != a_matrix.index.tolist():
                raise ValueError('S and A matrices have different iModulon names')

            # Initialize sample and gene names
            self._gene_names = s_matrix.index.tolist()
            self._sample_names = a_matrix.columns.tolist()
            self._imod_names = s_matrix.columns.tolist()

            # Store S and A
            self._s = s_matrix
            self._a = a_matrix
        else:
            raise TypeError('S and A must be pandas dataframes or file names')
            # TODO: Allow S and A to be filenames

        # Initialize thresholds
        self._thresholds = {k: compute_threshold(self._s[k], dagostino_cutoff) for k in self._imod_names}

        # Check X matrix [optional]
        if x_matrix is None:
            self._x = None
        elif isinstance(x_matrix, pd.DataFrame):
            if x_matrix.columns.tolist() != self._sample_names:
                raise ValueError('X and A matrices have different sample names')
            if x_matrix.index.tolist() != self._gene_names:
                print(x_matrix.index.tolist())
                print(self._gene_names)
                raise ValueError('X and S matrices have different gene names')

            # Store X
            self._x = x_matrix
        else:
            raise TypeError('X must be a pandas dataframe or filename')
            # TODO: Allow X to be filename
            # TODO: Move to X setter function?

        # Set iModulon names [optional]
        if isinstance(imodulons, list):
            self.imodulon_names = imodulons

        # Set gene, sample, and iModulon tables
        if gene_table is None:
            gene_table = pd.DataFrame(index=self._gene_names)
        self.gene_table = gene_table

        if sample_table is None:
            sample_table = pd.DataFrame(index=self._sample_names)
        self.sample_table = sample_table

        if imodulon_table is None:
            imodulon_table = pd.DataFrame(index=self._imod_names)
        self.imodulon_table = imodulon_table

        # Set TRN
        if trn is None:
            trn = pd.DataFrame()
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
        # Check X matrix columns and indices
        if isinstance(x_matrix, pd.DataFrame):
            if x_matrix.columns != self._a.columns:
                raise ValueError('X and A matrices have different sample names')
            if x_matrix.index != self._x.index:
                raise ValueError('X and S matrices have different gene loci')
        self._x = x_matrix

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
        self._gene_table = new_table

    @property
    def sample_table(self):
        return self._sample_table

    @sample_table.setter
    def sample_table(self, new_table):
        self._sample_table = new_table

    @property
    def imodulon_table(self):
        return self._imod_table

    @imodulon_table.setter
    def imodulon_table(self, new_table):
        self._imod_table = new_table

    # TRN
    @property
    def trn(self):
        return self._trn

    @trn.setter
    def trn(self, new_trn):
        self._trn = new_trn

    def compute_regulon_enrichment(self, imodulon: ImodName, regulator: str):
        pass

    def compute_trn_enrichment(self, fdr: float = 1e-5, max_regs: int = 1, save: bool = False) -> pd.DataFrame:
        """
        Compare iModulons against all regulons
        :param fdr: False detection rate (default: 1e-5)
        :param max_regs: Maximum number of regulators to include in complex regulon (default: 1)
        :param save: Save regulons with highest enrichment scores to the imodulon_table
        :return: Pandas Dataframe of statistically significant enrichments
        """
        pass
