import sys
import unittest

import pandas as pd

sys.path.append("/home/sbrg/Desktop/pymodulon/")
from pymodulon.compare import *


class MyTestCase(unittest.TestCase):
    def test_compare_ica(self):
        S2 = pd.read_csv("data/mtb_S.csv", index_col=0)
        S1 = pd.read_csv("data/ecoli_S.csv", index_col=0)
        ortho_dir = (
            "/home/sbrg/Desktop/modulome/data/organism_compare/"
            "bbh_csv/eColi_full_protein_vs"
            "_mTuberculosis_full_protein_parsed.csv"
        )
        dot, links = compare_ica(S1, S2, ortho_file=ortho_dir)
        print(links)


if __name__ == "__main__":
    unittest.main()
