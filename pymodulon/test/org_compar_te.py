import unittest

import sys

import pandas as pd

sys.path.append("/home/sbrg/Desktop/pymodulon/")
from pymodulon.util import *


class MyTestCase(unittest.TestCase):

    def test_compare_ica(self):
        S1 = pd.read_csv("data/mtb_S.csv", index_col=0)
        S2 = pd.read_csv("data/ecoli_S.csv", index_col=0)
        ortho_dir = "/home/sbrg/Desktop/modulome/data/organism_compare/bbh_csv"
        dot, links = compare_ica(S1, S2, ortho_dir=ortho_dir)
        print(links)

    def test_compare_ica_flip(self):
        S2 = pd.read_csv("data/mtb_S.csv", index_col=0)
        S1 = pd.read_csv("data/ecoli_S.csv", index_col=0)
        ortho_dir = "/home/sbrg/Desktop/modulome/data/organism_compare/bbh_csv"
        dot, links = compare_ica(S1, S2, ortho_dir=ortho_dir)
        print(links)

    def test_compare_ica_manual(self):
        S2 = pd.read_csv("data/mtb_S.csv", index_col=0)
        S1 = pd.read_csv("data/ecoli_S.csv", index_col=0)
        ortho_dir = "/home/sbrg/Desktop/modulome/data/organism_compare/bbh_csv"
        dot, links = compare_ica(S1, S2, ortho_dir=ortho_dir, auto_find=False, org_1_name="eColi",
                                 org_2_name="mTuberculosis")

        print(links)


if __name__ == '__main__':
    unittest.main()
