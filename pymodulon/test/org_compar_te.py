import unittest

import sys

import pandas as pd

sys.path.append("/home/sbrg/Desktop/pymodulon/")
from pymodulon.util import *

DATA_DIR = "../data/"


class MyTestCase(unittest.TestCase):
    def test_load(self):
        ortho_dir = "../org_comp_data"
        ortho_DF = load_ortho_matrix(ortho_dir)
        self.assertIsInstance(ortho_DF, pd.DataFrame)
        self.assertEqual(ortho_DF.shape, (56672, 56672))

    def test_extraction(self):
        S1 = pd.read_csv("data/mtb_S.csv", index_col=0)
        S2 = pd.read_csv("data/ecoli_S.csv", index_col=0)
        ortho_dir = "../org_comp_data"
        ortho_DF = load_ortho_matrix(ortho_dir)
        gene_list_1 = list(S1.index)
        gene_list_2 = list(S2.index)
        reduced_DF = extract_genes(gene_list_1, gene_list_2, ortho_DF)
        self.assertGreater(ortho_DF.shape, reduced_DF.shape)

    def test_translate(self):
        ortho_dir = "../org_comp_data"
        ortho_DF = load_ortho_matrix(ortho_dir)
        gene_list_1 = ["Rv0001", "Rv0002", "Rv0003", "STM474_RS01630", "STM_RS01635"]
        gene_list_2 = ["b0002", "b0004", "b0008", "b0239"]
        reduced_DF = extract_genes(gene_list_1, gene_list_2, ortho_DF)
        translated_genes = translate_genes(["b0002", "b0004", "b0008", "b0239"], reduced_DF)
        self.assertEqual(translated_genes, ["b0002", "b0004", "b0008", "STM474_RS01630"])

    def test_compare_ica(self):
        S1 = pd.read_csv("data/mtb_S.csv", index_col=0)
        S2 = pd.read_csv("data/ecoli_S.csv", index_col=0)
        ortho_dir = "../org_comp_data"
        dot, links = compare_ica(S1, S2, ortho_dir=ortho_dir)
        print(dot)

if __name__ == '__main__':
    unittest.main()
