# from os.path import abspath, dirname, join
#
# from pymodulon.compare import convert_gene_index
# from pymodulon.io import load_json_model


# def test_convert_gene_index():
#
#     ica_data1 = load_json_model(join(DATA_DIR, "model.json"))
#     ica_data2 = load_json_model(join(DATA_DIR, "10genes.json"))
#     ica_data_org = load_json_model(join(DATA_DIR, "saci.json"))
#
#     # Test conforming data from same organism
#     M1, M2 = convert_gene_index(ica_data1.M, ica_data2.M)
#     assert (M1.index == M2.index).all()
#     assert len(M1.index) == 10
#
#     # Test conforming data from different organisms
#     orgM1, orgM2 = convert_gene_index(
#         ica_data1.M, ica_data_org.M, ortho_file=join(DATA_DIR, "example_bbh.csv")
#     )
#     assert (orgM1.index == orgM2.index).all()
#     assert len(orgM1) > 10
#
#     # Test conforming gene info data
#     org_table1, org_table2 = convert_gene_index(
#         ica_data1.gene_table,
#         ica_data_org.gene_table,
#         ortho_file=join(DATA_DIR, "example_bbh.csv"),
#     )
#     assert (org_table1.index == org_table2.index).all()
#     assert (org_table1.index == orgM1.index).all()


# import unittest
#
# import pandas as pd
#
# from pymodulon.compare import compare_ica
#
#
# def test_compare_ica():
#     S2 = pd.read_csv("data/mtb_S.csv", index_col=0)
#     S1 = pd.read_csv("data/ecoli_S.csv", index_col=0)
#     ortho_dir = (
#         "/home/sbrg/Desktop/modulome/data/organism_compare/"
#         "bbh_csv/eColi_full_protein_vs"
#         "_mTuberculosis_full_protein_parsed.csv"
#     )
#     dot, links = compare_ica(S1, S2, ortho_file=ortho_dir)
#
#
# class MyTestCase(unittest.TestCase):
#     pass
#
#
# if __name__ == "__main__":
#     unittest.main()
