from os.path import isfile

import numpy as np

from pymodulon.io import load_json_model, save_to_json


def test_save_to_json(tmp_path, ecoli_obj):
    fname = str(tmp_path / "ecoli_obj.json")
    save_to_json(ecoli_obj, fname)
    assert isfile(fname)

    new_icd = load_json_model(fname)
    # Data
    assert np.allclose(new_icd.M, ecoli_obj.M)
    assert np.allclose(new_icd.A, ecoli_obj.A)
    assert np.allclose(new_icd.X, ecoli_obj.X)
    # Names
    assert new_icd.imodulon_names == ecoli_obj.imodulon_names
    assert new_icd.sample_names == ecoli_obj.sample_names
    assert new_icd.gene_names == ecoli_obj.gene_names
    # Tables
    assert new_icd.gene_table.equals(ecoli_obj.gene_table.replace("", np.nan))
    assert new_icd.sample_table.equals(ecoli_obj.sample_table.replace("", np.nan))
    assert new_icd.imodulon_table.equals(ecoli_obj.imodulon_table.replace("", np.nan))
    # Thresholds
    assert np.allclose(
        [new_icd.thresholds[x] for x in new_icd.imodulon_names],
        [ecoli_obj.thresholds[x] for x in ecoli_obj.imodulon_names],
    )
    assert new_icd.cutoff_optimized == ecoli_obj.cutoff_optimized
    assert new_icd.dagostino_cutoff == ecoli_obj.dagostino_cutoff
