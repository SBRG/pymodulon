import os
from os.path import isfile

import numpy as np

from pymodulon.io import load_json_model, save_to_json


def test_save_to_json(tmp_path, ecoli_obj):
    fname = str(tmp_path / "ecoli_obj.json")
    save_to_json(ecoli_obj, fname)
    assert isfile(fname)

    icd_from_json = load_json_model(fname)
    assert np.allclose(icd_from_json.M, ecoli_obj.M)
    # TODO: Test remaining attributes
