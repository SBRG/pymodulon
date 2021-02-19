from os.path import abspath, dirname, isfile, join

import numpy as np

from pymodulon.io import load_json_model, save_to_json


PYMOD_DIR = abspath(join(dirname(abspath(__file__)), ".."))
"""str: The directory location of where :mod:`pymodulon` is installed."""

DATA_DIR = join(PYMOD_DIR, "tests", "data", "models")
"""str: The directory location of the test data."""


def test_save_to_json(mini_obj):
    fname = join(DATA_DIR, "mini_model.json")
    save_to_json(mini_obj, fname)
    assert isfile(fname)


def test_load_json_model(mini_obj):
    fname = join(DATA_DIR, "mini_model.json")
    icd_from_json = load_json_model(fname)
    assert np.allclose(icd_from_json.M, mini_obj.M)
    # TODO: Add remaining attributes
