import os
from os.path import abspath, dirname, isfile, join

import numpy as np

from pymodulon.io import load_json_model, save_to_json


PYMOD_DIR = abspath(join(dirname(abspath(__file__)), ".."))
"""str: The directory location of where :mod:`pymodulon` is installed."""

TEST_DIR = join(PYMOD_DIR, "tests")
"""str: The directory location of the test directory."""


def test_save_to_json(mini_obj):
    fname = join(TEST_DIR, "mini_model.json")
    save_to_json(mini_obj, fname)
    assert isfile(fname)

    icd_from_json = load_json_model(fname)
    assert np.allclose(icd_from_json.M, mini_obj.M)
    # TODO: Test remaining attributes
    os.remove(fname)
