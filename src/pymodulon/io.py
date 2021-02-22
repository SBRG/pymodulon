"""
Functions for reading and writing model into files.
"""

import gzip
import json
from typing import TextIO, Union

import pandas as pd

from pymodulon.core import IcaData


def save_to_json(model, fname, compress=False):
    """
    Save model to the json file

    Parameters
    ----------
    model: IcaData
       ICA model to be saved to json file
    fname: str
       Path to json file where the model will be saved
    compress: bool
        Indicates if the JSON file should be compressed into a gzip archive
    """

    if model.A is None or model.M is None:
        raise ValueError("The model must include the M and the A matrix.")

    # only keeps params that are used to initialize the model
    load_params = IcaData.__init__.__code__.co_varnames
    param_dict = {
        key: getattr(model, key) for key in vars(IcaData) if key in load_params
    }

    # serialize pandas DataFrames and change sets to lists
    for key, val in param_dict.items():
        if isinstance(val, pd.DataFrame) or isinstance(val, pd.Series):
            param_dict[key] = val.to_json()
        elif isinstance(val, set):
            param_dict[key] = list(val)

        # Encode MotifInfo objects as dictionaries
        elif key == "motif_info":
            new_val = {}
            for k1, v1 in model.motif_info.items():
                new_v1 = {}
                for k2, v2 in v1.__dict__.items():
                    try:
                        new_v1[k2[1:]] = v2.to_json(orient="table")
                    except AttributeError:
                        new_v1[k2[1:]] = v2
                new_val[k1] = new_v1

            param_dict[key] = new_val

    if fname.endswith(".gz") or compress:
        if not fname.endswith(".json.gz"):
            fname += ".json.gz"
        with gzip.open(fname, "wt", encoding="ascii") as zipfile:
            json.dump(param_dict, zipfile)
    else:
        if not fname.endswith(".json"):
            fname += ".json"
        with open(fname, "w") as fp:
            json.dump(param_dict, fp)


def load_json_model(filename):
    """
    Load a ICA model from a file in JSON format.

    Parameters
    ----------
    filename : str or TextIO
        File path or descriptor that contains the JSON document describing the
        ICA model.

    Returns
    -------
    IcaData
        The ICA model as represented in the JSON document.
    """
    if isinstance(filename, str):
        if filename.endswith(".gz"):
            with gzip.GzipFile(filename, "r") as zipfile:
                serial_data = json.loads(zipfile.read().decode("utf-8"))
        else:
            with open(filename, "r") as file_handle:
                serial_data = json.load(file_handle)
    else:
        serial_data = json.load(filename)

    # Remove deprecated arguments
    deprecated_args = ["cog_colors", "_dagostino_cutoff"]
    for arg in deprecated_args:
        if arg in serial_data.keys():
            serial_data.pop(arg)

    return IcaData(**serial_data)
