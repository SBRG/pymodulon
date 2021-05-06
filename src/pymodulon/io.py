"""
Functions for reading and writing data into files.
"""

import gzip
import json

import pandas as pd

from pymodulon.core import IcaData


def save_to_json(data, filename, compress=False):
    """
    Save :class:`~pymodulon.core.IcaData` object to a json file

    Parameters
    ----------
    data: ~pymodulon.core.IcaData
       ICA dataset to be saved to json file
    filename: str
       Path to json file where the data will be saved
    compress: bool
        Indicates if the JSON file should be compressed into a gzip archive

    Returns
    -------
    None: None
    """

    # only keeps params that are used to initialize the data
    load_params = IcaData.__init__.__code__.co_varnames
    param_dict = {
        key: getattr(data, key) for key in vars(IcaData) if key in load_params
    }

    # serialize pandas DataFrames and change sets to lists
    for key, val in param_dict.items():
        if isinstance(val, pd.DataFrame):
            # Replace string NaN with empty string
            str_cols = val.select_dtypes(object).columns
            val[str_cols] = val[str_cols].fillna("")
            param_dict[key] = val.astype(str).to_json()
        elif isinstance(val, pd.Series):
            if val.dtype == object:
                newval = val.fillna("")
            param_dict[key] = newval.astype(str).to_json()
        elif isinstance(val, set):
            param_dict[key] = list(val)

        # Encode MotifInfo objects as dictionaries
        elif key == "motif_info":
            new_val = {}
            for k1, v1 in data.motif_info.items():
                new_v1 = {}
                for k2, v2 in v1.__dict__.items():
                    try:
                        new_v1[k2[1:]] = v2.to_json(orient="table")
                    except AttributeError:
                        new_v1[k2[1:]] = v2
                new_val[k1] = new_v1

            param_dict[key] = new_val

    # Add _cutoff_optimized
    param_dict["_cutoff_optimized"] = data.cutoff_optimized
    param_dict["_dagostino_cutoff"] = data.dagostino_cutoff

    if filename.endswith(".gz") or compress:
        if not filename.endswith(".json.gz"):
            filename += ".json.gz"
        with gzip.open(filename, "wt", encoding="ascii") as zipfile:
            json.dump(param_dict, zipfile)
    else:
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename, "w") as fp:
            json.dump(param_dict, fp)


def load_json_model(filename):
    """
    Load :class:`~pymodulon.core.IcaData` object from a file in JSON format.

    Parameters
    ----------
    filename : str or ~io._io.StringIO
        File path or descriptor that contains the JSON document describing the
        ICA dataset.

    Returns
    -------
    IcaData : ~pymodulon.core.IcaData
        The :class:`~pymodulon.core.IcaData` object as represented in the JSON
        document.
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

    # Add cutoff information
    try:
        cutoff_optimized = serial_data.pop("_cutoff_optimized")
    except KeyError:
        cutoff_optimized = False

    try:
        dagostino_cutoff = serial_data.pop("_dagostino_cutoff")
    except KeyError:
        dagostino_cutoff = None

    # Remove deprecated arguments
    deprecated_args = ["cog_colors"]
    for arg in deprecated_args:
        if arg in serial_data.keys():
            serial_data.pop(arg)

    data = IcaData(**serial_data)
    data._cutoff_optimized = cutoff_optimized
    data._dagostino_cutoff = dagostino_cutoff
    return data
