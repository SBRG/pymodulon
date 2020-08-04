import json
import pandas as pd
from typing import Union
from pymodulon.core import IcaData
"""
Functions for reading and writing model into files.
"""

#TODO: should only save subset of the model. i.e. things like colors can be discarded to reduce file size.
#TODO: Change M to S when core is changed


def model_to_dict(model) -> dict:
    """

    Convert model to dictionary with serialized objects.
    Parameters
    ----------
    model : ICA model
        ICA model to be converted to dict. Must contain M and A matrix

    Returns
    -------
    model_dict: dict
        dictionary version of ICA model object.

    """

    if model.A == None or model.M == None:
        raise ValueError('The model must include the M and the A matrix.')

    #serialize pandas DataFrames
    for key, val in model.__dict__.items():
        if isinstance(val, pd.core.frame.DataFrame):
            model.__dict__.update({key: val.to_json()})
        elif isinstance(val, set):
            model.__dict__.update({key: list(val)})
    return model.__dict__


def save_to_json(model: IcaData, fname: str):
    """

    Save model to the json file
    Parameters
    ----------
    model: ICA model
       ICA model to be saved to json file
    fname: string
       path to json file where the model will be saved

    """
    if not fname.endswith('.json'):
        fname = fname += '.json'

    with open(fname, 'w') as fp:
        json.dump(model_to_dict(model), fp)

def load_json_model(filename: Union[str, file]) -> IcaData:
    """
    Load a ICA model from a file in JSON format.
    Parameters
    ----------
    filename : str or file-like
        File path or descriptor that contains the JSON document describing the
        ICA model.
    Returns
    -------
    ica_model
        The ICA model as represented in the JSON document.

    """
    if isinstance(filename, string_types):
        with open(filename, "r") as file_handle:
            serial_data =  json.load(file_handle)
    else:
        serial_data = json.load(filename)

    #deserialize the dataframes
    DF_list = ['M', 'A', 'X', 'gene_table', 'sample_table', 'imodulon_table',
               'trn']
    for dtype in DF_list:
        df = pd.DataFrame(serial_data[dtype])
        serial_data.update({dtype: df})

    return model_from_dict(serial_data)

def model_from_dict(object: dict) -> IcaData:
    """

    Parameters
    ----------
    object: dict
        dictionary containing the ICA model, must contain M and A matrices

    Returns
    -------
    ica_model:
        ICA model as represented in the dictionary

    """
    if 'A' not in  object or 'M' not in object:
        raise ValueError('The model must include the M and the A matrix.')

    model = IcaData(**object)