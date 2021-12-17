import pandas as pd
import json
import os


def load_json(*paths):
    """Loads file and returns a dictionary.

    Args:
        paths (strs): Relative paths in proper order.

    Returns:
        dict: the file
    """
    fullpath = os.path.join(*paths)
    with open(fullpath, "r") as file:
        return json.load(file)


def save_json(data, *paths):
    """Saves dictionary to json file.

    Args:
        data (dict): Dictionary to be saved as yaml.
        paths (strs): Relative paths in proper order.
    """
    fullpath = os.path.join(*paths)
    with open(fullpath, "w") as file:
        json.dump(data, file, indent=4, default=str)


def convert_tuple_to_str(tuple):
    """Convert tuple to string

    Args:
        tuple (tuple): tuple variable
    """
    string = ''
    for tup in tuple:
        string += f",{str(tup)}"

    return string[1:]


def convert_str_to_tuple(string):
    """Convert tuple to string

    Args:
        str (str): str varible
    """
    ls_str = string.split(',')
    ls_int = [int(str) for str in ls_str]
    return tuple(ls_int)