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