import os
import pickle
import json


def dump_pickle_file(data_to_save, save_path: str):
    """save pickle file"""
    file_o = open(save_path, "wb")
    pickle.dump(data_to_save, file_o)


def dump_json_file(data_to_save: dict, save_path: str):
    """save json file"""
    file_o = open(save_path, "w")
    json.dump(data_to_save, file_o)


def create_dir(path: str):
    """create directory"""
    if not os.path.exists(path):
        os.makedirs(path)
