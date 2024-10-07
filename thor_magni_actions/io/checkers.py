import os
from .create import create_dir


def check_dir(dir_path: str, create: bool = False):
    """check if a certain path is a directory"""
    if not os.path.isdir(dir_path):
        if not create:
            raise ValueError(f"{dir_path} is not a directory")
        else:
            create_dir(dir_path)


def check_path(file_path: str):
    """check if a certain file path is a path"""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(file_path)
