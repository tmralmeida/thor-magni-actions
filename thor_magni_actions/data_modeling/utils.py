from thor_magni_actions.io import load_yaml_file


def load_config(path, default_path=None):
    """Loads config file.
    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    """
    cfg_spec = load_yaml_file(path)
    if default_path:
        cfg = load_yaml_file(default_path)
    else:
        inherit_from = cfg_spec.get("inherit_from")
        cfg = load_yaml_file(inherit_from)
    update_recursive(cfg, cfg_spec)
    return cfg


def update_recursive(dict1, dict2):
    """Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    """
    for k, v in dict2.items():
        # Add item if not yet in dict1
        if k not in dict1:
            dict1[k] = None
        # Update
        if isinstance(dict1[k], dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def merge_dicts(dict_1, dict_2):
    """merge dictionaries"""
    dict_3 = {**dict_1, **dict_2}
    for key, value in dict_3.items():
        if key in dict_1 and key in dict_2:
            kd = dict_1[key]
            if not isinstance(value, list):
                new_val = [value]
            if not isinstance(dict_1[key], list):
                kd = [dict_1[key]]
            new_val.extend(kd)
            dict_3[key] = new_val
    return dict_3