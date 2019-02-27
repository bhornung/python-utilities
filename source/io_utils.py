def load_from_json(path_to_db):
    """
    Loads the contents of a json file.
    
    Parameters:
        path_to_db (str) : full path to a json file.
        
    Returns:
        data_ (object) : the loaded object.
    """
    with open(path_to_db, 'r') as fproc:
        data_ = json.load(fproc)
            
    return data_


def load_dict_from_json(path_to_db, convert_keys_to_int = False):
    """
    Loads a dictionary from a json file. It optionally tries to convert the keys to integers.
    (Integers cannot be keys in standard json.)
    
    Parameters:
        path_to_db (str) : full path to a json file.
        convert_keys_to_int (bool) : whether to coerce keys to ints. Default: False
        
    Returns:
        dict_ ({:}) : the loaded dictionary
    """
    with open(path_to_db, 'r') as fproc:
        dict_ = json.load(fproc)

    if not isinstance(dict_, dict):
        raise TypeError("Loaded object is not a dictionary.")
            
    if convert_keys_to_int:
        try:
            dict_ = {int(k) : v for k, v in dict_.items()}
        except:
            raise
            
    return dict_


def save_to_json(path_to_db, data):
    """
    Saves an obect to a file.
    
    Parameters:
        path_to_db (str) : full path to file
        data (object) : object to save
    """
    with open(path_to_db, 'w') as fproc:
        json.dump(data, fproc, indent = 4)