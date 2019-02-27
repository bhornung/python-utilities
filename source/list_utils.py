"""
Collection of simple list transformation utilities.
"""

def chunker(X, size):
    """
    Splits a list to consequtive chunks.
    Parameters:
        X ([]) : any list-like that supports indexing.
        size (int) : size of a chunk
    Returns:
        chunked ([[]]) : a 2D nested list of the original list.
    """
    chunked = [X[idx : idx + size] for idx in range(0, len(X), size)]
    return chunked


def pad_and_crop_lists(lists_, length, pad_value = ""):
    """
    Sets lists to equal size.
    Parameters:
        lists_ ({[[]], generator of lists}) : sequence of lists
        length (int) : desired length of the lists
        pad_value (object) : if a list shorter than 'length' it will be right padded with this value. Default "".
    Returns:
        cropped_padded ([[]]) : list of cropped and padded lists.
    """
    
    cropped_padded = [x.__add__([pad_value] * (length - len(x)))[:length] for x in lists_]
    
    return cropped_padded


def list_multiplier(list_):
    """
    Creates generators iterating through the same list.
    Parameters:
        list_ ([]) a list or other iterable. 
    Returns:
        new_generator (generator) : generator of the list elements.
    Notes:
        Do not pass generators to this function, for they are exhausted in the first iteration.
    Usage
    > list_resource = list_multiplier(list_)
    > input = next(list_resource)
    """
    
    while True:
        new_generator = (x for x in list_)
        yield new_generator


def list_apply_rule_multiplier(list_, rule):
    """
    Creates generators iterating through the same list whilst applying a rule to each element of the list.
    Parameters:
        list_ ([]) a list or other iterable. 
        rule (callable) : a function operating on the elements of the list.
    Returns:
        new_generator (generator) : generator of the list elements.
    Notes:
        Do not pass generators to this function, for they are exhausted in the first iteration.
    Usage
    > list_resource = list_multiplier(list_)
    > input = next(list_resource)
    """
    while True:
        new_generator = (rule(x) for x in list_)
        yield new_generator