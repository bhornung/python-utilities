

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