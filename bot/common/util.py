"""
util.py
====================================
Contains common utils used across the code
"""


def read_lines(filename: str) -> object:
    """

    Parameters
    ----------
    filename

    Returns
    -------
    returns a list of sentences from the file

    """
    return open(filename).read().split('\n')[:-1]