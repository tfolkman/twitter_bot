"""
util.py
====================================
Contains common utils used across the code
"""
from typing import List, Tuple
import re

def read_lines(filename: str) -> List[str]:
    """

    Parameters
    ----------
    filename

    Returns
    -------
    returns a list of sentences from the file

    """
    return open(filename).read().split('\n')[:-1]


def filter_line(line: str) -> str:
    """
    Removes any non-alphanumeric characters from a string
    Parameters
    ----------
    line : str
        The string you want filtered

    Returns
    -------
    str
        The line with non-alphanumerics filtered out

    """
    return re.sub(r'[^a-zA-Z0-9_\s]+', "", line)


def filter_data(sequences: List[str], min_length: int = 3, max_length: int = 20) -> Tuple[List[str], List[str]]:
    """

    Parameters
    ----------
    sequences : List[str]
        List of sting sequences where evens are queries and odds answers
    min_length: int
        Min length of a query or answer
    max_length: int
        Max length of a query or answer

    Returns
    -------
    Tuple[List[str], List[str]]
        The filtered queries, the filtered answers

    """
    filtered_query, filtered_answer = [], []
    # even rows are queries and odd are answers
    for i in range(0, len(sequences), 2):
        query = sequences[i]
        answer = sequences[i+1]
        query_length = len(query.split())
        answer_length = len(answer.split())
        if min_length <= query_length <= max_length and min_length <= answer_length <= max_length:
            filtered_query.append(query)
            filtered_answer.append(answer)
    return filtered_query, filtered_answer
