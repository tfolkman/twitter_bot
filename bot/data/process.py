"""
process.py
====================================
Contains data utils used across the code
"""
from typing import List, Tuple
from collections import Counter
from re import sub
from itertools import chain
import pickle
import os.path
import sys
import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

UNK = 'unk_token'
PAD = 'pad_token'
VOCAB_SIZE = 8000
MAX_LENGTH = 20
MIN_LENGTH = 3


def read_lines(filename: str) -> List[str]:
    """
    Read in file and returns each line as an entry in a list
    Parameters
    ----------
    filename : str
        Path to the data to be read

    Returns
    -------
    list
        a list of sentences from the file

    """
    lines = []
    with open(filename, "r") as f:
        for l in f:
            lines.append(l.strip())
    return lines


def filter_characters(line: str) -> str:
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
    return sub(r'[^a-zA-Z0-9_\s]+', "", line)


def filter_size(sequences: List[str], min_length: int, max_length: int) -> Tuple[List[str], List[str]]:
    """
    Filters data based on the size of the sentence
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


def create_vocab(tokenized_sentences, vocab_size):
    """
    Takes in a list of tokenized sentences and returns the index2word and word2index dictionaries
    Parameters
    ----------
    tokenized_sentences : List[List[str]]
        List of lists of words
    vocab_size : int
        Size of vocabulary you want to keep

    Returns
    -------
    dict
        Index2Word for vocab_size most data words
    dict
        Word2Index for vocab_size most data words

    """
    counter = Counter(chain.from_iterable(tokenized_sentences))
    vocab = [v[0] for v in counter.most_common(vocab_size)]
    vocab = ['_'] + [UNK] + [PAD] + vocab
    index2word = {i: w for i, w in enumerate(vocab)}
    word2index = {w: i for i, w in index2word.items()}
    return index2word, word2index


def convert_sequence_to_padded_indexes(sequence, word2index, max_length):
    """
    This function takes in a sequence of words and converts to a sequence of indexes.
    Pads and inserts unknown tokens where appropriate
    Parameters
    ----------
    sequence : List[str]
        List of words
    word2index : dict
        Word -> Index. Including the UNK and PAD words
    max_length : int
        The maximum length of a sequence across all data

    Returns
    -------
    List[int]
        A list of indexes for the words padded to be max length.

    """
    indices = []
    for word in sequence:
        if word in word2index:
            indices.append(word2index[word])
        else:
            indices.append(word2index[UNK])
    return indices + [word2index[PAD]]*(max_length - len(sequence))


def convert_all_sequences(queries, answers, word2index, max_length):
    assert len(queries) == len(answers)
    n_rows = len(queries)

    index_q = np.zeros([n_rows, max_length], dtype=np.int32)
    index_a = np.zeros([n_rows, max_length], dtype=np.int32)

    for i in range(n_rows):
        index_q[i] = convert_sequence_to_padded_indexes(queries[i], word2index, max_length)
        index_a[i] = convert_sequence_to_padded_indexes(answers[i], word2index, max_length)

    return index_q, index_a


def process_file(file_path, export_directory):
    """
    Process the twitter data into indexes and lines
    Parameters
    ----------
    file_path : str
        The filepath to the twitter data
    export_directory : str
        The directory to export index2word, word2index, query lines, and answer lines
    """
    logger.info("Reading in: {}".format(file_path))
    lines = read_lines(file_path)
    logging.info("Number of lines: {}".format(len(lines)))
    lines = [line.lower() for line in lines]
    logger.info("Filtering lines...")
    lines = [filter_characters(line) for line in lines]
    query_lines, answer_lines = filter_size(lines, MIN_LENGTH, MAX_LENGTH)
    logger.info("Splitting data...")
    query_tokens = [sentence.split() for sentence in query_lines]
    answer_tokens = [sentence.split() for sentence in answer_lines]
    logger.info("Converting to indexes...")
    index2word, word2index = create_vocab(query_tokens + answer_tokens, VOCAB_SIZE)
    index_q, index_a = convert_all_sequences(query_tokens, answer_tokens, word2index, MAX_LENGTH)
    logger.info("Writing out data...")
    pickle.dump(index2word, open(os.path.join(export_directory, "index2word.pickle"), "wb"))
    pickle.dump(word2index, open(os.path.join(export_directory, "word2index.pickle"), "wb"))
    pickle.dump(index_q, open(os.path.join(export_directory, "querylines.pickle"), "wb"))
    pickle.dump(index_a, open(os.path.join(export_directory, "answerlines.pickle"), "wb"))


if __name__ == '__main__':
    """
    Takes in the file path to twitter data and directory to export pickled files to
    """
    process_file(sys.argv[1], sys.argv[2])
