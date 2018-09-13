"""
test_process.py
====================================
Tests the util functions
"""

from bot.data import process


def test_read_lines():
    path = "./resources/twitter_en_head.txt"
    lines = process.read_lines(path)
    assert lines == ["yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable",
                    "yeah dude i would definitely consider a daniel defence super reliable and they are just bad ass",
                     "i'm about to meet my mans ex friend with benefit, tune in next week to see if i have to put hands on"]


def test_filter_characters():
    filtered = process.filter_characters("^&twitter is %awesome!!")
    assert filtered == "twitter is awesome"


def test_filter_size():
    rows = ["yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable",
                     "yeah dude i would definitely consider a daniel defence super reliable and they are just bad ass",
                     "this is a good questions",
                     "short answer",
                     "l l l l l l l l l l l l l l l l l l l l l",
                     "s s s s"]
    filtered_queries, filtered_answers = process.filter_size(rows)
    assert filtered_queries == ["yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable"]
    assert filtered_answers == ["yeah dude i would definitely consider a daniel defence super reliable and they are just bad ass"]


def test_create_vocab():
    tokenized_sentences = [["the", "fox", "ran"], ["the", "fox", "jumped"], ["the", "fox", "swam"]]
    index2word, word2index = process.create_vocab(tokenized_sentences, 2)
    assert index2word == {0: "_", 1: process.UNK, 2: process.PAD, 3: 'the', 4: 'fox'}
    assert word2index == {"_": 0, process.UNK: 1, process.PAD: 2, 'the': 3, 'fox': 4}


def test_convert_sequence_to_padded_indexes():
    sequence = ["the", "fox", "ran", "insanely"]
    word2index = {"the": 0, "fox": 1, "ran": 2, process.UNK: 3, process.PAD: 4}
    converted_sequence = process.convert_sequence_to_padded_indexes(sequence, word2index, 10)
    assert converted_sequence == [0, 1, 2, 3, 4, 4, 4, 4, 4, 4]
