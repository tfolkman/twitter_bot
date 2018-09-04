"""
test_util.py
====================================
Tests the util functions
"""

from bot.common import util


def test_read_lines():
    path = "../resources/twitter_en_head.txt"
    lines = util.read_lines(path)
    assert lines == ["yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable",
                    "yeah dude i would definitely consider a daniel defence super reliable and they are just bad ass",
                     "i'm about to meet my mans ex friend with benefit, tune in next week to see if i have to put hands on"]


def test_filter_line():
    filtered = util.filter_line("^&twitter is %awesome!!")
    assert filtered == "twitter is awesome"


def test_filter_data():
    rows = ["yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable",
                     "yeah dude i would definitely consider a daniel defence super reliable and they are just bad ass",
                     "this is a good questions",
                     "short answer",
                     "l l l l l l l l l l l l l l l l l l l l l",
                     "s s s s"]
    filtered_queries, filtered_answers = util.filter_data(rows)
    assert filtered_queries == ["yeah i'm preparing myself to drop a lot on this man, but definitely need something reliable"]
    assert filtered_answers == ["yeah dude i would definitely consider a daniel defence super reliable and they are just bad ass"]
