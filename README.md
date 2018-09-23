### Goal of Project

I have been wanting to write a twitter chat bot as well as try and write 
some code that is well structured and tested. If you see problems with either, please let me know!
Inspired from this post and some code borrowed as well:  
http://suriyadeepan.github.io/2016-12-31-practical-seq2seq/

Also, used some Seq2Seq code from Pytorch tutorials:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
### Getting Started

This is a [Pipenv project](https://pipenv.readthedocs.io/en/latest/)

1. Install Pipenv: pip install --user pipenv
2. Install packages: pipenv install (assumes cuda 9.2)
3. Activate pipenv shell: pipenv shell
4. Generate necessary data: python bot/common/util.py [path to twitter data] [directory to save extracted data to]
    1. Twitter data can be found [here](https://github.com/Marsan-Ma/chat_corpus/blob/master/twitter_en.txt.gz)
    


### How to run tests

1. Activate pipenv shell: pipenv shell
2. Run tests:

    cd bot/tests
    
    pytest