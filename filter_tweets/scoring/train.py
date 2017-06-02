from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
import json
import select
import numpy as np
import os
import time
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
from ml_utils import load_data
from ml_utils import train


def fix_data(text, labels):
    """
    This gets rid of all tweets labeled with "ambiguous" as positive,
    as well as cleans the tweet text to get rid of the "CODE THIS TWEET: ", etc.
    which is an artifact of how it was stored/displayed in NEXT
    """

    text_out = []
    labels_out = []

    for t, l in zip(text, labels):
        if l[5] != 1:
            text_out.append(t)
            labels_out.append(l)

    def fix_text(t):
        return t.split('RETWEET,')[0].strip().split('TWEET:')[1].strip()

    text_out = [fix_text(t) for t in text_out]

    return text_out, labels_out


def main(data, train_data, test_data):

    CLASSES_TO_TRAIN = [0, 1, 2, 3, 4]

    # load the data
    train_text, train_labels, test_text, test_labels = load_data(data, train_data, test_data, id_key='__index__', text_key='text')
    # fix the data
    train_text, train_labels = fix_data(train_text, train_labels)
    test_text, test_labels = fix_data(test_text, test_labels)
    # train on the data
    train(train_text, train_labels, test_text, test_labels, CLASSES_TO_TRAIN)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("\nUsage: \n\tpython train.py DATA TRAIN TEST\n\n\t"
              "DATA: file that contains all the data.  See ml_utils.load_data for more info"
              "\n\tTRAIN: .json file with train labels"
              "\n\tTEST: .json file with test labels"
              "\n\nExample:"
              "\n\tpython train.py ../test_in.csv examples/train_labels.json examples/test_labels.json\n")
    else:
        main(*sys.argv[1:])
