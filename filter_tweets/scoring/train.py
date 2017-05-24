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


def main(data, train_data, test_data):

    CLASSES_TO_TRAIN = [0, 1, 2, 3]

    train_text, train_labels, test_text, test_labels = load_data(data, train_data, test_data)
    train(train_text, train_labels, test_text, test_labels, CLASSES_TO_TRAIN)

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("\nUsage: \n\tpython train.py DATA TRAIN TEST\n\n\t"
              "DATA: source chr(1)-separated .csv file that contains all metadata"
              "\n\tTRAIN: .json file with train labels"
              "\n\tTEST: .json file with test labels"
              "\n\nExample:"
              "\n\tpython train.py ../test_in.csv examples/train_labels.json examples/test_labels.json\n")
    else:
        main(*sys.argv[1:])
