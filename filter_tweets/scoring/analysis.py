import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from scipy.sparse import issparse
from utils import Text2Vec
from utils import tokenize
import matplotlib.pyplot as plt
import json
import random
import sys
import select
import numpy as np
import os
import time


CLASSES_TO_ANALYZE = [0, 1, 2, 3]

# if True, if a tweet has all zeros as its label it will be disregarded
IGNORE_ALL_NEGATIVE = True


def main(data, train, test):

    train_text, train_labels, test_text, test_labels = load_data(data, train, test)

    i = 0

    if not os.path.exists('results'):
        os.mkdir('results')

    for cls in CLASSES_TO_ANALYZE:

        params = get_model(cls)
        these_train_labels = [l[cls] for l in train_labels]
        these_test_labels = [l[cls] for l in test_labels]

        t0 = time.time()

        n_train = len(train_text)
        n_test = len(test_text)

        if params['features'] == 'text2vec':
            features = Text2Vec
            params['feature_params']['word2vec'] = 'twitter_guns_w2v.pickle'
        elif params['features'] == 'tfidf':
            features = TfidfVectorizer

        # get feature matrix
        X = get_features(train_text + test_text, features, tokenizer=tokenize, **params['feature_params'])
        X_train = X[:n_train]
        X_test = X[n_train:]
        del X

        svm = LinearSVC()
        svm.set_params(**params['clf_params'])
        svm.fit(X_train, these_train_labels)

        results = dict()

        results['recall'] = recall_score(these_test_labels, svm.predict(X_test))
        results['precision'] = precision_score(these_test_labels, svm.predict(X_test))
        results['f1'] = f1_score(these_test_labels, svm.predict(X_test))
        results['auc'] = roc_auc_score(these_test_labels, svm.decision_function(X_test))

        fpr, tpr, thresholds = roc_curve(these_test_labels, svm.decision_function(X_test))

        plt.figure()
        plt.plot(fpr, tpr)
        plt.savefig('results/results_roc_class{}.png'.format(cls))

        print("Trained class {}.  Iteration took {}s".format(cls, time.time() - t0))

        with open('results/results_class{}.json'.format(cls), 'w') as f:
            json.dump(results, f)


def get_features(raw_text, features, pca_d, **kwargs):
    """
    Turns raw_text into a feature matrix.
    
    if pca_d is specified (and is larger than # of examples and # of words, PCA is performed)
    
    Note: Tfidf outputs a sparse matrix, but PCA only accepts dense matrices.
    So if PCA is not used, this will return a sparse matrix, but otherwise
    it will be dense.
    """

    X = features(**kwargs).fit_transform(raw_text)
    if pca_d is not None and pca_d < min(*X.shape):
        try:
            if issparse(X):
                X = X.toarray()
            X = PCA(pca_d).fit_transform(X)
        except MemoryError:
            # If matrix is too big this will probably cause MemoryError due to being
            # converted to dense.  TruncatedSVD keeps it sparse.
            X = TruncatedSVD(pca_d).fit_transform(X)

    return X


def get_model(cls):
    """
    Loads the parameters that we got from train.py
    """
    if not os.path.exists('models/params_class{}.json'.format(cls)):
        print("models/params_class{}.json not found.  Run train.py first.".format(cls))
        sys.exit()
    with open('models/params_class{}.json'.format(cls)) as f:
        params = json.load(f)

    return params


def load_data(data, train, test):
    """
    Loads data.
    data should be the .json file with all the metadata.
    train and test should be .json files that are dictionaries with keys like:
    <article_source>_<article_index>
    that point to 0 or 1, according to article's label.
    
    e.g.:
    
    test_data['WP9_December2012-February2013_1-500.txt_292'] = 1
    
    etc...
    
    This is kind of weird but it's simplest/most consistent way I could think of.
    
    returns:
        train_text and test_text, which are lists of article text
        train_labels and test_labels, which are lists of labels
    """

    sample = pd.read_csv(data, header=None, quoting=3, sep=chr(1))

    with open(train) as f:
        train_data = dict(json.load(f))

    with open(test) as f:
        test_data = dict(json.load(f))

    train_text = []
    train_labels = []
    test_text = []
    test_labels = []

    for d in sample.iterrows():
        key = int(d[1][0])
        text = d[1][11] + ' ' + d[1][23]
        text = text.replace('\\N', '')
        if key in train_data:
            if not IGNORE_ALL_NEGATIVE or max(train_data[key]) > 0:
                train_text.append(text)
                train_labels.append(train_data[key])
        elif key in test_data:
            if not IGNORE_ALL_NEGATIVE or max(test_data[key]) > 0:
                test_text.append(text)
                test_labels.append(test_data[key])

    return train_text, train_labels, test_text, test_labels


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("\nUsage: \n\tpython analysis.py DATA TRAIN TEST\n\n\t"
              "DATA: source chr(1)-separated .csv file that contains all metadata"
              "\n\tTRAIN: .json file with train labels"
              "\n\tTEST: .json file with test labels"
              "\n\nExample:"
              "\n\tpython train.py ../test_in.csv examples/train_labels.json examples/test_labels.json\n")
    else:
        main(*sys.argv[1:])
