import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
from scipy.sparse import issparse
import json
import random
import sys
import select
import numpy as np
import os
import time
from utils import Text2Vec
from utils import tokenize

CLASSES_TO_TRAIN = [0, 1, 2, 3]


def main(data, train, test):

    train_text, train_labels, test_text, test_labels = load_data(data, train, test)

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0

    while True:

        class_to_train = CLASSES_TO_TRAIN[i % len(CLASSES_TO_TRAIN)]

        these_train_labels = [l[class_to_train] for l in train_labels]
        these_test_labels = [l[class_to_train] for l in test_labels]

        params_file = 'models/params_class{}.json'.format(class_to_train)

        if os.path.exists(params_file):
            with open(params_file) as f:
                best_params = json.load(f)
            best_score = best_params['score']
        else:
            best_score = 0

        t0 = time.time()

        # this is what stops the loop
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            break

        if i % 5 == 0:
            print("Optimizing classifier... press any key to stop after current iteration...")

        # generate feature_params
        params = dict()
        params['feature_params'] = get_feature_params()

        n_train = len(train_text)
        n_test = len(test_text)

        if np.random.rand() > .5:
            features = TfidfVectorizer
            params['features'] = 'tfidf'
        else:
            features = Text2Vec
            params['features'] = 'text2vec'
            params['feature_params']['word2vec'] = 'twitter_guns_w2v.pickle'

        print(params)
        # get feature matrix
        t1 = time.time()
        X = get_features(train_text + test_text, features, tokenizer=tokenize, **params['feature_params'])
        print("Getting features took {}s".format(time.time() - t1))

        # this makes it so the search always uses the "test" part of the set as it's scoring part
        folds = [-1] * n_train + [0] * n_test
        split = PredefinedSplit(folds)

        # ranges/options for random svm parameters
        clf_params = {'C': np.logspace(-3, 4, 100000)}
        clf = LinearSVC()

        # perform the search
        t1 = time.time()
        search = RandomizedSearchCV(clf, clf_params, n_iter=64, cv=split, n_jobs=-1).fit(X, these_train_labels + these_test_labels)
        print("Grid search took {}s".format(time.time() - t1))

        # grab the best performing classifier/params/score
        clf = search.best_estimator_
        params['clf_params'] = clf.get_params()
        params['score'] = search.best_score_

        # if better, save, etc.
        if search.best_score_ > best_score:
            print("New high score!")
            best_params = params
            best_score = search.best_score_
            with open(params_file, 'w') as f:
                json.dump(best_params, f)

        print("Trained class {}.  Iteration took {}s".format(class_to_train, time.time() - t0))
        i += 1


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


def get_feature_params():
    """
    Returns a random set of reasonable parameters for get_features,
    Used for random grid search.
    """

    params = dict()
    params['pca_d'] = random.randint(20, 150)
    params['stop_words'] = random.choice([None, 'english'])
    params['max_features'] = random.randint(500, 20000)
    #params['ngram_range'] = random.choice([(1, 1), (1, 2), (2, 2)])

    params['min_df'] = 5

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
            train_text.append(text)
            train_labels.append(train_data[key])
        elif key in test_data:
            test_text.append(text)
            test_labels.append(test_data[key])

    return train_text, train_labels, test_text, test_labels


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
