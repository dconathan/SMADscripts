from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit
import json
import random
import sys
import select
import numpy as np
import os
import time


def main(data, train, test):
    """
    This performs a random search optimization to find the best parameters both for
    feature generation AND svm classification
    
    Basically it will keep looping indefinitely until you press a key, always saving the parameters
    if it finds the best score.
    
    Parameters are automatically saved and automatically loaded by score.py
    """

    train_text, train_labels, test_text, test_labels = load_data(data, train, test)

    if not os.path.exists('models'):
        os.mkdir('models')

    if os.path.exists('models/params.json'):
        with open('models/params.json') as f:
            best_params = json.load(f)
        best_score = best_params['score']
    else:
        best_score = 0

    i = 0

    while True:

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

        # get feature matrix
        X = get_features(train_text + test_text, **params['feature_params'])

        # this makes it so the search always uses the "test" part of the set as it's scoring part
        folds = [-1] * n_train + [0] * n_test
        split = PredefinedSplit(folds)

        # ranges/options for random svm parameters
        clf_params = {'C': np.logspace(-3, 4, 100000), 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'degree': range(2, 8)}
        clf = SVC()

        # perform the search
        search = RandomizedSearchCV(clf, clf_params, n_iter=64, cv=split, n_jobs=-1).fit(X, train_labels + test_labels)

        # grab the best performing classifier/params/score
        clf = search.best_estimator_
        params['clf_params'] = clf.get_params()
        params['score'] = search.best_score_

        # if better, save, etc.
        if search.best_score_ > best_score:
            print("New high score!")
            best_params = params
            best_score = search.best_score_
            with open('models/params.json', 'w') as f:
                json.dump(best_params, f)

        print("Iteration took {}s".format(time.time() - t0))
        i += 1


def get_features(raw_text, pca_d, **kwargs):
    """
    Turns raw_text into a feature matrix.
    
    if pca_d is specified (and is larger than # of examples and # of words, PCA is performed)
    
    Note: Tfidf outputs a sparse matrix, but PCA only accepts dense matrices.
    So if PCA is not used, this will return a sparse matrix, but otherwise
    it will be dense.
    """

    X = TfidfVectorizer(**kwargs).fit_transform(raw_text)
    if pca_d is not None and pca_d < min(*X.shape):
        X = PCA(pca_d).fit_transform(X.toarray())

    return X


def get_feature_params():
    """
    Returns a random set of reasonable parameters for get_features,
    Used for random grid search.
    """

    params = dict()
    params['pca_d'] = random.randint(10, 200)
    params['stop_words'] = random.choice([None, 'english'])
    params['max_features'] = random.randint(500, 20000)
    params['ngram_range'] = random.choice([(1, 1), (1, 2), (2, 2)])

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

    with open(data) as f:
        sample = json.load(f)

    with open(train) as f:
        train_data = json.load(f)

    with open(test) as f:
        test_data = json.load(f)

    train_text = []
    train_labels = []
    test_text = []
    test_labels = []

    for d in sample:
        key = d['source'] + '_' + str(d['source_index'])
        if key in train_data:
            train_text.append(d['article'])
            train_labels.append(train_data[key])
        elif key in test_data:
            test_text.append(d['article'])
            test_labels.append(test_data[key])

    return train_text, train_labels, test_text, test_labels


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("\nUsage: \n\tpython train.py DATA TRAIN TEST\n\n\t"
              "DATA: source .json file that contains all metadata"
              "\n\tTRAIN: .json file with train labels"
              "\n\tTEST: .json file with test labels"
              "\n\nExample:"
              "\n\tpython train.py examples/sample.json examples/train_labels.json examples/test_labels.json\n")
    else:
        main(*sys.argv[1:])
