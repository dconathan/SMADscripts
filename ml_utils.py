"""
Common machine learning scripts and utilities for SMADscripts
"""
import numpy as np
import pandas as pd
import json
import nltk
import pickle
import re
import random
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
try:
    from gensim.models.word2vec import Word2Vec
except ImportError:
    print("install the gensim package to use word2vec-based models")
    Word2Vec = False
import matplotlib.pyplot as plt
import csv
import os
import time
import sys
import select


VERBOSE = 1

sep = chr(1)
quoting = csv.QUOTE_NONE

# WORD2VEC_MODEL = os.path.join(os.path.dirname(__file__), 'twitter_guns_w2v.json')
WORD2VEC_MODEL = os.path.join(os.path.dirname(__file__), 'news_guns_w2v.json')
USE_WORD2VEC = True

if USE_WORD2VEC:
    with open(WORD2VEC_MODEL) as f:
        WORD2VEC_MODEL = json.load(f)
    if isinstance(WORD2VEC_MODEL, dict):
        WORD2VEC_MODEL = WORD2VEC_MODEL.items()
    WORD2VEC_MODEL = dict([(k, np.array(v)) for k, v in WORD2VEC_MODEL])


def analyze(train_text, train_labels, test_text, test_labels, classes_to_analyze=(0,), threshold=.5):
    i = 0

    if not os.path.exists('analysis'):
        os.mkdir('analysis')

    for cls in classes_to_analyze:

        params = get_model(cls)
        these_train_labels = [l[cls] for l in train_labels]
        these_test_labels = [l[cls] for l in test_labels]

        t0 = time.time()

        n_train = len(train_text)
        n_test = len(test_text)

        if params['features'] == 'text2vec':
            features = Text2Vec
            params['feature_params']['word2vec'] = WORD2VEC_MODEL
        elif params['features'] == 'tfidf':
            features = TfidfVectorizer

        # get feature matrix
        X = get_features(train_text + test_text, features, tokenizer=tokenize, **params['feature_params'])
        X_train = X[:n_train]
        X_test = X[n_train:]
        del X

        svm = SVC(class_weight='balanced', kernel='linear', probability=True)

        svm.set_params(**params['clf_params'])

        svm.fit(X_train, these_train_labels)

        results = dict()

        predicted_scores = svm.predict_proba(X_test)[:, 1]
        results['auc'] = float(roc_auc_score(these_test_labels, predicted_scores))

        predicted_labels = (predicted_scores > threshold).astype(int)

        results['recall'] = float(recall_score(these_test_labels, predicted_labels))
        results['precision'] = float(precision_score(these_test_labels, predicted_labels))
        results['f1'] = float(f1_score(these_test_labels, predicted_labels))

        fpr, tpr, thresholds = roc_curve(these_test_labels, predicted_scores)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], ls='--', color='grey')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')

        roc_data = list(zip(fpr, tpr, thresholds))
        n_roc_data = len(roc_data)
        if n_roc_data > 5:
            roc_data = [roc_data[i] for i in range(1, n_roc_data-1, n_roc_data//5)]

        plt.savefig('analysis/analysis_roc_class{}.png'.format(cls))

        for fpr, tpr, threshold in roc_data:
            plt.annotate('{:.3f}'.format(threshold), (fpr, tpr))

        plt.savefig('analysis/analysis_roc_class{}_thresholds.png'.format(cls))

        print("Analyzed class {}.  Iteration took {}s".format(cls, time.time() - t0))

        with open('analysis/analysis_class{}.txt'.format(cls), 'w') as f:
            json.dump(results, f, indent=2)


def train(train_text, train_labels, test_text, test_labels, classes_to_train=(0,)):

    if not os.path.exists('models'):
        os.mkdir('models')

    i = 0

    while True:

        class_to_train = classes_to_train[i % len(classes_to_train)]

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

        if np.random.rand() > .5 and USE_WORD2VEC:
            features = Text2Vec
            params['features'] = 'text2vec'
            params['feature_params']['word2vec'] = WORD2VEC_MODEL
        else:
            features = TfidfVectorizer
            params['features'] = 'tfidf'

        # get feature matrix
        t1 = time.time()
        X = get_features(train_text + test_text, features, tokenizer=tokenize, **params['feature_params'])
        print("Getting features took {}s".format(time.time() - t1))

        # this makes it so the search always uses the "test" part of the set as it's scoring part
        folds = [-1] * n_train + [0] * n_test
        split = PredefinedSplit(folds)

        # ranges/options for random svm parameters
        clf_params = {'C': np.logspace(-3, 2, 256)}
        #clf = SVC(class_weight='balanced', kernel='linear')
        clf = LinearSVC(class_weight='balanced')

        # perform the search
        t1 = time.time()
        #search = RandomizedSearchCV(clf, clf_params, n_iter=4, cv=split, n_jobs=1, scoring='roc_auc').fit(X, these_train_labels + these_test_labels)
        search = GridSearchCV(clf, clf_params, cv=split, n_jobs=-1, scoring='roc_auc').fit(X, these_train_labels + these_test_labels)
        print("Grid search took {}s".format(time.time() - t1))

        # grab the best performing classifier/params/score
        clf = search.best_estimator_
        params['clf_params'] = {'C': clf.get_params()['C']}
        params['score'] = search.best_score_

        # if better, save, etc.
        if search.best_score_ > best_score:
            print("New high score!")
            best_params = params
            best_score = search.best_score_
            if 'word2vec' in best_params['feature_params']:
                del best_params['feature_params']['word2vec']
            with open(params_file, 'w') as f:
                json.dump(best_params, f, indent=2)

        print("Trained class {}.  Iteration took {}s".format(class_to_train, time.time() - t0))
        i += 1


def score(train_text, train_labels, score_text, score_ids, classes_to_score=(0,)):

    if not os.path.exists('scores'):
        os.mkdir('scores')

    for cls in classes_to_score:

        params = get_model(cls)
        these_train_labels = [l[cls] for l in train_labels]

        n_train = len(train_labels)

        if params['features'] == 'text2vec':
            features = Text2Vec
        elif params['features'] == 'tfidf':
            features = TfidfVectorizer
            params['feature_params']['word2vec'] = 'twitter_guns_w2v.pickle'

        # Turn all text into features
        X = get_features(train_text + score_text, features, tokenizer=tokenize, **params['feature_params'])

        # X_train is all that we have labels for
        X_train = X[:n_train]

        svm = SVC(class_weight='balanced', kernel='linear', probability=True)

        svm.set_params(**params['clf_params'])

        # Fit the classifier on our training set
        svm.fit(X_train, these_train_labels)

        # Score all text
        score = svm.predict_proba(X)[:, 1]

        scored_samples = {}

        for i, s in zip(score_ids, score):
            scored_samples[i] = s

        out_filename = 'scores/scored_class{}.json'.format(cls)

        with open(out_filename, 'w') as f:
            json.dump(scored_samples, f, indent=2)


def get_features(raw_text, transformer, pca_d=None, **kwargs):
    """
    Turns raw_text into a feature matrix.
    
    if pca_d is specified (and is larger than # of examples and # of words, PCA is performed)
    
    Note: Tfidf outputs a sparse matrix, but PCA only accepts dense matrices.
    So if PCA is not used, this will return a sparse matrix, but otherwise
    it will be dense.
    """

    X = transformer(**kwargs).fit_transform(raw_text)
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
    params['min_df'] = 5

    return params


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


def load_data(data, train, test=None, text_key=(11, 23), id_key=(0,), scoring=False):
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
    t0 = time.time()

    if not isinstance(id_key, (list, tuple)):
        id_key = [id_key]
    if not isinstance(text_key, (list, tuple)):
        text_key = [text_key]

    id_key = [str(k) for k in id_key]
    text_key = [str(k) for k in text_key]

    if data.endswith('json'):
        with open(data) as f:
            main_data = json.load(f)
    else:
        main_data = pd.read_csv(data, header=None, quoting=quoting, sep=sep).fillna('')
        main_data.columns = [str(k) for k in main_data.columns]
        main_data = main_data.to_dict(orient='records')

    with open(train) as f:
        train_data = dict(json.load(f))
        train_data = {str(k): v for k, v in train_data.items()}

    if test is not None:
        with open(test) as f:
            test_data = dict(json.load(f))
            test_data = {str(k): v for k, v in test_data.items()}

    else:
        test_data = dict()

    train_text = []
    train_labels = []
    test_text = []
    test_labels = []
    all_text = []
    all_ids = []

    for d in main_data:
        key = '_'.join([str(d[k]) for k in id_key])
        text = ' '.join([str(d[k]) for k in text_key])
        text = text.replace('\\N', '')
        if key in train_data:
            train_text.append(text)
            label = train_data[key]
            if not isinstance(label, list):
                label = [label]
            train_labels.append(label)
        elif key in test_data:
            test_text.append(text)
            label = test_data[key]
            if not isinstance(label, list):
                label = [label]
            test_labels.append(label)
        if scoring:
            all_text.append(text)
            all_ids.append(key)

    n_train = len(train_text)
    n_test = len(test_text)
    n_all = len(all_text)

    # if this fails something is effed up
    assert len(train_labels) == n_train
    assert len(test_labels) == n_test
    assert len(all_ids) == n_all

    if VERBOSE:
        print("Loaded {} train examples and {} test examples. Took {}s".format(n_train, n_test, time.time() - t0))

    if not scoring:
        return train_text, train_labels, test_text, test_labels
    else:
        return train_text + test_text, train_labels + test_labels, all_text, all_ids


MENTION_REGEX = re.compile(r'@\w+')
RETWEET_REGEX = re.compile(r'\brt\b')
URL = re.compile(r'\bhttp\S+\b')
REPLACE_WITH_SPACE = re.compile(r'[!/\\()*\-$=%’\?,"‘;:“]')
DELETE = re.compile(r"['.]")

w_tokenizer = nltk.TreebankWordTokenizer()

w_tokenizer.ENDING_QUOTES = [(re.compile(r'"'), " '' "),
        (re.compile(r'(\S)(\'\')'), r'\1 \2 ')]
w_tokenizer.PUNCTUATION = [
        (re.compile(r'([:,])([^\d])'), r' \1 \2'),
        (re.compile(r'([:,])$'), r' \1 '),
        (re.compile(r'\.\.\.'), r' ... '),
        (re.compile(r'[;@$%&]'), r' \g<0> '),
        (re.compile(r'([^\.])(\.)([\]\)}>"\']*)\s*$'), r'\1 \2\3 '),
        (re.compile(r'[?!]'), r' \g<0> '),
        (re.compile(r"([^'])' "), r"\1 ' ")]


def clean_tweet(t):
    t = t.lower()
    t = URL.sub(' ', t)
    t = MENTION_REGEX.sub(' ', t)
    t = RETWEET_REGEX.sub(' ', t)
    t = REPLACE_WITH_SPACE.sub(' ', t)
    t = DELETE.sub('', t)
    t = t.strip()
    return t


def tokenize(t):
    return w_tokenizer.tokenize(clean_tweet(t))


class Text2Vec(TfidfVectorizer):
    def __init__(self, word2vec, **kwargs):
        assert Word2Vec, 'the gensim package is necessary to use Text2Vec/word2vec-based models'
        self._word2vec = word2vec
        if isinstance(self._word2vec, str):
            with open(self._word2vec, 'rb') as f:
                self._word2vec = pickle.load(f)
        if isinstance(self._word2vec, Word2Vec):
            self._word2vec = dict(zip(self._word2vec.wv.index2word, self._word2vec.wv.syn0))
        self._word2vecd = len(list(self._word2vec.values())[0])
        super(Text2Vec, self).__init__(**kwargs)

    def _make_word_matrix(self):
        self._word_matrix = np.zeros((len(self.vocabulary_), self._word2vecd))
        for k, v in self.vocabulary_.items():
            if k in self._word2vec:
                self._word_matrix[v, :] = self._word2vec[k]

    def fit(self, raw_documents, y=None):
        super(Text2Vec, self).fit(raw_documents)
        self._make_word_matrix()
        return self

    def fit_transform(self, raw_documents, y=None):
        X = super(Text2Vec, self).fit_transform(raw_documents)
        self._make_word_matrix()
        return X.dot(self._word_matrix)

    def transform(self, raw_documents, copy=True):
        X = super(Text2Vec, self).transform(raw_documents)
        return X.dot(self._word_matrix)


def tests():
    assert clean_tweet('"rt http:url.com @YOURFACE hello world"') == 'hello world'
    assert tokenize('"rt http:url.com @YOURFACE hello world"') == ['hello', 'world']
