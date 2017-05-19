import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from utils import Text2Vec
from utils import tokenize
import json
import os
import sys

CLASSES_TO_SCORE = [0, 1, 2, 3]


def main(data, train, test=None):

    train_labels, sample = load_data(data, train, test)

    for cls in CLASSES_TO_SCORE:

        params = get_model(cls)
        these_train_labels = [l[cls] for l in train_labels]
        all_text = [t['text'] for t in sample]

        n_train = len(train_labels)

        if params['features'] == 'text2vec':
            features = Text2Vec
        elif params['features'] == 'tfidf':
            features = TfidfVectorizer
            params['feature_params']['word2vec'] = 'twitter_guns_w2v.pickle'

        # Turn all text into features
        X = get_features(all_text, features, tokenizer=tokenize, **params['feature_params'])

        # X_train is all that we have labels for
        X_train = X[:n_train]

        svm = LinearSVC()

        # This lets us use svm.predict_proba
        params['clf_params']['probability'] = True

        svm.set_params(**params['clf_params'])

        # Fit the classifier on our training set
        svm.fit(X_train, these_train_labels)

        # Score all text
        score = svm.predict_proba(X)[:, 1]

        scored_samples = {}

        for d, s in zip(sample, score):
            scored_samples[d['id']] = s

        out_filename = os.path.splitext(data)[0] + '_scored_class{}.json'.format(cls)

        with open(out_filename, 'w') as f:
            json.dump(scored_samples, f)


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


def get_features(raw_text, features, pca_d, **kwargs):
    """
    See train.py
    """

    X = features(**kwargs).fit_transform(raw_text)
    if pca_d is not None and pca_d < min(*X.shape):
        try:
            X = PCA(pca_d).fit_transform(X.toarray())
        except MemoryError:
            X = TruncatedSVD(pca_d).fit_transform(X)

    return X


def load_data(data, train, test=None):
    """
    NOTE: this is different than the load_data in train.py,
    since here we are interested in scoring ALL the data, not just the train/test data...
    
    Since we don't care about testing here, we put train and test into one train set.
    (because of this, test is optional)
    
    returns:
        train_labels (list of labels)
        out (list of article metadata such as the first articles in the list
            correspond to train_labels)
    """

    with open(data) as f:
        sample = pd.read_csv(data, header=None, quoting=3, sep=chr(1))

    with open(train) as f:
        train_data = dict(json.load(f))

    if test is not None:
        with open(test) as f:
            test_data = dict(json.load(f))
    else:
        test_data = {}

    train_labels = []
    out = []

    for d in sample.iterrows():
        key = int(d[1][0])
        text = d[1][11]+ ' ' + d[1][23]
        text = text.replace('\\N', '')
        if key in train_data:
            out.append({'id': key, 'text': text})
            train_labels.append(train_data[key])
        elif key in test_data:
            out.append({'id': key, 'text': text})
            train_labels.append(test_data[key])

    for d in sample.iterrows():
        key = int(d[1][0])
        text = d[1][11] + ' ' + d[1][23]
        text = text.replace('\\N', '')
        if key not in train_data and key not in test_data:
            out.append({'id': key, 'text': text})

    return train_labels, out


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("\nUsage: \n\tpython score.py DATA TRAIN [TEST]\n\n\t"
              "DATA:\t\tsource .json file that contains all metadata to score"
              "\n\tTRAIN: .json file with train labels"
              "\n\tTEST: .json file with test labels (optional)"
              "\n\nExample:"
              "\n\tpython score.py examples/sample.json examples/train_labels.json examples/test_labels.json\n")
    else:
        main(*sys.argv[1:])
