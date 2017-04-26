from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import json
import os
import sys


def main(data, train, test=None):

    params = get_model()
    train_labels, sample = load_data(data, train, test)
    n_train = len(train_labels)
    all_text = [s['article'] for s in sample]

    # Turn all text into features
    X = get_features(all_text, **params['feature_params'])

    # X_train is all that we have labels for
    X_train = X[:n_train]

    svm = SVC()

    # This lets us use svm.predict_proba
    params['clf_params']['probability'] = True

    svm.set_params(**params['clf_params'])

    # Fit the classifier on our training set
    svm.fit(X_train, train_labels)

    # Score all text
    score = svm.predict_proba(X)[:, 1]

    for d, s in zip(sample, score):
        d['score'] = s

    with open(data + '.scored', 'w') as f:
        json.dump(sample, f)


def get_model():
    """
    Loads the parameters that we got from train.py
    """
    if not os.path.exists('models/params.json'):
        print("models/params.json not found.  Run train.py first.")
        sys.exit()
    with open('models/params.json') as f:
        params = json.load(f)

    return params


def get_features(raw_text, pca_d, **kwargs):
    """
    See train.py
    """

    X = TfidfVectorizer(**kwargs).fit_transform(raw_text)
    if pca_d is not None and pca_d < min(*X.shape):
        X = PCA(pca_d).fit_transform(X.toarray())

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
        sample = json.load(f)

    with open(train) as f:
        train_data = json.load(f)

    if test is not None:
        with open(test) as f:
            test_data = json.load(f)
    else:
        test_data = {}

    train_labels = []
    out = []

    for d in sample:
        key = d['source'] + '_' + str(d['source_index'])
        if key in train_data:
            train_labels.append(train_data[key])
            out.append(d)
        elif key in test_data:
            train_labels.append(test_data[key])
            out.append(d)

    for d in sample:
        key = d['source'] + '_' + str(d['source_index'])
        if key not in train_data and key not in test_data:
            out.append(d)

    return train_labels, out


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("\nUsage: \n\tpython score.py DATA TRAIN [TEST]\n\n\t"
              "DATA:\t\tsource .json file that contains all metadata to score"
              "\n\tTRAIN: .json file with train labels"
              "\n\tTEST: .json file with test labels (optional)"
              "\n\nExample:"
              "\n\tpython score.py examples/sample.json examples/train_labels.json examples/test_labels.json\n")
    else:
        main(*sys.argv[1:])
