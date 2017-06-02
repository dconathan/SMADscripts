import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
from ml_utils import load_data
from ml_utils import score


def fix_data(text, labels):

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


def main(data, train, test=None):

    classes_to_score = [0, 1, 2, 3]
    train_text, train_labels, score_text, score_ids = load_data(data, train, test, classes_to_score, scoring=True)
    train_text, train_labels = fix_data(train_text, train_labels)
    score(train_text, train_labels, score_text, score_ids)


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
