import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
from ml_utils import load_data
from ml_utils import score


def main(data, train, test=None):

    classes_to_score = [0, 1, 2, 3]
    train_text, train_labels, score_text, score_ids = load_data(data, train, test, classes_to_score, scoring=True)
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
