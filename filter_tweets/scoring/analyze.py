import os
import sys
DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(DIR)))
from ml_utils import load_data
from ml_utils import analyze


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


def main(data, train_data, test_data):

    classes_to_analyze = [0, 1, 2, 3, 4]

    train_text, train_labels, test_text, test_labels = load_data(data, train_data, test_data, id_key='__index__', text_key='text')
    train_text, train_labels = fix_data(train_text, train_labels)
    test_text, test_labels = fix_data(test_text, test_labels)
    analyze(train_text, train_labels, test_text, test_labels, classes_to_analyze)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("\nUsage: \n\tpython analyze.py DATA TRAIN TEST\n\n\t"
              "DATA: see ml_utils.load_data for more info"
              "\n\tTRAIN: .json file with train labels"
              "\n\tTEST: .json file with test labels"
              "\n\nExample:"
              "\n\tpython analyze.py ../test_in.csv examples/train_labels.json examples/test_labels.json\n")
    else:
        main(*sys.argv[1:])
