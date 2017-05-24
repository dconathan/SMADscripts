from gensim.models.word2vec import Word2Vec
from ml_utils import tokenize
import json
import sys
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def preprocess(filename):

    with open(filename) as f:
        data = json.load(f)

    data = [d['article'] for d in data]
    data = [tokenize(d) for d in data]

    return data


def main(filename):

    out_filename = filename + '.w2v.json'

    model = Word2Vec(preprocess(filename), size=200, iter=100, workers=4)

    word2vec = dict(zip(model.wv.index2word, model.wv.syn0.tolist()))

    with open(out_filename, 'w') as f:
        json.dump(word2vec, f)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("blah")
    else:
        main(sys.argv[1])