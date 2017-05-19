from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.word2vec import Word2Vec
import pickle
import numpy as np
import re
import nltk


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


def clean(t):
    t = t.lower()
    t = URL.sub(' ', t)
    t = MENTION_REGEX.sub(' ', t)
    t = RETWEET_REGEX.sub(' ', t)
    t = REPLACE_WITH_SPACE.sub(' ', t)
    t = DELETE.sub('', t)
    t = t.strip()
    return t


def tokenize(t):
    return w_tokenizer.tokenize(clean(t))


class Text2Vec(TfidfVectorizer):
    def __init__(self, word2vec, **kwargs):
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


assert clean('"rt http:url.com @YOURFACE hello world"') == 'hello world'
assert tokenize('"rt http:url.com @YOURFACE hello world"') == ['hello', 'world']
