from pandas.core.series import Series
import re
import csv
import random
import pandas as pd
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import SnowballStemmer
from nltk.parse.stanford import StanfordDependencyParser
from sklearn.base import TransformerMixin
import json as _json
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import pickle
from gensim.models.tfidfmodel import TfidfModel
from gensim.models.word2vec import Word2Vec
from gensim import corpora
from gensim.matutils import corpus2csc
import datetime
import os
import multiprocessing
from nltk.metrics.agreement import AnnotationTask
from nltk.corpus import words
from functools import partial
from itertools import repeat
from itertools import takewhile
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from scipy import sparse
from collections import deque
from itertools import compress

# TODO this is getting out of hand ... break into modules?

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

W2V_MODEL = '/home/devin/smad/guns/news/classification/models/word2vec.pickle'
ENGLISH_WORDS = set(words.words())

nonword_regex = re.compile(r'\W+')
nonletter_regex = re.compile(r'[^a-zA-Z0-9# ]+')
letter_regex = re.compile(r'[a-zA-Z0-9#@]+')  #includes # for hashtags and @ for mentions
mention_regex = re.compile(r'@\w+')
http_regex = re.compile(r'(?i)http[\S]+')
hashtag_regex = re.compile(r'#\w+')
retweet_regex = re.compile(r'(?i)\brt\b')
# TODO this might change depending on news source...
start_regex = re.compile(r'LENGTH:[^\n]+')
end_regex = re.compile(r'LOAD-DATE:[^\n]+')
whitespace_regex = re.compile(r'\s+')
news_meta_regex = re.compile(r'[A-Z-]+: [^\n]+')
email_address_regex = re.compile(r'\b\S+@\S+.com\b')
news_location_regex = re.compile(r'\n\n[A-Za-z -:]+ -+\s')
recipe_regex = re.compile(r'\scups?\s|\steaspoons?\s|\stablespoons?\s|\s[0-9]+ servings?\b')
fraction_regex = re.compile(r'([0-9]+)/([0-9]+)')
bad_characters = re.compile(r'[^a-zA-Z\'\-\s.,0-9:$?!@()/;#]')
drawing_regex = re.compile(r'DRAWING[^\n]+')
editor_regex = re.compile(r'To the Editor: ')
ending_regex = re.compile(r'Follow The New York Times[^\n]+\n[^\n]+')
author_regex = re.compile(r'[A-Z ]+\n')
location_regex = re.compile(r'\w,? ?\w? ?\w?.?\w?.?\n')
highlight_regex = re.compile(r'HIGLIGHT: ')


parser = StanfordDependencyParser()
ST_FILE = os.path.join(os.environ.get('NLTK_DATA', ''),
        'custom_models/170225_punkt_sentence_tokenizer_news.pickle')
if os.path.exists(ST_FILE):
    s_tokenizer = load_pickle(ST_FILE)
else:
    print("{} not found. Using default sentence tokenizer, \
            which won't tokenize things like U.S. or U.N. correctly.")
    s_tokenizer = PunktSentenceTokenizer()
w_tokenizer = TreebankWordTokenizer()
stemmer = SnowballStemmer('english')

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


class LineIterator:
    def __init__(self, source, process=None):
        self.source = source
        self.length = None
        self.iter_ = self.__iter__()
        if process is None:
            def _process(doc):
                return doc
        else:
            if not isinstance(process, (list, tuple)):
                process = [process]
            def _process(doc):
                for f in process:
                    doc = f(doc)
                return doc
        self.process = _process

    def __iter__(self):
        with open(self.source) as f:
            for line in f:
                yield self.process(line.strip())

    def __str__(self):
        return self.source

    def __len__(self):
        if self.length is None:
            self.length = count_lines(self.source)
        return self.length

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        j = 0

        if i < 0:
            i %= len(self)
        elif i >= len(self):
            raise IndexError("Can't get line {}, only {} lines in source".format(i, len(self)))
        iter_ = self.__iter__()
        t = next(iter_)
        while j < i:
            t = next(iter_)
            j += 1
        return t

    def __next__(self):
        return next(self.iter_)


ITERS = (list, tuple, Series, LineIterator)


VERBOSE = 1




NUM_CPUS = multiprocessing.cpu_count()


def split_by_sentence(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            return parallelize(split_by_sentence, doc)
        else:
            return [split_by_sentence(d, False) for d in doc]
    doc = s_tokenizer.tokenize(doc)
    return doc


def sql_to_header(sql_file):
    with open(sql_file) as f:
        data = f.read()

    data = data.lower()
    data = data.split('select')[1].split('from')[0].split(',\n')

    out = ''

    for d in data:
        d = d.strip()
        if d != '':
            if d[-1] == ',':
                d = d[:-1]
            if 'regexp_replace' in d:
                d = d.split('(')[1].split(',')[0]
            d = d.replace('`', '')
            out += d + '\t'

    out = out[:-1] + '\n'

    with open(sql_file + '.header', 'w') as f:
        f.write(out)


def fix_dataframe(filename, sep=chr(1)):
    if os.path.isdir(filename):
        out_filename = filename.replace('/', '') + '.csv'
        filename = get_all_files(filename)
    elif os.path.isfile(filename):
        out_filename = filename + ".csv"
        filename = [filename]
    else:
        assert False, "need valid file or directory"

    data = pd.DataFrame()
    for f in filename:
        data = data.append(pd.read_csv(f, sep=sep, header=None, quoting=csv.QUOTE_NONE))
    data.to_csv(out_filename, sep='\t', quoting=csv.QUOTE_ALL, index=False, header=None)


def tokenize(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            return parallelize(tokenize, doc)
        else:
            return [tokenize(d, False) for d in doc]
    doc = doc.replace(chr(1), ' ')
    doc = whitespace_regex.sub(' ', doc)
    doc = w_tokenizer.tokenize(doc.lower())
    doc = [nonletter_regex.sub('', d) for d in doc]
    doc = [d for d in doc if d]
    while doc and doc[0] == 'user':
        doc = doc[1:]
    return doc


def split(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            return parallelize(split, doc)
        else:
            return [split(d, False) for d in doc]
    doc = doc.split()
    return doc


def clean_tweet(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            return parallelize(clean_tweet, doc)
        else:
            return [clean_tweet(d, False) for d in doc]
    doc = mention_regex.sub('@user', doc)
    doc = retweet_regex.sub('', doc)
    doc = http_regex.sub('httpurl', doc)
    return doc.strip()


def file_to_regex(filename):
    if not isinstance(filename, ITERS):
        assert os.path.isfile(filename), '{} does not exist'.format(filename)
        with open(filename) as f:
            raw_data = f.readlines()
        assert len(raw_data) > 0, '{} is empty'.format(filename)
    else:
        raw_data = filename
    clean_data = [d.strip() + 's?' for d in raw_data]
    regex = r'\b' + r'\b|\b'.join(clean_data) + r'\b'

    return re.compile(regex)


ENGLISH_THRESHOLD = 4


def filter_in_english(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize(filter_in_english, doc)
        else:
            doc = [filter_in_english(d, False) for d in doc]
        return [d for d in doc if d is not False]
    n = 0
    for word in doc.lower().replace(chr(1), ' ').replace('"', ' ').split():
        if word in ENGLISH_WORDS and word != 'user':
            n += 1
            if n == ENGLISH_THRESHOLD:
                return doc
    return False


def filter_out_english(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize(filter_out_english, doc)
        else:
            doc = [filter_out_english(d, False) for d in doc]
        return [d for d in doc if d is not False]
    n = 0
    for word in doc.lower().replace(chr(1), ' ').replace('"', ' ').split():
        if word in ENGLISH_WORDS and word != 'user':
            n += 1
            if n == ENGLISH_THRESHOLD:
                return False
    return doc



def filter_out_short(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize(filter_out_short, doc)
        else:
            doc = [filter_out_short(d, False) for d in doc]
        return [d for d in doc if d is not False]
    if len(doc.split()) < 5:
        return False
    else:
        return doc


def filter_out_long(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize(filter_out_long, doc)
        else:
            doc = [filter_out_long(d, False) for d in doc]
        return [d for d in doc if d is not False]
    if len(doc.split()) > 30:
        return False
    else:
        return doc


def filter_out_regex(regex, doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize((filter_out_regex, (regex,)), doc)
        else:
            doc = [filter_out_regex(regex, d, False) for d in doc]
        return [d for d in doc if d is not False]
    if bool(regex.search(doc.lower())):
        return False
    else:
        return doc


def filter_in_regex(regex, doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize((filter_in_regex, (regex,)), doc)
        else:
            doc = [filter_in_regex(regex, d, False) for d in doc]
        return [d for d in doc if d is not False]
    if bool(regex.search(doc.lower())):
        return doc
    else:
        return False


def filter_out_list(l, doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize((filter_out_list, (l,)), doc)
        else:
            doc = [filter_out_list(l, d, False) for d in doc]
        return [d for d in doc if d is not False]
    for w in l:
        if w in doc:
            return False
    return doc


def filter_in_list(l, doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize((filter_in_list, (l,)), doc)
        else:
            doc = [filter_in_list(l, d, False) for d in doc]
        return [d for d in doc if d is not False]
    for w in l:
        if w in doc:
            return doc
    return False


def stem(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            doc = parallelize(stem, doc)
        else:
            doc = [stem(d, False) for d in doc]
    return stemmer.stem(doc)


def join(doc, parallel=True):
    if isinstance(doc, ITERS) and doc and not isinstance(doc[0], str):
        if parallel:
            doc = parallelize(join, doc)
        else:
            doc = [join(d, False) for d in doc]
        return doc
    return ' '.join(doc)


def join_lines(doc, parallel=True):
    if isinstance(doc, ITERS) and doc and not isinstance(doc[0], str):
        if parallel:
            doc = parallelize(join_lines, doc)
        else:
            doc = [join_lines(d, False) for d in doc]
        return doc
    return '\n'.join(doc)


def save_json(l, filename):
    if not isinstance(l, ITERS):
        l = [l]
    with open(filename, 'w') as f:
        for j in l:
            f.write(_json.dumps(j) + '\n')


def load_json(filename):
    with open(filename) as f:
        d = f.readlines()
    try:
        d = [_json.loads(j) for j in d]
    except:
        pass
    if len(d) == 1:
        d = d[0]
    return d


def file_to_list(filename):
    with open(filename) as f:
        d = f.readlines()
    return [j.strip() for j in d]


def scatter_plot(X, num_to_plot=None, labels=None, filename=None):
    assert len(X.shape) == 2 and X.shape[1] >= 2
    if num_to_plot is not None:
        X = X[:num_to_plot]
    if X.shape[1] > 2:
        #X = TSNE(n_iter=5000, init='pca').fit_transform(X)
        X = PCA(2).fit_transform(X)
    if labels is not None and len(set(labels)) < 6:
        unique_labels = list(set(labels))
        colors = ['r', 'g', 'b', 'm', 'k']
    else:
        colors = None

    labeled = set()

    lw = .2
    fontsize = 10

    fig = plt.figure(figsize=(10,10), dpi=200)
    for i, x in enumerate(X):
        x0, x1 = x
        if colors is None:
            plt.scatter(x0, x1, lw=lw)
        elif labels[i] in labeled:
            plt.scatter(x0, x1, c=colors[unique_labels.index(labels[i])], lw=lw)
        else:
            plt.scatter(x0, x1, c=colors[unique_labels.index(labels[i])], label=labels[i], lw=lw)
            labeled.add(labels[i])
        if labels is not None and len(set(labels)) >= 6:
            if labels[i] == 'VBNdobjNN':
                plt.annotate(labels[i], xy=(x0, x1), xytext=(30, 2), textcoords='offset points', ha='right', va='bottom', fontsize=fontsize)
            else:
                plt.annotate(labels[i], xy=(x0, x1), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=fontsize)
    plt.axis('off')
    if colors is not None:
        plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()


def train_word2vec(filename, **kwargs):
    data = LineIterator(filename, tokenize)
    model = Word2Vec(data, workers=NUM_CPUS, **kwargs)
    save_pickle(model, filename + '.w2v.pickle')
    return model


class Dictionary(TransformerMixin):
    def __init__(self, docs=None, stopwords=None, size=None, archive=False):

        self.size = size
        self.archive = archive  # archive is necessary for doing stuff like skip-gram sets or transform_all()
        self.word2id = {'UNK': 0}
        self.id2word = ['UNK']
        self.doc_ids = None
        self.word_ids = None
        self.word2docfreq = {'UNK': 1}
        self.word2freq = {'UNK': 1}
        self.num_docs = 0
        self.num_words = 0
        if stopwords is None:
            self.stopwords = set()
        else:
            self.stopwords = set(stopwords)

        if docs is not None:
            self.fit(docs)
        self.index = 0

    def fit(self, docs, archive=None):
        if archive is not None:
            self.archive = archive
        self.word2docfreq['UNK'] = len(docs)
        assert docs, "argument to Dictionary.fit must not be empty"
        assert isinstance(docs[0], ITERS) and docs[0] and isinstance(docs[0][0], str),\
            "argument to Dictionary.fit must be a list of nonempty lists of tokens"
        for doc in docs:
            self.num_docs += 1
            doc_words = set()
            for word in doc:
                if word not in self.stopwords:
                    self.word2freq[word] = self.word2freq.get(word, 0) + 1
                    doc_words.add(word)
            for word in doc_words:
                self.word2docfreq[word] = self.word2docfreq.get(word, 0) + 1
        unk_freq = self.word2freq['UNK']
        del self.word2freq['UNK']
        self.id2word = sorted(self.word2freq.keys(), key=lambda x: -self.word2freq[x])
        self.id2word.insert(0, 'UNK')
        self.word2freq['UNK'] = unk_freq
        self.word2id = dict([(word, i) for i, word in enumerate(self.id2word)])
        self.num_words = len(self.id2word)
        if self.archive:
            self.doc_ids = list()
            self.word_ids = list()
            for i, doc in enumerate(docs):
                for word in doc:
                    if word in self.word2id:
                        self.doc_ids.append(i)
                        self.word_ids.append(self.word2id[word])
        if self.size is not None:
            self.prune()
        return self

    def prune(self, size=None):
        if size is None:
            size = self.size
        else:
            self.size = size
        if size < self.num_words:
            delete_ids = set(range(self.num_words)[size:])
            if self.archive:
                for i, word in enumerate(self.word_ids):
                    if word in delete_ids:
                        self.word_ids[i] = 0
            for i in delete_ids:
                word = self.id2word[i]
                self.id2word[i] = None
                self.word2freq['UNK'] += self.word2freq[word]
                del self.word2freq[word]
                del self.word2id[word]
                del self.word2docfreq[word]
            self.id2word = [w for w in self.id2word if w is not None]
            self.id2word = self.id2word[:size]
            self.num_words = size
            assert len(self.id2word) == len(self.word2id) == len(self.word2docfreq) == self.num_words

    def transform(self, doc, parallel=True):
        if isinstance(doc, str):
            doc = tokenize(doc)
        elif doc and isinstance(doc[0], ITERS):
            if parallel:
                return sparse.vstack(parallelize(self.transform, doc))
            else:
                return sparse.vstack([self.transform(d) for d in doc])
        if doc and isinstance(doc[0], str):
            doc = [self.word2id[word] if word in self.word2id else 0 for word in doc]
        return self._transform(doc)

    def _transform(self, doc):
        doc = sorted(doc)  # sorting makes constructing lil_matrix more efficient
        vec = sparse.lil_matrix((1, self.num_words))
        for word in doc:
            vec[0, word] += 1
        return normalize(vec)

    def transform_all(self):
        if not self.archive:
            assert False, "Fit must be run with archive=True in order to run transform_all()"
        docs = [[] for i in range(self.num_docs)]
        for doc_id, word_id in zip(self.doc_ids, self.word_ids):
            docs[doc_id].append(word_id)
        return self.transform(docs)

    def generate_batch(self, batch_size, window_size, mode='pv'):
        assert batch_size % window_size == 0, 'batch_size must be a multiple of window_size'
        if mode == 'pv':
            batch = np.zeros((batch_size, window_size + 1), dtype=np.int32)
            labels = np.zeros((batch_size, 1), dtype=np.int32)
            span = window_size + 1
            buffer_words = deque()
            buffer_docs = deque()
        for i in range(span):
            buffer_words.append(self.word_ids[self.index])
            buffer_docs.append(self.doc_ids[self.index])
            self.index += 1
            self.index %= len(self.word_ids)

        mask = [1] * span
        mask[-1] = 0
        for i in range(batch_size):
            if len(set(buffer_docs)) == 1:
                doc_id = buffer_docs[0]
                batch[i, :] = list(compress(buffer_words, mask)) + [doc_id]
                labels[i, 0] = buffer_words[-1]
            buffer_words.append(self.word_ids[self.index])
            buffer_docs.append(self.doc_ids[self.index])
            self.index += 1
            self.index %= len(self.word_ids)
        return batch, labels


class Tfidf(Dictionary):
    def __init__(self, docs=None, stopwords=None, pca_d=None, size=None, archive=False):
        if pca_d is None:
            self.pca = None
        else:
            self.pca = PCA(pca_d)
        super().__init__(docs, stopwords, size, archive)

    def fit(self, docs, archive=None):
        super().fit(docs, archive)
        if self.pca is not None and self.pca.get_params()['n_components'] > self.size:
            pass
        return self

    def _transform(self, doc):
        doc = sorted(doc)  # sorting makes constructing lil_matrix more efficient
        vec = sparse.lil_matrix((1, self.num_words))
        for word in doc:
            vec[0, word] += (1 + np.log(self.num_docs / self.word2docfreq[self.id2word[word]]))
        return normalize(vec)


class BagOfWords(TransformerMixin):
    def __init__(self, pca_d=None):
        self.pca_d = pca_d

    def fit(self, source, filter_extremes=False):
        self.source = source
        self.dictionary = corpora.Dictionary(source)
        if filter_extremes:
            self.dictionary.filter_extremes(keep_n=2000)
        self.tfidf = TfidfModel(dictionary=self.dictionary)
        self.num_terms = len(self.dictionary.items())
        self.num_docs = self.dictionary.num_docs
        self.word_mat = corpus2csc([self.tfidf[self.dictionary.doc2bow(t)] for t in source], num_terms=self.num_terms, num_docs=self.num_docs).transpose()
        self.stack = sparse.vstack
        if self.pca_d is not None and self.pca_d < self.num_terms:
            self.pca = TruncatedSVD(self.pca_d).fit(self.word_mat)
            self.word_mat = self.pca.transform(self.word_mat)
            self.stack = np.vstack
        else:
            self.pca = None
        # self.doc_index = dict([(v, k) for k, v in enumerate(source)])
        self.doc_index = {}
        return self

    def transform(self, text):
        if isinstance(text, ITERS):
            return self.stack([self.transform(d) for d in text])
        elif text in self.doc_index:
            return self.word_mat[self.doc_index[text], :]
        else:
            x = corpus2csc([self.tfidf[self.dictionary.doc2bow([text])]],num_terms=self.num_terms, num_docs=self.num_docs).transpose()
            if self.pca is not None:
                x = self.pca.transform(x)
            return x

class Text2VecSimple(TransformerMixin):
    def __init__(self, X=None, labels=None, w2v=None, unk=True, weights=None):
        self.unk = unk
        assert w2v is not None or (X is not None and labels is not None)
        if w2v is None:
            assert X.shape[0] == len(labels) == len(set(labels))
            self.X_dict = dict(zip(labels, X))
            self.d = X.shape[1]
        else:
            self.X_dict = w2v
            if isinstance(self.X_dict, str):
                self.X_dict = load_pickle(self.X_dict)
            self.d = self.X_dict.layer1_size
        self.dictionary = None

        self.weights = weights
        if self.weights is None:
            self.weights = 'mean'
        if self.weights == 'tfidf':
            
            def _get_weight(word):
                if word not in self.dictionary.word2freq:
                    word = 'UNK'
                return (1 + np.log(self.dictionary.word2freq[word]) * 
                        np.log(self.dictionary.num_docs/self.dictionary.word2docfreq[word]))
        else:
            def _get_weight(word):
                return 1

        self._get_weight = _get_weight


    def fit(self, text):
        self.dictionary = Dictionary(text)
        return self

    def transform(self, text):
        if isinstance(text, ITERS) and text and isinstance(text[0], ITERS):
            return np.vstack([self.transform(t) for t in text])
        v = np.zeros(self.d)
        n = 0
        for t in text:
            if t in self.X_dict:
                v += np.ravel(self.X_dict[t]) * self._get_weight(t)
                n += 1
            else:
                if 'UNK' in self.X_dict and self.unk:
                    v += self.X_dict['UNK'] * self._get_weight('UNK')
                    n += 1
        v /= max(1, n)
        return v


class Text2Vec(TransformerMixin):
    def __init__(self, word2vec=None, pca_d=None, remove_first=False, a=1):
        assert word2vec is not None, "Must supply word2vec model for Text2Vec"
        self.pca_d = pca_d
        self.a = a
        self.word2vec = word2vec
        if isinstance(self.word2vec, str):
            self.word2vec = load_pickle(self.word2vec)
        self.remove_first = remove_first
        self.pca = None
        self.u = None

    def fit(self, source):

        if not isinstance(source, ITERS):
            if os.path.exists(source):
                source = LineIterator(source, tokenize)
            else:
                source = tokenize(source)
        elif source and not isinstance(source[0], ITERS):
            source = tokenize(source)

        self.dictionary = Dictionary(source)
        self.d = self.word2vec.layer1_size

        word_mat = self.transform(source)

        if self.pca_d is not None and self.pca_d < self.d:
            self.pca = TruncatedSVD(self.pca_d).fit(word_mat)

        if self.remove_first:
            self.u = PCA(1).fit_transform(word_mat.T)

        return self

    def transform(self, doc, parallel=True):
        if isinstance(doc, ITERS) and doc and isinstance(doc[0], ITERS):
            if parallel:
                return np.vstack(parallelize(self.transform, doc))
            else:
                return np.vstack([self.transform(d) for d in doc])
        if isinstance(doc, str):
            doc = tokenize(doc)
        doc_v = np.zeros(self.d)
        doc_length = 0
        for word in doc:
            if word in self.word2vec and word in self.dictionary.word2freq:
                doc_length += 1
                doc_v += (self.a / (self.a + self.dictionary.word2freq[word] / self.dictionary.num_words)) * self.word2vec[word]
        doc_v /= max(doc_length, 1)

        if self.u is not None:
            doc_v -= np.dot(np.outer(self.u, self.u), doc_v)
        if self.pca is not None:
            doc_v = self.pca.transform(doc_v.reshape(1, -1))
        return doc_v


def word_cloud(doc, filename=None):
    def color_func(*args, **kwargs):
        c = random.randint(0, 100)
        return 'rgb({},{},{})'.format(c, c, c)
    if isinstance(doc, ITERS):
        doc = ' '.join(doc)
    cloud = WordCloud(background_color='white').generate(doc).recolor(color_func=color_func)
    if filename is None:
        plt.figure()
        plt.imshow(cloud)
        plt.axis('off')
        plt.show()
    else:
        filename = filename.replace('/', '_')
        cloud.to_file(filename)


def get_task_(*args):
    assert len({len(a) for a in args}) == 1, 'Number of coded objects must be the same'
    n = len(args[0])
    ids = list(range(n))
    data = []
    for i, a in enumerate(args):
        data += list(zip(['c{}'.format(i)]*n, ids, a))
    return AnnotationTask(data)


def scotts_pi(*args):
    return get_task_(*args).pi()


def krippendorff(*args):
    return get_task_(*args).alpha()


def partition(doc, n=NUM_CPUS):
    assert isinstance(doc, ITERS), 'arg to partition must be an iterable'
    step = (len(doc) // n) + 1
    return [doc[i:i+step] for i in range(0, len(doc), step)]


def parse_triples(doc, parallel=True):
    if isinstance(doc, ITERS) and doc and isinstance(doc[0], str):
        doc = [d if nonletter_regex.sub('', d) != '' else 'a' for d in doc]  # without this, empty strings get skipped rather than returning []
        doc = [fraction_regex.sub(r'\1\2', d) for d in doc]  # fractions screw up the parser...
        doc = [bad_characters.sub('', d) for d in doc]  # bad characters do too...
        if len(doc) < 32 or parallel is False:
            return [[t[0][1] + t[1] + t[2][1] for t in next(d).triples()] for d in parser.raw_parse_sents(doc)]
        else:
            return [i for s in parallelize(parse_triples, partition(doc)) for i in s]
    elif isinstance(doc, ITERS) and doc and isinstance(doc[0], ITERS):
        return [parse_triples(d) for d in doc]
    elif nonletter_regex.sub('', doc) == '':
        return []
    else:
        doc = bad_characters.sub('', doc)
        doc = fraction_regex.sub(r'\1\2', doc)
        return [t[0][1] + t[1] + t[2][1] for t in next(parser.raw_parse(doc)).triples()]


def parse_deps(doc, parallel=True):
    if isinstance(doc, ITERS):
        if parallel:
            return parallelize(parse_deps, doc)
        else:
            return [parse_deps(d, False) for d in doc]
    else:
        return next(parser.raw_parse(doc))
   

def extract_news(filename, parallel=True):
    if isinstance(filename, ITERS):
        if parallel:
            return [d for n in parallelize(extract_news, filename) for d in n]
        else:
            return [d for n in [extract_news(f) for f in filename] for d in n]
    with open(filename) as f:
        data = f.read()
    #return [whitespace_regex.sub(' ', end_regex.split(d)[0]).strip() for d in start_regex.split(data)[1:]]
    data = news_location_regex.sub('\n\n', data)
    data = email_address_regex.sub('\n', data)
    data = news_meta_regex.sub('\n', data)
    data = drawing_regex.sub('\n', data)
    #data = location_regex.sub('\n', data)
    data = author_regex.sub('\n', data)
    data = editor_regex.sub('\n', data)
    data = highlight_regex.sub('\n', data)
    data = ending_regex.sub('\n', data)
    return [whitespace_regex.sub(' ', d).strip() for d in data.split('\n\n\n') if len(d.split()) > 300 and len(recipe_regex.findall(d)) < 5]


pool = multiprocessing.Pool(processes=NUM_CPUS) 
def parallelize(f, arg):
    if isinstance(f, tuple):
        f, f_args = f
        out = pool.map(partial(f, *f_args, parallel=False), arg)
    else:
        out = pool.map(partial(f, parallel=False), arg)
    return out


def process(file_in, functions, file_out=None):
    if isinstance(file_in, str):
        if file_in[-1] == '/':
            file_in = file_in[:-1]
        if file_out is None:
            file_out = file_in + '.processed'
        if os.path.isdir(file_in):
            file_in = get_all_files(file_in)
        elif os.path.isfile(file_in):
            file_in = LineIterator(file_in)
    assert isinstance(file_in, ITERS), "First arg must be a valid file, directory, or iterable"
    if not isinstance(functions, ITERS):
        functions = [functions]
    output = []
    buff = []
    file_created = False
    CHUNK_SIZE = 2048
    t0 = time.time()
    checkpoint = 5

    def _process_buff(buff, output):
        for f in functions:
            buff = f(buff)
        if buff:
            if isinstance(buff[0], str):
                to_json = False
            else:
                to_json = True
            if file_out is None:
                output += buff
            elif not file_created:
                with open(file_out, 'w') as f:
                    if to_json:
                        f.writelines([_json.dumps(b)+'\n' for b in buff])
                    else:
                        f.writelines([b + '\n' for b in buff])
            else:
                with open(file_out, 'a') as f:
                    if to_json:
                        f.writelines([_json.dumps(b)+'\n' for b in buff])
                    else:
                        f.writelines([b + '\n' for b in buff])
            return output

    for i, doc in enumerate(file_in):
        buff.append(doc)
        if len(buff) == CHUNK_SIZE:
            output = _process_buff(buff, output)
            file_created = True
            buff = []
            timedelta = time.time() - t0
            if VERBOSE and timedelta > checkpoint:
                print("{0:d}/{1:d} processed. Estimated {2:.2f}s remaining".format(i, len(file_in),
                                                                         (timedelta / max(i, 1)) * (len(file_in) - i)))
                checkpoint += 5

    if buff:
        output = _process_buff(buff, output)

    if file_out is not None and not False:
        output = None
    return output


def parse_matrix(docs):
    parsed = parse_triples(docs)
    print(parsed)
    return Dictionary().fit_transform(parsed)


def clean_query(filename, headerfile=None):
    df = pd.read_csv(filename, sep=chr(1), header=None, quoting=csv.QUOTE_NONE).fillna('')
    if headerfile is None:
        header = None
    else:
        with open(headerfile) as f:
            header = f.read().split('\t')
    df.to_csv(filename + '.csv', sep='\t', quoting=csv.QUOTE_ALL, index=False, header=header)


def count_lines(filename):
    with open(filename, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        num_lines = sum(buf.count(b'\n') for buf in bufgen)
    return num_lines


def get_all_files(directory):
    assert os.path.isdir(directory), '{} is not a valid directory'.format(directory)
    return [os.path.abspath(os.path.join(directory, f)) for f in next(os.walk(directory))[2]]


def test_process():
    test = ['This is a really great test. This should end up as tokenized sentences. '
            'The next sentence will not make it. adsjfkl asdwtw awegb. But this one definitely will. '
            'The next one will not either. Short sentence.']
    output = process(test, [split_by_sentence, filter_in_english, filter_out_short, tokenize])
    print(output)
    output = process('/home/devin/data/fnc-1/train_stances.csv', [filter_in_english, filter_out_short, tokenize], 'test')


def test_agree():
    x1 = np.random.choice(2, 100)
    x2 = np.random.choice(2, 100)
    scotts_pi(x1, x2)
    krippendorff(x1, x2)


def test_regex():
    p = re.compile(r'test')
    test = ['this is a test', 'you better believe it', 'anything with t-e-s-t should filter out', 'test test test', 'it should work']
    filter_regex(p, 'this is a test')
    test = filter_regex(p, test)


def test_filter_list():
    test = ['this is a test', 'you better believe it', 'anything with t-e-s-t should filter out', 'test test test', 'it should work']
    filter_list(['test'], test)


def test_tokenize():
    test = ['this is a test', 'you better believe it', 'this should be run in parallel', 'it should work']
    tokenize(test)


def test_filters():
    test = ['this is a test', 'you better believe it', 'this should be run in parallel', 'it should work', 'adsfasdfawegf iosudios, aovwighy']
    assert(filter_in_english(test) == test[:-1])
    assert(filter_out_short(test)) == [test[2]]


def test_parse():
    test = ['this is a test', 'you better believe it', 'this should be run in parallel', 'it should work']
    print(parse_triples(test))


def test_parse_matrix():
    test = ['this is a test', 'you better believe it', 'it sure is']
    return parse_matrix(test)


def test_big_parse():
    t0 = time.time()
    datain = LineIterator('test_data/clean_news')[:10]
    print(parse_triples(datain))
    print("Parsing {} sentences took {}s".format(len(datain), time.time() - t0))


def test_dictionary():

    docs = [['this', 'is', 'a', 'test'],
            ['there', 'should', 'be', 'four', 'documents'],
            ['with', 'thirteen', 'unique', 'words'],
            ['unique', 'words', 'this', 'is', 'a', 'documents', 'test'],
            'i am willing to see what is going on over there in that house over the hill'.split()]

    d = Dictionary(docs)
    d.prune(5)

    docs = LineIterator('/home/devin/smad/test_data/clean_news')
    d = Dictionary(docs)
    d.prune(100)


def test_fileiterator():

    docs = LineIterator('/home/devin/smad/test_data/clean_news')
    len(docs)
    test = docs[0]
    test = docs[:5]
    for d in docs:
        if d in docs:
            break

def test_directory():
    print(get_all_files('.'))


def test_text2vec():
    w2v = '/home/devin/smad/guns/news/classification/models/word2vec.pickle'
    data = '/home/devin/data/smad/guns/news/clean/filtered_labeled170211.json'
    data = [d['text'] for d in json_to_list(data)]
    t2v = Text2Vec(w2v, pca_d=5, remove_first=True).fit(data[:20])
    X = t2v.transform(tokenize(data[25:30]))
    print(X)
    print(X.shape)

def test_all():
    test_tokenize()
    test_agree()
    test_filters()
    test_regex()
    test_filter_list()
    test_parse()


def main():
    test_text2vec()



if __name__ == '__main__':

    main()
