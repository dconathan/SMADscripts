from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
from scipy import sparse
from gensim.models import Word2Vec
import cPickle as pickle
import numpy as np


ITERS = (list, tuple)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def tokenize(doc):
    """
    Super simple tokenize function. Works on a single string or list of strings.
    """
    if isinstance(doc, ITERS):
        return [tokenize(d) for d in doc]
    return doc.lower().split()


class Dictionary(TransformerMixin):
    """
    Object for getting word counts, doc counts, creating BOW vectors.
    Will be used as a base for TF-IDF, etc.
    """
    def __init__(self, docs=None, stop_words=None, size=None):

        self.size = size
        self.word2id = {'UNK': 0}
        self.id2word = ['UNK']
        self.doc_ids = list()
        self.word_ids = list()
        self.word2docfreq = {'UNK': 0}
        self.word2freq = {'UNK': 0}
        self.num_docs = 0
        self.num_words = 0
        if stop_words is None:
            self.stop_words = set()
        else:
            self.stop_words = set(stop_words)

        if docs is not None:
            self.fit(docs)
        self.index = 0

    def fit(self, docs):
        """
        Counts words and sets up vocab, etc.
        """
        self.word2docfreq['UNK'] = len(docs)
        assert docs, "argument to Dictionary.fit must not be empty"
        assert isinstance(docs[0], ITERS) and docs[0] and isinstance(docs[0][0], str),\
            "argument to Dictionary.fit must be a list of nonempty lists of tokens"
        for doc in docs:
            self.num_docs += 1
            doc_words = set()
            for word in doc:
                if word not in self.stop_words:
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
        for i, doc in enumerate(docs):
            for word in doc:
                if word in self.word2id:
                    self.doc_ids.append(i)
                    self.word_ids.append(self.word2id[word])
        if self.size is not None:
            self.prune()
        return self

    def prune(self, size=None):
        """
        Shrinks vocabulary to the "size" most common words.
        """
        if size is None:
            size = self.size
        else:
            self.size = size
        if size < self.num_words:
            delete_ids = set(range(self.num_words)[size:])
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

    def transform(self, doc):
        """
        Returns sparse BOW representation as a lil_matrix.
        Doc can be a single string (1 document), a list of tokens (1 document),
        or a list of list of tokens (many documents)
        """
        if isinstance(doc, str):
            doc = tokenize(doc)
        elif doc and isinstance(doc[0], ITERS):
            return sparse.vstack([self.transform(d) for d in doc])
        if doc and isinstance(doc[0], str):
            doc = [self.word2id[word] if word in self.word2id else 0 for word in doc]
        doc = sorted(doc)  # sorting makes constructing lil_matrix more efficient
        vec = sparse.lil_matrix((1, self.num_words))
        for word in doc:
            vec[0, word] += 1
        return vec

    def transform_corpus(self):
        """
        Returns sparse matrix of entire corpus that was Dictionary.fit on
        """
        docs = [[] for _ in range(self.num_docs)]
        for doc_id, word_id in zip(self.doc_ids, self.word_ids):
            docs[doc_id].append(word_id)
        return self.transform(docs)


class Text2Vec(TransformerMixin):
    """
    Object for word2vec-based sentence/document representation. Basically does a weighted averaging.
    Future plans involve bringing more weighting schemes into the picture...

    word2vec should be a gensim word2vec model, or a string filename for a pickle file for one.
    If word2vec is not supplied, a new model will be trained. Unless you are fitting a huge corpus,
    this is probably not what you want to do... (and you probably want to tweak parameters anyway)

    pca_d: will run PCA on your corpus to shrink dimension of this transformer. pca_d must be less than
    the original (word2vec) dimension and the size of your corpus.

    remove_first: as per https://openreview.net/pdf?id=SyK00v5xx , remove the first PCA direction from the vectors.
    This is supposedly good for "similarity analysis" but seems to be bad for classification?

    a: a weighting parameter. See paper for details.
    """
    def __init__(self, word2vec=None, pca_d=None, remove_first=False, a=1):
        if word2vec is None:
            print("No word2vec model supplied.  A new one will be trained when calling fit,"
                  " which is probably not what you want to do. If this is just a test, ignore this message.\n")
        self.pca_d = pca_d
        self.a = a
        self.word2vec = word2vec
        if isinstance(self.word2vec, str):
            self.word2vec = load_pickle(self.word2vec)
        self.remove_first = remove_first
        self.pca = None
        self.u = None
        self.dictionary = None
        self.d = None

    def fit(self, docs):
        assert docs, "argument to Text2Vec.fit must not be empty"
        assert isinstance(docs[0], ITERS) and docs[0] and isinstance(docs[0][0], str),\
            "argument to Text2Vec.fit must be a list of nonempty lists of tokens"

        if self.word2vec is None:
            self.word2vec = Word2Vec(docs, size=32, iter=1, min_count=1)

        self.dictionary = Dictionary(docs)
        self.d = self.word2vec.layer1_size

        word_mat = self.transform(docs)

        if self.pca_d is not None and self.pca_d < min(self.d, self.dictionary.num_docs):
            self.pca = PCA(self.pca_d).fit(word_mat)

        if self.remove_first:
            self.u = PCA(1).fit_transform(word_mat.T)

        return self

    def transform(self, doc):
        if isinstance(doc, ITERS) and doc and isinstance(doc[0], ITERS):
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


def run_tests():
    TEST_CORPUS = ['this is a test',
                   'This should show the bag of words and text2vec representations of these sentences',
                   'It is just a bunch of random sentences',
                   'I like candy']
    TEST_CORPUS = tokenize(TEST_CORPUS)
    print("tokenizing works...")
    d = Dictionary()
    d.fit(TEST_CORPUS)
    print("dictionary fit works...")
    assert np.allclose(d.transform(TEST_CORPUS).toarray(), d.transform_corpus().toarray())
    print("dictionary transform works...")
    d.prune(5)
    print("pruning dictionary works...")
    X = d.transform(TEST_CORPUS)
    assert np.allclose(X.toarray(), d.transform_corpus().toarray())
    assert X.shape == (4, 5)
    print("dictionary works...")

    t = Text2Vec()
    t.fit(TEST_CORPUS)
    print("text2vec fit works...")
    X = t.transform(TEST_CORPUS)
    assert X.shape[0] == 4
    print("text2vec transform works...")
    X = Text2Vec(pca_d=3).fit_transform(TEST_CORPUS)
    assert X.shape == (4, 3)
    print("text2vec with pca works...")
    X = Text2Vec(remove_first=True).fit_transform(TEST_CORPUS)
    assert X.shape[0] == 4
    print("text2vec with removing first pca direction works...")
    X = Text2Vec(pca_d=3, remove_first=True).fit_transform(TEST_CORPUS)
    assert X.shape == (4, 3)
    print("text2vec works...\n")

    print("all tests passed! hooray!")

if __name__ == '__main__':
    run_tests()
