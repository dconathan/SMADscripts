import pandas as pd
import os
# TODO use other tokenizers?
from nltk.tokenize import TweetTokenizer
# TODO use other stemmers?
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.ldamodel import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
import time
import pickle

tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True)
stemmer = SnowballStemmer('english')

stopwords = stopwords.words('english')
stopwords + ['---', '...']
with open('stopwords') as f:
    stopwords += f.read().split(', ')

stopwords = set(stopwords)


def savePickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def loadPickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    ''' Returns a random greyscale rgb triplet, for wordcloud '''
    c = random.randint(0, 100)
    return "rgb({}, {}, {})".format(c, c, c)


def saveWordCloud(text, filename=None, show=False):
    '''
    Generates word cloud for text, which can be string or list of tokens.
    If filename is supplied, saves wordcloud image as filename.
    If show=True, displays the image using matplotlib.
    '''
    if type(text) != str:
        text = ' '.join(text)
    wordcloud = WordCloud(background_color='white')
    wordcloud.generate(text)
    wordcloud = wordcloud.recolor(color_func=grey_color_func)
    if filename:
        wordcloud.to_file(filename)
    if show:
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.show()


def tokenize(article, toke=True, filt=True, stem=True):
    '''
    Takes string (article/paragraph/text) as input, outputs fully cleaned list of tokens to go into model
    '''

    if toke:
        tokens = tokenizer.tokenize(article)
    else:
        tokens = article
    if filt:
        tokens = filter(filter_token, tokens)
    if stem:
        tokens = map(stemmer.stem, tokens)

    return list(tokens)


def filter_token(token, stopwords):
    '''
    Returns False if token should be removed, True otherwise
    '''
    return len(token) > 2 and \
        not token.isdigit() and \
        token not in stopwords and \
        token[0] != '@' and \
        token[0:4] != 'http'


class LDAModel:
    def __init__(self, dataframe=None, base='../data', output='output', lda=None, corpus=None, dictionary=None, load_model=False):

        # Set working directories
        self.base = base
        self.raw = os.path.join(self.base, 'raw')
        self.clean = os.path.join(self.base, 'clean')
        self.models = os.path.join(self.base, 'models')
        self.clean_csv = os.path.join(self.clean, 'clean_data.csv')
        self.output = os.path.join(self.base, output)

        # Make sure the appropriate output directories exist:
        for d in [self.clean, self.models, self.output]:
            if not os.path.exists(d):
                os.mkdir(d)

        # Initialize dataframe if not supplied
        self.dataframe = dataframe
        if self.dataframe is None:
            self.dataframe = pd.DataFrame()

        if load_model:
            self.lda = loadPickle(os.path.join(self.models), 'lda.pickle')
            self.corpus = loadPickle(os.path.join(self.models), 'corpus.pickle')
            self.dictionary = loadPickle(os.path.join(self.models), 'dictionary.pickle')
        else:
            self.lda = lda
            self.corpus = corpus
            self.dictionary = dictionary

    def cleanArticles(self, csv_file='clean_data.csv', source='sample', by_paragraph=False):
        '''
        Reads articles from raw source and turns it into a cleaned dataframe, appends to self.dataframe
        If by_paragraph=True, each paragraph gets its own row
        '''

        fullraw = os.path.join(self.raw, source)

        with open(fullraw) as f:
            raw_text = ''.join(f.readlines())

        pre_token = 'LENGTH: '
        if 'NYT' in source:
            post_token = 'URL: '
        else:
            post_token = 'LOAD-DATE: '

        raw_text = raw_text.split(pre_token)

        cooked_data = []

        for i, raw_article in enumerate(raw_text[1:]):
            raw_article = raw_article.split('\n')
            article = ''
            paragraph = ''
            paragraph_n = 0
            while True:
                article += paragraph + '\n\t'
                paragraph = raw_article[paragraph_n + 2]
                paragraph_n += 1
                paragraph = paragraph.replace('"', "''")
                if post_token in paragraph:
                    break
                if by_paragraph:
                    paragraph = '\t' + paragraph + '\n'
                    cooked_data.append((source, 'a{}p{}'.format(i, paragraph_n), paragraph))
            if not by_paragraph:
                article = article[2:]
                while article[-2:] == '\n\t':
                    article = article[:-2]
                article = '\t' + article
                cooked_data.append((source, 'a{}'.format(i), article))

        this_dataframe = pd.DataFrame.from_dict(dict(enumerate(cooked_data)), orient='index')
        this_dataframe.columns = ('source', 'source_index', 'text')
        self.dataframe.append(this_dataframe)

    def saveCleanCSV(self):
        ''' Saves self.dataframe as filename self.clean_csv '''
        self.dataframe.to_csv(self.clean_csv, index=False)

    def loadClean(self):
        ''' Walks through and loads all .csv files in the /data/clean/ directory '''

        self.dataframe = pd.DataFrame()
        path, _, files = next(os.walk(self.clean))
        num_files = len(files)
        files = [os.path.join(path, f) for f in files]

        for i, f in enumerate(files):
            this_df = pd.read_csv(f)
            self.dataframe = self.dataframe.append(this_df)
            print("Loaded {}/{} files".format(i + 1, num_files))

    def process(self):
        '''
        Tokenizes, filters and stems all text in the "text" column of self.dataframe, creates corpus and dictionary.
        '''

        t0 = time.time()
        tokenized = self.dataframe['text'].apply(tokenize, filt=False, stem=False)
        print("Tokenizing took {}s".format(time.time() - t0))

        t0 = time.time()
        tokenized = tokenized.apply(tokenize, toke=False, stem=False)
        print("Filtering took {}s".format(time.time() - t0))

        t0 = time.time()
        tokenized = tokenized.apply(tokenize, toke=False, filt=False)
        print("Stemming took {}s".format(time.time() - t0))

        tokenized.name = 'tokenized_text'
        self.dataframe = self.dataframe.join(tokenized)

        t0 = time.time()
        dictionary = Dictionary(self.dataframe['tokenized_text'])
        print("Creating dictionary took {}s".format(time.time() - t0))

        # Filter out extremes
        dictionary.filter_extremes(no_below=20, no_above=.5, keep_n=None)

        self.dictionary = dictionary
        savePickle(dictionary, os.path.join(self.models, 'dictionary.pickle'))

        t0 = time.time()
        corpus = self.dataframe['tokenized_text'].apply(self.dictionary.doc2bow)
        print("Creating BOW embeddings took {}s".format(time.time() - t0))
        corpus.name = 'bow_vector'

        self.corpus = corpus
        savePickle(corpus, os.path.join(self.models, 'corpus.pickle'))

    def trainLDA(self, num_topics=20, use_multicore=False):
        """ Trains LDA model using self.corpus and self.dictionary """

        t0 = time.time()
        if use_multicore:
            lda = LdaMulticore(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary, workers=3, passes=8)
        else:
            lda = LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=8, alpha='auto')
        print("Training LDA model took {}s".format(time.time() - t0))

        self.lda = lda
        savePickle(lda, '../data/models/lda.pickle')

    def predictTopic(self, article):
        '''
        Returns the topic with highest probability according to self.lda,
        "article" can be raw string, list of tokens, or bow vector
        '''

        if type(article) == str:
            if article == '':
                # Empty article, don't assign a topic
                return None
            # raw article, need to tokenize and vectorize
            tokens = tokenize(article)
            processed = self.dictionary.doc2bow(tokens)

        elif type(article) == list:
            if len(article) == 0:
                # Empty article, don't assign a topic
                return None
            elif type(article[0]) == str:
                # article is list of tokens, turn into BOW vector
                processed = self.dictionary.doc2bow(article)

            elif type(article[0]) == tuple:
                # article is BOW vector, no need to preprocess
                processed = article

        topics = self.lda.get_document_topics(processed)
        # Return topic with max probability
        return max(topics, key=lambda x: x[1])[0]

    def sortByTopic(self):

        topic_term_file = os.path.join(self.output, 'topic_terms.txt')

        # Write summary of topic terms
        with open(topic_term_file, 'w') as f:
            for topic, terms in self.lda.print_topics(-1):
                f.write('{}: {}\n\n'.format(topic, terms))

        topic2word_map = {i: [] for i in range(self.lda.num_topics)}
        article_dicts = {i: {'topic': [], 'source': [], 'source_index': [], 'text': []} for i in range(self.lda.num_topics)}

        # TODO this could benefit from being parallelized

        t0 = time.time()
        for i, c in enumerate(self.corpus):
            t = self.predictTopic(c)
            if t is None:
                print("article {} is empty".format(i))
            else:
                topic2word_map[t] += tokenize(self.dataframe.at[i, 'text'])
                article_dict = {'topic': [t],
                                'source': [self.dataframe.at[i, 'source']],
                                'source_index': [self.dataframe.at[i, 'source_index']],
                                'text': [self.dataframe.at[i, 'text']]}
                for k in article_dicts[t]:
                    article_dicts[t][k] += article_dict[k]
        print("Sorting articles by topic took {}s".format(time.time() - t0))

        for t, article_dict in article_dicts.items():
            filename = 'topic{}_articles.csv'.format(t)
            full_filename = os.path.join(self.output, filename)
            dataframe = pd.DataFrame.from_dict(article_dict, orient='columns')
            dataframe.to_csv(full_filename, index=False, quoting=1)

        t0 = time.time()
        for i, text in topic2word_map.items():
            if len(text) > 0:
                full_filename = os.path.join(self.output, 'topic{}_wordcloud.png'.format(i))
                saveWordCloud(text, full_filename)
        print("Generating wordclouds took {}s".format(time.time() - t0))

if __name__ == '__main__':
    lda_model = LDAModel()
    lda_model.cleanArticles()
    lda_model.process()
    lda_model.trainLDA()
    lda_model.sortByTopic()
