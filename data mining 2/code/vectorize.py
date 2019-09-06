from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.stem.snowball import SnowballStemmer
from nltk import pos_tag

import scipy.sparse as sp
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

#example text text = 'What can I say about this place. The staff of these restaurants is nice and the eggplant is not bad'

class Splitter(object):
    """
    split the document into sentences and tokenize each sentence
    """
    def __init__(self):
        self.splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self,text):
        """
        out : ['What', 'can', 'I', 'say', 'about', 'this', 'place', '.']
        """
        # split into single sentence
        sentences = self.splitter.tokenize(text)
        # tokenization in each sentences
        tokens = [self.tokenizer.tokenize(sent) for sent in sentences]
        return tokens


class LemmatizationWithPOSTagger(object):
    def __init__(self):
        pass
    def get_wordnet_pos(self,treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def pos_tag(self,tokens):
        # find the pos tagginf for each tokens [('What', 'WP'), ('can', 'MD'), ('I', 'PRP') ....
        pos_tokens = [nltk.pos_tag(token) for token in tokens]

        # lemmatization using pos tagg
        # convert into feature set of [('What', 'What', ['WP']), ('can', 'can', ['MD']), ... ie [original WORD, Lemmatized word, POS tag]
        pos_tokens = [ lemmatizer.lemmatize(word,self.get_wordnet_pos(pos_tag)) for pos in pos_tokens for (word,pos_tag) in pos ]
        return pos_tokens

lemmatizer = WordNetLemmatizer()
splitter = Splitter()

class PosLemmatisingCountVectorizer(CountVectorizer):

    lemmatization_using_pos_tagger = LemmatizationWithPOSTagger()

    def build_analyzer(self):

        def get_string(path):
            with open(path) as file:  # Use file to refer to the file object
                the_string = file.read()
            return the_string
        return lambda document: PosLemmatisingCountVectorizer.lemmatization_using_pos_tagger.pos_tag(splitter.split(get_string(document)))





class StemmingCountVectorizer(CountVectorizer):

    stemmer = SnowballStemmer('english')

    def build_analyzer(self):
        analyzer = super(StemmingCountVectorizer, self).build_analyzer()
        return lambda document: ([StemmingCountVectorizer.stemmer.stem(word) for word in analyzer(document)])


class LemmatisingCountVectorizer(CountVectorizer):

    lemmatiser = WordNetLemmatizer()

    def build_analyzer(self):
        analyzer = super(LemmatisingCountVectorizer, self).build_analyzer()
        return lambda document: ([LemmatisingCountVectorizer.lemmatiser.lemmatize(word, pos_tag(word)) for word in analyzer(document)])



def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)

class TfidfTransformerStrict(TfidfTransformer):


        def fit(self, X, y=None):
            """Learn the idf vector (global term weights)

            Parameters
            ----------
            X : sparse matrix, [n_samples, n_features]
                a matrix of term/token counts
            """
            if not sp.issparse(X):
                X = sp.csc_matrix(X)
            if self.use_idf:
                n_samples, n_features = X.shape
                df = _document_frequency(X)

                # perform idf smoothing if required
                df += int(self.smooth_idf)
                n_samples += int(self.smooth_idf)

                # log+1 instead of log makes sure terms with zero idf don't get
                # suppressed entirely.
                idf = np.log(float(n_samples) / df)
                self._idf_diag = sp.spdiags(idf, diags=0, m=n_features,
                                            n=n_features, format='csr')

            return self