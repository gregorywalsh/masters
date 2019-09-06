from pathlib import Path

import numpy as np

from vectorize import PosLemmatisingCountVectorizer, TfidfTransformerStrict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from gensim.models import KeyedVectors

# Get file paths for all documents
root = Path('/Users/gregwalsh/Google Drive/Study/Data Science Masters/Modules/Semester 2/Data Mining/CW2/Data/corpus')
paths = list(root.glob('**/*.txt'))
paths.sort(key=lambda x: int(getattr(x, "stem")[0:2]))

with open('doc_matrices/doc_name_list.txt', 'w+') as f:
    for path in paths:
        f.write(path.name + '\n')

# Create vectorizers for bag-of-words and tf-idf approaches
count_vectorizer = CountVectorizer(input='filename',
                                   analyzer="word",
                                   stop_words='english',
                                   min_df=2
                                   )

lemmatising_vectorizer = PosLemmatisingCountVectorizer(input='filename',
                                   analyzer="word",
                                   stop_words='english',
                                   min_df=2
                                   )

tfidf_transformer = TfidfTransformer(norm='l2', use_idf=True, smooth_idf=False, sublinear_tf=False)
tf_transformer = TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False, sublinear_tf=False) # Why L2?

# Create document_term and document_symbol matrices
bow_doc_term_matrix = count_vectorizer.fit_transform(paths).todense()
lemma_bow_doc_term_matrix = lemmatising_vectorizer.fit_transform(paths).todense()

tf_bow_doc_term_matrix = tf_transformer.fit_transform(bow_doc_term_matrix).todense()
tf_lemma_bow_doc_term_matrix = tf_transformer.fit_transform(lemma_bow_doc_term_matrix).todense()

tfidf_doc_term_matrix = tfidf_transformer.fit_transform(bow_doc_term_matrix).todense()
lemma_tfidf_doc_term_matrix = tfidf_transformer.fit_transform(lemma_bow_doc_term_matrix).todense()

terms = count_vectorizer.get_feature_names()

# Load the GoogleNews trained word2vec model
model = KeyedVectors.load_word2vec_format('~/Downloads/GoogleNews-vectors-negative300.bin', binary=True)

doc_symbol_matrix = np.zeros((24, 300))
for i in range(bow_doc_term_matrix.shape[0]): # Loop over documents
    for j in range(bow_doc_term_matrix.shape[1]): # Loop over terms
        if bow_doc_term_matrix[i, j] > 0:
            try:
                term_vector = model[terms[j]]
                doc_symbol_matrix[i] = np.add(doc_symbol_matrix[i], np.multiply(term_vector, bow_doc_term_matrix[i, j]))  # 1. Find the total document vector
            except KeyError:
                pass

tf_doc_symbol_matrix = tf_transformer.fit_transform(doc_symbol_matrix).todense()  # 2. Normalise the total document vectors (why use L2 rather than L1?)
# NOTE - everything is now on a sphere so Euclidean distance based clustering doesn't seem quite as appropriate

feature_matrices = [bow_doc_term_matrix,
                    lemma_bow_doc_term_matrix,

                    tf_bow_doc_term_matrix,
                    tf_lemma_bow_doc_term_matrix,

                    tfidf_doc_term_matrix,
                    lemma_tfidf_doc_term_matrix,

                    doc_symbol_matrix,
                    tf_doc_symbol_matrix
                    ]

matrices_names = ['bow_doc_term_matrix',
                  'lemma_bow_doc_term_matrix',

                  'tf_bow_doc_term_matrix',
                  'tf_lemma_bow_doc_term_matrix',

                  'tfidf_doc_term_matrix',
                  'lemma_tfidf_doc_term_matrix',

                  'doc_symbol_matrix',
                  'tf_doc_symbol_matrix']

for i, feature_matrix in enumerate(feature_matrices):
    np.save('doc_matrices/' + matrices_names[i], feature_matrix)
