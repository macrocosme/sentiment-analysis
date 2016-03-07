__author__ = 'danyvohl'

import cPickle as pickle
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

CATEGORIES = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
PARAMETERS = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
             }

def get_train_data():
    try:
        twenty_train = pickle.load("twenty_train.p")
    except:
        twenty_train = fetch_20newsgroups(subset='train', categories=CATEGORIES, shuffle=True, random_state=42)
        pickle.dump(twenty_train, open("twenty_train.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return twenty_train

def get_test_data():
    try:
        twenty_test = pickle.load("twenty_test.p")
    except:
        twenty_test = fetch_20newsgroups(subset='test',
             categories=CATEGORIES, shuffle=True, random_state=42)
        pickle.dump(twenty_test, open("twenty_test.p", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    return twenty_test

def bag_of_words():
    twenty_train = pickle.load("twenty_train.p")
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(twenty_train.data)
    X_train_counts.shape
    count_vect.vocabulary_.get(u'algorithm')
    occurrences_to_frequencies(X_train_counts, twenty_train, count_vect)

def occurrences_to_frequencies(X_train_counts, twenty_train, count_vect):
    tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
    X_train_tf = tf_transformer.transform(X_train_counts)
    X_train_tf.shape

    #V2. All in one.
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_train_tfidf.shape
    train_classifier(X_train_tfidf, twenty_train, count_vect, tfidf_transformer)


def train_classifier(X_train_tfidf, twenty_train, count_vect, tfidf_transformer):
    clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

    docs_new = ['God is love', 'OpenGL on the GPU is fast']
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, twenty_train.target_names[category]))

def pipeline_multinomialNB():
    # Create pipeline vectorizer => transformer => classifier
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
    ])

    # Get data
    twenty_train = get_train_data()
    twenty_test = get_test_data()

    # train the model with a single command (fit data)
    text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

    # Predict data
    predicted = text_clf.predict(twenty_test.data)

    return text_clf, predicted, twenty_test

def pipeline_svm():
    # Create pipeline
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, n_iter=5, random_state=42)),
    ])

    # Get data
    twenty_train = get_train_data()
    twenty_test = get_test_data()

    # Fit data
    _ = text_clf.fit(twenty_train.data, twenty_train.target)

    # Predict data
    predicted = text_clf.predict(twenty_test.data)

    return text_clf, predicted, twenty_test

def eval_accuracy(predicted, twenty_test):
    np.mean(predicted == twenty_test.target)
    print(metrics.classification_report(twenty_test.target, predicted,
                                        target_names=twenty_test.target_names))

    print metrics.confusion_matrix(twenty_test.target, predicted)

def grid_search(text_clf):
    # Get data
    twenty_train = get_train_data()
    twenty_test = get_test_data()
    
    gs_clf = GridSearchCV(text_clf, PARAMETERS, n_jobs=-1)
    gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])

    # Predict some string...
    twenty_train.target_names[gs_clf.predict(['God is love'])]

    # But mostly, show optimal parameters
    best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))

    print score

