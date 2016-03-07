__author__ = 'danyvohl'

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from twitter import Twitter
from twitter import oauth_dance, OAuth

def connect_to_twitter():
    # 'QdrOrGTx0JqAzTSAZkeF8ZCxT',
    # '5TNSrWNfOq2ek55HMAjKbgmd24NSUY8fbWhmZeHNyQ6ELVEl4b',
    # '1606571954-nQrQ4iQaOku7NcABsB44y1bp1ukSkYGumZNMeE8',
    # 'dzMLIjhcnABo3fCJITbyOhoNh84eOM6Kh289bilVxSXx2'

    CONSUMER_KEY = 'QdrOrGTx0JqAzTSAZkeF8ZCxT'
    CONSUMER_SECRET = '5TNSrWNfOq2ek55HMAjKbgmd24NSUY8fbWhmZeHNyQ6ELVEl4b'

    oauth_token, oauth_token_secret = oauth_dance("Test Bot", CONSUMER_KEY, CONSUMER_SECRET)
    t = Twitter(auth=OAuth(oauth_token, oauth_token_secret, CONSUMER_KEY, CONSUMER_SECRET))

    return t

def get_tweets(subject):
    try:
        t = connect_to_twitter()
    except:
        print("can't connect to twitter. bummer. frowny face.")

    tweets = t.search.tweets(q='#' + subject, src='typd')

if __name__ == "__main__":
    # the training data folder must be passed as first argument
    movie_reviews_data_folder = sys.argv[1]
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent
    pipeline = Pipeline([
        ('vect', TfidfVectorizer(min_df=3, max_df=0.95)),
        ('clf', LinearSVC(C=1000)),
    ])

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2)],
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1)
    grid_search.fit(docs_train, y_train)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search
    print(grid_search.grid_scores_)

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted
    y_predicted = grid_search.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    plt.matshow(cm)
    plt.show()
