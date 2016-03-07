# Experiment to classify sentiment in tweeter text based on python's scikit-learn module

** Author: Dany Vohl

Currently includes two notebooks:

- Sentiment Analysis tutorial.ipynb is based on scikit-learn tutorial for working with text data
- Sentiment Analysis of Twitter data.ipynb is the main experiment

Similar python code is contained in sentiment.py and tweets_sentiment.py

Requires:
Python 2.7
- scikit-learn 0.18.dev0
- numpy
- cPickle
(jupyter notebook is recommended)

To install scikit-learn v. 0.18.dev0, currently do the following:

``pip install git+git://github.com/scikit-learn/scikit-learn.git``

Use the GetOldTweets scripts to retrieve more tweets that with the common Twitter API (which only allows to search back a week).

    # Example 1 - Get tweets by username [barackobama]
    python Export.py --username 'barackobama' --maxtweets 1

    # Example 2 - Get tweets by query search [europe refugees]
    python Export.py --querysearch 'europe refugees' --maxtweets 1

    # Example 3 - Get tweets by username and bound dates [barackobama, '2015-09-10', '2015-09-12']
    python Export.py --username 'barackobama' --since 2015-09-10 --until 2015-09-12 --maxtweets 1

## Questions?

[Drop me a line](http://macrocosme.github.io/#contact) here!