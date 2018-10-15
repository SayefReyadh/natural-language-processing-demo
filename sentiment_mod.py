#File: sentiment_mod.py

import nltk
import random
#from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


def load_from_pickle(string):  # "nb.pickle"
    classifier_f = open(string, "rb")
    var_name = pickle.load(classifier_f)
    classifier_f.close()
    return var_name


# documents_f = open("pickled_algos/documents.pickle", "rb")
# documents = pickle.load(documents_f)
# documents_f.close()

documents = load_from_pickle("documents.pickle")

# word_features5k_f = open("pickled_algos/word_features5k.pickle", "rb")
# word_features = pickle.load(word_features5k_f)
# word_features5k_f.close()

word_features = load_from_pickle("word_features5k.pickle")

def find_features(document):

    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


featuresets = load_from_pickle("featuresets.pickle")

random.shuffle(featuresets)
print(len(featuresets))

testing_set = featuresets[10000:]
training_set = featuresets[:10000]


classifier = load_from_pickle("nb.pickle")

MNB_classifier = load_from_pickle("MultinomialNB_classifier.pickle")

BernoulliNB_classifier = load_from_pickle("BernoulliNB_classifier.pickle")

LogisticRegression_classifier = load_from_pickle("LogisticRegression_classifier.pickle")

LinearSVC_classifier = load_from_pickle("LinearSVC_classifier.pickle")

SGDC_classifier = load_from_pickle("SGDClassifier_classifier.pickle")

voted_classifier = VoteClassifier(
                                  classifier,
                                  LinearSVC_classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier)




def sentiment(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats) * 100