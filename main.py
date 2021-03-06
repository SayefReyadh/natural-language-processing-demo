import nltk
import random
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize, sent_tokenize
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode


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


def save_as_pickle(var_name, string):
    save_pickle = open(string, "wb")
    pickle.dump(var_name, save_pickle)
    save_pickle.close()


def load_from_pickle(string):  # "nb.pickle"
    classifier_f = open(string, "rb")
    var_name = pickle.load(classifier_f)
    classifier_f.close()
    return var_name


# documents = [(list(movie_reviews.words(fileid)) , category)
#             for category in movie_reviews.categories()
#             for fileid in movie_reviews.fileids(category)]

# ------ Old Data Set ------
# documents = []
#
# for category in movie_reviews.categories():
#     for fileid in movie_reviews.fileids(category):
#         documents.append((list(movie_reviews.words(fileid)) , category))
#
# random.shuffle(documents)
#
# # print(documents[1])
#
# all_words = []
#
# for w in movie_reviews.words():
#     all_words.append(w.lower())

# New Data


short_pos = open("short_reviews/positive.txt", "r").read()
short_neg = open("short_reviews/negative.txt", "r").read()

documents = []
all_words = []

# j is adject , r is adverb adn v is verb
# allowed_word_types = ["J" , "R" , "V"]
allowed_word_types = ["J"]

for p in short_pos.split('\n'):
    documents.append((p, "pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for p in short_neg.split('\n'):
    documents.append((p, "neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_as_pickle(documents, "documents.pickle")

# short_pos_words = word_tokenize(short_pos)
# short_neg_words = word_tokenize(short_neg)
#
# for w in short_pos_words:
#     all_words.append(w.lower())
#
# for w in short_neg_words:
#     all_words.append(w.lower())


all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:5000]

save_as_pickle(word_features, "word_features5k.pickle")


def find_featires(document):
    # words = set(document)
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


# print((find_featires(movie_reviews.words("neg/cv000_29416.txt"))))

featuresets = [(find_featires(rev), category) for (rev, category) in documents]

save_as_pickle(featuresets , "featuresets.pickle")

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

# posterior = prior occurences x liklihood / evdience

classifier = nltk.NaiveBayesClassifier.train(training_set)
# classifier = load_classifier_from_pickle("nb.pickle")
print("Orginal Naive Bayes Classifier Algorithm Accuracy Percent : ",
      (nltk.classify.accuracy(classifier, testing_set)) * 100)
# classifier.show_most_informative_features(15)
save_as_pickle(classifier, "orginal_nb.pickle")

# MultinomialNB
MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training_set)
print("MultinomialNB Classifier Algorithm Accuracy Percent : ",
      (nltk.classify.accuracy(MultinomialNB_classifier, testing_set)) * 100)

save_as_pickle(MultinomialNB_classifier, "MultinomialNB_classifier.pickle")

# GaussianNB
# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB Algorithm Accuracy Percent : ", (nltk.classify.accuracy(GaussianNB_classifier,testing_set))*100)

# BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB Classifier Algorithm Accuracy Percent : ",
      (nltk.classify.accuracy(BernoulliNB_classifier, testing_set)) * 100)

save_as_pickle(BernoulliNB_classifier, "BernoulliNB_classifier.pickle")

# LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression Classifier Algorithm Accuracy Percent : ",
      (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100)

save_as_pickle(LogisticRegression_classifier, "LogisticRegression_classifier.pickle")

# SGDClassifier
SGDClassifier_classifier = SklearnClassifier(BernoulliNB())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier Classifier Algorithm Accuracy Percent : ",
      (nltk.classify.accuracy(SGDClassifier_classifier, testing_set)) * 100)

save_as_pickle(SGDClassifier_classifier, "SGDClassifier_classifier.pickle")

# # SVC
# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC Classifier Algorithm Accuracy Percent : ", (nltk.classify.accuracy(SVC_classifier,testing_set))*100)

# LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC Classifier Algorithm Accuracy Percent : ",
      (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100)

save_as_pickle(LinearSVC_classifier, "LinearSVC_classifier.pickle")

# NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC Classifier Algorithm Accuracy Percent : ", (nltk.classify.accuracy(NuSVC_classifier, testing_set)) * 100)

save_as_pickle(NuSVC_classifier, "NuSVC_classifier.pickle")

# New Classifier With voting capabilities
voted_classifier = VoteClassifier(classifier,
                                  MultinomialNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)

print("Voted Classifier Accuracy Percent : ", (nltk.classify.accuracy(voted_classifier, testing_set)) * 100)

# for i in range(1, 5):
#     print("Classification: ", voted_classifier.classify(testing_set[i][0]), "Confidence %: ",
#           voted_classifier.confidence(testing_set[i][0]) * 100)


def sentiment(text):
    feats = find_featires(text)
    return voted_classifier.classify(feats)