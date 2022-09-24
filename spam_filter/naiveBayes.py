from __future__ import division
import time
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, mean_squared_error
from math import sqrt


class NaiveBayesClassifier(BaseEstimator):

    def fit(self, X, y):
        self.ham = X[y == 0, :54]
        self.spam = X[y == 1, :54]

        self.n_doc = float(len(self.ham) + len(self.spam))
        # p(y = 1)
        self.p_y1 = len(self.spam) / self.n_doc
        # p(y=0)
        self.p_y0 = len(self.ham) / self.n_doc

        # mean of features of class spam
        self.mean_1 = np.mean(self.spam, axis=0)
        # mean of features of class ham
        self.mean_0 = np.mean(self.ham, axis=0)
        # variances of feature i given that the data is in class spam
        self.var_1 = np.var(self.spam, axis=0) + 1e-128
        # variances of feature i given that the data is in class ham
        self.var_0 = np.var(self.ham, axis=0) + 1e-128

    def score(self, X, y):
        # pre produttoria
        p_x_y1_i = (2 * np.pi * self.var_1) ** (-1. / 2) * np.exp(-1. / (2 * self.var_1) * ((X - self.mean_1) ** 2))
        # pre produttoria
        p_x_y0_i = (2 * np.pi * self.var_0) ** (-1. / 2) * np.exp(-1. / (2 * self.var_0) * ((X - self.mean_0) ** 2))
        # p(x | y = 1)
        p_x_y1 = np.prod(p_x_y1_i, axis=1)
        # p(x | y = 0)
        p_x_y0 = np.prod(p_x_y0_i, axis=1)

        evidence = p_x_y0 + p_x_y1 + 1e-128
        p_x_y1 = p_x_y1 / evidence
        p_x_y0 = p_x_y0 / evidence

        # p_x = (p_x_y1 * self.p_y1 + p_x_y0 * self.p_y0) + 1e-128
        # p( y=1 | x) = p(x|y=1)*p(y=1) /p(x)
        p_y1_x = p_x_y1 * self.p_y1  # / p_x
        # p( y=0 | x) = = p(x|y=0)*p(y=0) /p(x)
        p_y0_x = p_x_y0 * self.p_y0  # / p_x

        winner_class = np.argmax([p_y0_x, p_y1_x], axis=0)
        return np.mean(winner_class == y)

    def predict(self, X):
        # pre produttoria
        p_x_y1_i = (2 * np.pi * self.var_1) ** (-1. / 2) * np.exp(-1. / (2 * self.var_1) * ((X - self.mean_1) ** 2))
        # pre produttoria
        p_x_y0_i = (2 * np.pi * self.var_0) ** (-1. / 2) * np.exp(-1. / (2 * self.var_0) * ((X - self.mean_0) ** 2))
        # p(x | y = 1)
        p_x_y1 = np.prod(p_x_y1_i, axis=1)
        # p(x | y = 0)
        p_x_y0 = np.prod(p_x_y0_i, axis=1)

        evidence = p_x_y0 + p_x_y1 + 1e-128
        p_x_y1 = p_x_y1 / evidence
        p_x_y0 = p_x_y0 / evidence

        # p_x = (p_x_y1 * self.p_y1 + p_x_y0 * self.p_y0) + 1e-128
        # p( y=1 | x) = p(x|y=1)*p(y=1) /p(x)
        p_y1_x = p_x_y1 * self.p_y1  # / p_x
        # p( y=0 | x) = = p(x|y=0)*p(y=0) /p(x)
        p_y0_x = p_x_y0 * self.p_y0  # / p_x

        winner_class = np.argmax([p_y0_x, p_y1_x], axis=0)
        return winner_class


def run_NaiveBayes(X, Y):
    clf_NB = NaiveBayesClassifier()
    start = time.time()
    scores = cross_val_score(clf_NB, X, Y, cv=10)
    print("Time", time.time() - start, "sec")
    print("\n-----------------------------------------------------------------------")
    print("Naive Bayes Classifier")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("MIN ACCURACY:", scores.min())
    print("MAX ACCURACY:", scores.max())
    cv_results = cross_validate(NaiveBayesClassifier(), X, Y, cv=10)
    print("CROSS VALIDATE FIT_TIME: ", np.sum(cv_results['fit_time']))

    # trained model
    clf_NB = NaiveBayesClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    clf_NB.fit(X_train, y_train)
    # print ("\n-----------------------------------------------------------------------")
    print("\nNaive Bayes Classifier")
    print("Actual accuracy of trained Naive Bayes: ", clf_NB.score(X_test, y_test))
    print("EXTRA:\n- p(ham) = %0.4f;\n- p(spam) = %0.4f" % (clf_NB.p_y0, clf_NB.p_y1))
    print("-----------------------------------------------------------------------\n")

    train_preds = clf_NB.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("RMSE of training set:", rmse)

    # RMSE for test set
    test_preds = clf_NB.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print("RMSE of test set:", rmse)

    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    print("CONFUSION MATRIX:\n", cm)

