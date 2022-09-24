import time
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, mean_squared_error
from math import sqrt


def print_scores(scores):
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("MIN ACCURACY:", scores.min())
    print("MAX ACCURACY:", scores.max())


# SVM with linear kernel and pre processed tfidf and 10 way cross validation
def SVM_linear(x, y):
    clf = SVC(kernel='linear', C=1, gamma='scale')
    start = time.time()
    scores = cross_val_score(clf, x, y, cv=10)
    # print("Time", time.time() - start, "sec")
    print("SVM with linear kernel")
    print_scores(scores)
    cv_results = cross_validate(SVC(kernel='linear', C=1, gamma='scale'), x, y, cv=10)
    print("CROSS VALIDATE FIT_TIME: ", np.sum(cv_results['fit_time']))
    print("----------------------------------------------\n")
    return scores


# SVM with polynomial of degree 2 and 10 way cross validation
def SVM_poly2(x, y):
    clf = SVC(kernel='poly', degree=2, C=1, gamma='scale')
    start = time.time()
    scores = cross_val_score(clf, x, y, cv=10)
    # print("Time", time.time() - start, "sec")
    print("SVM with polinomial of degree 2 kernel")
    print_scores(scores)
    cv_results = cross_validate(SVC(kernel='poly',degree=2, C=1, gamma='scale'), x, y, cv=10)
    print("CROSS VALIDATE FIT_TIME: ", np.sum(cv_results['fit_time']))
    print("----------------------------------------------\n")
    return scores


# SVM with Gaussian kernel and pre processed tfidf and 10 way cross validation
def SVM_rbf(x, y):
    clf = SVC(kernel='rbf', C=1, gamma='scale')
    start = time.time()
    scores = cross_val_score(clf, x, y, cv=10)
    # print("Time", time.time() - start, "sec")
    print("SVM with RBF kernel")
    print_scores(scores)
    cv_results = cross_validate(SVC(kernel='rbf', C=1, gamma='scale'), x, y, cv=10)
    print("CROSS VALIDATE FIT_TIME: ", np.sum(cv_results['fit_time']))
    print("----------------------------------------------\n")
    return scores


def run_SVM(X, y):
    # SVM w/ linear kernel
    SVM_linear(X, y)
    # SVM w/ poly of degree 2
    SVM_poly2(X, y)
    # SVM w/ rbf kernel
    SVM_rbf(X, y)
    '''
        Now we train SVM classifiers
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    full_test = X
    np.random.shuffle(full_test)

    #  SVM classifier with linear kernel
    clf_lin = SVC(kernel='linear', C=1.0, gamma='scale').fit(X_train, y_train)
    print("SVM classifier with linear kernel")
    print("Actual accuracy of trained SVM: ", clf_lin.score(X_test, y_test))
    print("Classifier find ", np.count_nonzero(clf_lin.predict(X_test)),
          "ham emails while in the feature test there was: ", np.count_nonzero(y_test))
    print("Expected # support vectors / # training samples =", clf_lin.support_vectors_.shape[0], "/", X_train.shape[0],
          "=", clf_lin.support_vectors_.shape[0] / X_train.shape[0])
    # RMSE for training set
    train_preds = clf_lin.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("RMSE of training set:", rmse)
    # RMSE for test set
    test_preds = clf_lin.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print("RMSE of test set:", rmse)
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    print("CONFUSION MATRIX:\n", cm)
    print("\n-----------------------------------------------------------------------")
    #  SVM classifier with polynomial kernel
    clf_poly = SVC(kernel='poly', degree=2, C=1.0, gamma='scale').fit(X_train, y_train)
    print("SVM classifier with polynomial kernel of degree 2")
    print("Actual accuracy of trained SVM: ", clf_poly.score(X_test, y_test))
    print("Classifier find: ", np.count_nonzero(clf_poly.predict(X_test)),
          "ham emails while in the feature test there was: ", np.count_nonzero(y_test))
    print("Expected # support vectors / # training samples =", clf_poly.support_vectors_.shape[0], "/",
          X_train.shape[0],
          "=", clf_poly.support_vectors_.shape[0] / X_train.shape[0])
    # RMSE for training test
    train_preds = clf_poly.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("RMSE of training set:", rmse)
    # RMSE for test set
    test_preds = clf_poly.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print("RMSE of test set:", rmse)
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    print("CONFUSION MATRIX:\n", cm)
    print("\n-----------------------------------------------------------------------")

    #  SVM classifier with RBF kernel
    clf_rbf = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train, y_train)
    print("SVM classifier with RBF kernel")
    print("Actual accuracy of trained SVM: ", clf_rbf.score(X_test, y_test))
    print("Classifier find: ", np.count_nonzero(clf_rbf.predict(X_test)),
          "ham emails while in the feature there was: ", np.count_nonzero(y_test))
    print("Expected # support vectors / # training samples =", clf_rbf.support_vectors_.shape[0], "/", X_train.shape[0],
          "=", clf_rbf.support_vectors_.shape[0] / X_train.shape[0])
    # RMSE for training set
    train_preds = clf_rbf.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("RMSE of training set:", rmse)
    # RMSE for test set
    test_preds = clf_rbf.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print("RMSE of test set:", rmse)
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    print("CONFUSION MATRIX:\n", cm)
    print("\n-----------------------------------------------------------------------")
