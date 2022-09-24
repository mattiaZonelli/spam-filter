########################################################################################################################
################## NOTE: RUN ONLY ONE CLASSIFIER AT TIME, OTHERWISE ACCURACY OF OTHER CAN BE WRONG ######################
########################################################################################################################
'''
    code by: Eleonora Garbin 869831 and Zonelli Mattia 870038
'''

from someTest import nb_tests
from kNN import run_knn
from naiveBayes import run_NaiveBayes
from SVM import run_SVM
import numpy as np

FILEPATH = "data/spambase.data"


# compute tfidf for all docs
def tfidf_func(tf, n_doc):
    denom = np.zeros(len(tf[0]))

    for i in range(len(denom)):
        denom[i] = np.count_nonzero(np.transpose(tf)[i])

    idf = np.log10(n_doc / denom)
    return tf / 100 * idf


# main
if __name__ == '__main__':
    # dataset
    dataset = np.loadtxt(FILEPATH, delimiter=",")
    np.random.shuffle(dataset)

    # 4601 emails/docs, 54 words
    # tabella delle frequenze[doc][word],
    frequences = dataset[:, :54]

    # feature 57th 1 = spam, 0 = ham, len is 4601 = number of different emails/docs
    classes = dataset[:, 57]
    tfidf = tfidf_func(frequences, len(classes))
    '''
        SVM with length, over tfidf representation.
    '''
    run_SVM(tfidf, classes)

    '''
        now we  transform the kernels such that they use angular information (cosine similarity) and not the
        euclidean distance
    '''

    norms = np.sqrt(((tfidf + 1e-128) ** 2).sum(axis=1, keepdims=True))
    normalized_lenghts = np.where(norms > 0.00, tfidf / norms, 0.)

    # print("\n******************SVM with angular information\n")
    # run_SVM(normalized_lenghts, classes)

    ####################################################################################################################
    '''
        Now with Naive Bayes Classifier
    '''
    X = tfidf
    Y = classes
    # run_NaiveBayes(X, Y)

    # to check correctness of assumptions
    # nb_tests(FILEPATH)

    ####################################################################################################################
    '''
        Now with K-nn classifier, k = 5
    '''
    X = tfidf
    Y = classes
    # run_knn(X, Y)

    ####################################################################################################################
