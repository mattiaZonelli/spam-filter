import numpy as np
from scipy.stats import shapiro, normaltest
import random
import seaborn as sb
import matplotlib.pyplot as plt
import scipy

def modelling(X, y):
    ham = X[y == 0, :54]
    spam = X[y == 1, :54]

    #n_doc = float(len(ham) + len(spam))

    # mean of features of class spam
    mean_1 = np.mean(spam, axis=0)
    # mean of features of class ham
    mean_0 = np.mean(ham, axis=0)
    # variances of feature i given that the data is in class spam
    var_1 = np.var(spam, axis=0) + 1e-128
    # variances of feature i given that the data is in class ham
    var_0 = np.var(ham, axis=0) + 1e-128

    # coso nella produttoria
    p_x_y1_i = (2 * np.pi * var_1) ** (-1. / 2) * np.exp(-1. / (2 * var_1) * ((X - mean_1) ** 2))
    # coso nella produttoria
    p_x_y0_i = (2 * np.pi * var_0) ** (-1. / 2) * np.exp(-1. / (2 * var_0) * ((X - mean_0) ** 2))
    # p(x | y = 1)
    p_x_y1 = np.prod(p_x_y1_i, axis=1)
    # p(x | y = 0)
    p_x_y0 = np.prod(p_x_y0_i, axis=1)

    evidence = p_x_y0 + p_x_y1 + 1e-128
    p_x_y1 = p_x_y1 / evidence
    p_x_y0 = p_x_y0 / evidence

    return p_x_y0, p_x_y1


def gaussian_test(tbl, classes):
    p_x_ham, p_x_spam = modelling(tbl, classes)

    # Saphiro normality test
    print ('\n--------------------------Saphiro normality test--------------------------------------------------------')
    # for ham - Saphiro normality test
    stat, p = shapiro(p_x_ham)
    print('Statistics=%.3f, p=%.8f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Ham prob distrib looks Gaussian (fail to reject H0)')
    else:
        print('Ham prob distrib does not look Gaussian (reject H0)')
    # for spam - Saphiro normality test
    stat, p = shapiro(p_x_spam)
    print('Statistics=%.3f, p=%.8f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('SPam prob distrib looks Gaussian (fail to reject H0)')
    else:
        print('SPam prob distrib does not look Gaussian (reject H0)')

    # normality test - D Agostino s K**2 Test
    print ('\n--------------------------D-Agostino-s K**2 Test--------------------------------------------------------')
    # for ham
    stat, p = normaltest(p_x_ham)
    print('Statistics=%.3f, p=%.8f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Ham prob distrib looks Gaussian (fail to reject H0)')
    else:
        print('Ham prob distrib does not look Gaussian (reject H0)')
    # for spam
    stat, p = normaltest(p_x_spam)
    print('Statistics=%.3f, p=%.8f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('SPam prob distrib looks Gaussian (fail to reject H0)')
    else:
        print('SPam prob distrib does not look Gaussian (reject H0)')

    ###################################################################################################################


def nb_tests(path):
    FILEPATH = path
    data = np.loadtxt(FILEPATH, delimiter=",")
    tbl = data[:, :54]
    classes = data[:, 57]
    ham = tbl[classes == 0,]
    spam = tbl[classes == 1,]

    # to create
    hamS = np.sum(ham, axis=1)
    spamS = np.sum(spam, axis=1)

    # ham density plot
    fileName = r'plots/ham_density_plot.png'
    fig, ax = plt.subplots(1)
    sb.set_style('whitegrid')
    sb.distplot(hamS, hist=False, kde=True, color='darkblue')
    plt.ylabel("Density")
    plt.xlabel("Frequency")
    plt.title("Ham emails Density Plot ")
    plt.show()
    fig.savefig(fileName, format='png')
    plt.close(fig)
    # spam density plot
    fileName = r'plots/spam_density_plot.png'
    fig, ax = plt.subplots(1)
    sb.set_style('whitegrid')
    sb.distplot(spamS, hist=False, kde=True, color='red')
    plt.ylabel("Density")
    plt.xlabel("Frequency")
    plt.title("Spam emails Density Plot ")
    plt.show()
    fig.savefig(fileName, format='png')
    plt.close(fig)

    '''
        we can see from these plots that distributions of spam and ham 
    '''
    '''
        TO CHECK IF WORD_FREQUeNCY OF DIFFERENT WORD IN SAME CLASS ARE INDEPENDENT OR NOT
        there is a .r file
    '''
    ####################################################################################################################
    ''' 
        now we check if p(x|y) has the same behaviour of Gaussian distributions and if some random features are 
        distributed like a Gaussian

    '''
    gaussian_test(tbl, classes)
    #
    #
    fileName = r'plots/ham_qq-plot{0:02d}.png'
    for h in range(4):
        rnd = random.randrange(54)
        fig, ax = plt.subplots(1)
        scipy.stats.probplot(ham[:, rnd], dist="norm", plot=plt)
        title = 'Q-Q plot of feature classified as Ham ' + str(rnd) + '-th'
        plt.title(title)
        plt.show()
        fig.savefig(fileName.format(h), format='png')
        plt.close(fig)

    fileName = r'plots/spam_qq-plot{0:02d}.png'
    for s in range(4):
        rnd = random.randrange(54)
        fig, ax = plt.subplots(1)
        scipy.stats.probplot(spam[:, rnd], dist="norm", plot=plt)
        title = 'Q-Q plot of feature classified as SPam ' + str(rnd) + '-th'
        plt.title(title)
        plt.show()
        fig.savefig(fileName.format(s), format='png')
        plt.close(fig)