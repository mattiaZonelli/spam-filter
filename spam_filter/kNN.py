from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, cross_validate
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
import time

def KNN_5(x, y):
    clf = KNN(n_neighbors=5)
    start = time.time()
    scores = cross_val_score(clf, x, y, cv=10)
    print("Time", time.time() - start, "sec")
    print ("K-NN classifier, with k = 5")
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
    print("MIN ACCURACY:", scores.min())
    print("MAX ACCURACY:", scores.max())
    cv_results = cross_validate(KNN(n_neighbors=5), x, y, cv=10)
    print("CROSS VALIDATE FIT_TIME: ", np.sum(cv_results['fit_time']))
    print("----------------------------------------------\n")
    return scores

def run_knn(X,Y):
    KNN_5(X, Y)
    # trained model
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    clf_knn = KNN(n_neighbors=5).fit(X_train, y_train)

    print("K-nn classifier, K = 5")
    print("Actual accuracy of trained K-nn: ", clf_knn.score(X_test, y_test))
    print("Classifier find: ", np.count_nonzero(clf_knn.predict(X_test)),
          "ham emails while in the feature there was: ", np.count_nonzero(y_test))


    # RMSE root-mean-square error for training set
    train_preds = clf_knn.predict(X_train)
    mse = mean_squared_error(y_train, train_preds)
    rmse = sqrt(mse)
    print("RMSE of training set:", rmse)

    # RMSE for test set
    test_preds = clf_knn.predict(X_test)
    mse = mean_squared_error(y_test, test_preds)
    rmse = sqrt(mse)
    print("RMSE of test set:", rmse)


    # NOW WE TRY TO SEARCH IF THERE IS A BETTER COMBINATION FOR K-NN, IF K=5 IS THE BEST ONE
    parameters = {"n_neighbors": range(1, 30)}
    gridsearch = GridSearchCV(KNN(), parameters)
    gridsearch.fit(X_train, y_train)
    print(gridsearch.best_params_)
    '''
        GridSearchCV repeatedly fits kNN  on  part of the data and tests the performances on the remaining part of the data.
        Doing this repeatedly will yield a reliable estimate of the predictive performance of each of the values for k.
        In this example, you test the values from 1 to 30.
        With .best_params_ we can see that the best value for the number of neighbors is usally between 5 and 7,
        so 5 is a good trade-off
    '''
    train_preds_grid = gridsearch.predict(X_train)
    train_mse = mean_squared_error(y_train, train_preds_grid)
    train_rmse = sqrt(train_mse)
    test_preds_grid = gridsearch.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds_grid)
    test_rmse = sqrt(test_mse)
    print("RMSE for training set with gridsearch:", train_rmse)
    print("RMSE for test set with gridsearch:", test_rmse)

    # Plot: number of neighbors against accuracy
    neighbors = np.arange(1, 30)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # Loop over K values
    for i, k in enumerate(neighbors):
        knn = KNN(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Compute training and test data accuracy
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)

    # Generate plot
    fileName = r'plots/accuracy_vs_nn.png'
    fig, ax = plt.subplots(1)
    plt.plot(neighbors, test_accuracy, label='Testing dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training dataset Accuracy')

    plt.legend()
    plt.xlabel('n_neighbors')
    plt.ylabel('Accuracy')
    plt.show()
    fig.savefig(fileName, format='png')
    plt.close(fig)

    # Confusion matrix, in pos (1,0) we have false positives and in (0,1) false negatives
    cm = confusion_matrix(y_test, test_preds)
    print(cm)  # looks like we have 72 false positive and 70 false negatives

    error = []

    # Calculating error for K values between 1 and 40
    for i in range(1, 30):
        knn = KNN(n_neighbors=i)
        knn.fit(X_train, y_train)
        pred_i = knn.predict(X_test)
        error.append(np.mean(pred_i != y_test))

    fileName2 = r'plots/error_rate_kvalue.png'
    fig2, ax2 = plt.subplots(1)
    #plt.figure(figsize=(12, 6))
    plt.plot(range(1, 30), error, color='red', linestyle='dashed', marker='o',
             markerfacecolor='blue', markersize=10)
    plt.title('Error Rate K Value')
    plt.xlabel('K Value')
    plt.ylabel('Mean Error')
    plt.show()
    fig2.savefig(fileName2, format='png')
    plt.close(fig2)
    '''
        usually in most plot that we did the minimum error was when K =5.
        So we can conclude that K=5 is the best value for number of nearest neighbors for our problem.
    '''