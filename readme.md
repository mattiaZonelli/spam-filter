The goal of the assignment is to write a spam filter using discriminative and generative classifiers. Use the Spambase dataset which already represents spam/ham messages through a bag-of-words representations through a dictionary of 48 highly discriminative words and 6 characters. The first 54 features correspond to word/symbols frequencies; we ignore features 55-57; feature 58 is the class label (1 spam/0 ham).
- Perform SVM classification using linear, polynomial of degree 2, and RBF kernels over the TF/IDF representation.
  In order to use angular information only, it has been applied kernel transformation.
- Classify the same data also through a Naive Bayes classifier for continuous inputs, modeling each feature with a Gaussian distribution.
- Perform k-NN classification with k=5
