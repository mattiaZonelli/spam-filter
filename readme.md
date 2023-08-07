The goal of the assignment is to write a spam filter using discriminative and generative classifiers. Use the Spambase dataset which already represents spam/ham messages through a bag-of-words representations through a dictionary of 48 highly discriminative words and 6 characters. The first 54 features correspond to word/symbols frequencies; we ignore features 55-57; feature 58 is the class label (1 spam/0 ham).
- Perform SVM classification using linear, polynomial of degree 2, and RBF kernels over the TF/IDF representation.
  In order to use angular information only, it has been applied kernel transformation.
- Classify the same data also through a Naive Bayes classifier for continuous inputs, modeling each feature with a Gaussian distribution.
- Perform k-NN classification with k=5

For SVM and k-NN we use the functions provided by sklearn while we coded the Naive Bayes algorithm.
Before applying SVM classifier we needed to use TF-IDF on the original dataset.
![image](https://github.com/mattiaZonelli/spam-filter/assets/22390331/7bf416df-1d94-468b-b74c-54a6f046ba5a)

Comparison between Naive Bayes and SVM with linear kernel ![image](https://github.com/mattiaZonelli/spam-filter/assets/22390331/31143d9e-9c25-493c-92a8-0ca240625fd2)

Results for k-NN
![image](https://github.com/mattiaZonelli/spam-filter/assets/22390331/c22f5181-8cb8-42b6-a1d3-57941c982e64)

