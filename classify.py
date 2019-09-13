# classify.py
#
# Defines functions for each data set that runs the naive-bayes algorithm on each test observation using the specified
# training data set.

import naive_bayes as nb


# ------------------------------------------------------------------------
# FUNCTIONS FOR EXECUTING THE NAIVE BAYES CLASSIFIER ON SPECIFIC DATA SETS
# ------------------------------------------------------------------------
# Each returns the results of classifying the test data using the specified training data. The result is a list of
# tuples that represents the results for each row in the test data. The first item in the tuple is the actual class for
# the observation, and the second item in the tuple is a map of each class to the calculated probability that the
# observation belongs to the class.


# Classifies a list of test observations from the breast cancer data set based on a list of training data from the same
# data set.
def classify_breast_cancer_data(training, test):
    classes = ['2', '4']
    attribute_cols = list(range(1, 10))
    class_col = 10
    return classify(training, test, classes, attribute_cols, class_col)


# Classifies a list of test observations from the glass data set based on a list of training data from the same data
# set.
def classify_glass_data(training, test):
    classes = ['1', '2', '3', '4', '5', '6', '7']
    attribute_cols = list(range(1, 10))
    class_col = 10
    return classify(training, test, classes, attribute_cols, class_col)


# Classifies a list of test observations from the house voting data set based on a list of training data from the same
# data set.
def classify_house_data(training, test):
    classes = ['republican', 'democrat']
    attribute_cols = list(range(1, 17))
    class_col = 0
    return classify(training, test, classes, attribute_cols, class_col)


# Classifies a list of test observations from the iris data set based on a list of training data from the same data set.
def classify_iris_data(training, test):
    classes = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    attribute_cols = list(range(4))
    class_col = 4
    return classify(training, test, classes, attribute_cols, class_col)


# Classifies a list of test observations from the soybean data set based on a list of training data from the same data
# set.
def classify_soybean_data(training, test):
    classes = ['D1', 'D2', 'D3', 'D4']
    attribute_cols = list(range(35))
    class_col = 35
    return classify(training, test, classes, attribute_cols, class_col)


# -------------------------------------------------------------------
# HELPER FUNCTION FOR CLASSIFYING BASED ON CLASSES, COLUMN TYPES, ETC
# -------------------------------------------------------------------


# Generalized function for classifying observations in a test set using observations in a training set.
def classify(training, test, classes, attribute_cols, class_col):
    results = []
    # For each observation in the test data set...
    for line in test:
        probabilities = {}
        # ...and each possible class...
        for cls in classes:
            # ...calculate the probability that the observation belongs to the class.
            probabilities[cls] = nb.calc_classification_probability(training, line, attribute_cols, class_col, cls)

        # Save as a tuple in the form (*Actual class*, *Map of class to probability*).
        results.append((line[class_col], probabilities))
    return results
