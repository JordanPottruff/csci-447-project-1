# run.py
#
# This is the top-level file of our project. It includes three sets of five functions each that are used for running our
# classifier. The first set does cross-fold validation, the second does a 80/20 training/test split n-times, and the
# third also does an 80/20 split but adds feature shuffling. The main() function listed at the very bottom of the file
# runs each one of these tests on each data set, using n=25 when applicable. The results of the tests are displayed
# directly to the standard output, and include the (1) accuracy, (2) mean square error, and (3) cross entropy error.

import random
import math

import preprocessor as pr
import classify as cl
import validation as cv
import loss_function as lf


# Proportion of data that belongs to test set when running algorithm without x-validation.
TEST_SET_PROPORTION = 0.2


# ---------------------------------------------
# FUNCTIONS FOR RUNNING CROSS FOLDS ON THE DATA
# ---------------------------------------------


# Runs a 10-fold cross validation of our Naive Bayes classifier on the breast cancer data set.
def run_breast_cancer_data_cross_validation():
    run_cross_validation(pr.open_breast_cancer_data(), cl.classify_breast_cancer_data, "breast-cancer-wisconsin.data")


# Runs a 10-fold cross validation of our Naive Bayes classifier on the glass data set.
def run_glass_data_cross_validation():
    run_cross_validation(pr.open_glass_data(), cl.classify_glass_data, "glass.data")


# Runs a 10-fold cross validation of our Naive Bayes classifier on the house votes data set.
def run_house_data_cross_validation():
    run_cross_validation(pr.open_house_votes_data(), cl.classify_house_data, "house-votes-84.data")


# Runs a 10-fold cross validation of our Naive Bayes classifier on the iris data set.
def run_iris_data_cross_validation():
    run_cross_validation(pr.open_iris_data(), cl.classify_iris_data, "iris.data")


# Runs a 10-fold cross validation of our Naive Bayes classifier on the soybean data set.
def run_soybean_data_cross_validation():
    run_cross_validation(pr.open_soybean_small(), cl.classify_soybean_data, "soybean-small.data")


# ----------------------------------------------------
# FUNCTIONS FOR RUNNING WITH 20-80 TEST/TRAINING SPLIT
# ----------------------------------------------------


# Runs a 20-80 test/training split n-times on the breast cancer data set, returning the averages of our loss metrics.
def run_breast_cancer_data(n):
    run(n, pr.open_breast_cancer_data(), cl.classify_breast_cancer_data, "breast-cancer-wisconsin.data")


# Runs a 20-80 test/training split n-times on the glass data set, returning the averages of our loss metrics.
def run_glass_data(n):
    run(n, pr.open_glass_data(), cl.classify_glass_data, "glass.data")


# Runs a 20-80 test/training split n-times on the house votes data set, returning the averages of our loss metrics.
def run_house_data(n):
    run(n, pr.open_house_votes_data(), cl.classify_house_data, "house-votes-84.data")


# Runs a 20-80 test/training split n-times on the iris data set, returning the averages of our loss metrics.
def run_iris_data(n):
    run(n, pr.open_iris_data(), cl.classify_iris_data, "iris.data")


# Runs a 20-80 test/training split n-times on the soybean data set, returning the averages of our loss metrics.
def run_soybean_data(n):
    run(n, pr.open_soybean_small(), cl.classify_soybean_data, "soybean-small.data")


# -----------------------------------------------------
# FUNCTIONS FOR RUNNING WITH 10% OF ATTRIBUTES SHUFFLED
# -----------------------------------------------------


# Runs a ten percent attribute shuffle split n-times on the breast cancer data,
# returning the averages of our loss metrics.
def run_breast_cancer_shuffle_data(n):
    data = pr.open_breast_cancer_data()
    data = cv.feature_shuffling(data, pr.BREAST_CANCER_ATTR_COLS)
    run(n, data, cl.classify_breast_cancer_data, pr.BREAST_CANCER_DATA_FILE_NAME)


# Runs a ten percent attribute shuffle split n-times on the glass data, returning the averages of our loss metrics.
def run_glass_shuffle_data(n):
    data = pr.open_glass_data()
    data = cv.feature_shuffling(data, pr.GLASS_DATA_ATTR_COLS)
    run(n, data, cl.classify_glass_data, pr.GLASS_DATA_FILE_NAME)


# Runs a ten percent attribute shuffle split n-times on the house votes data,
# returning the averages of our loss metrics.
def run_house_shuffle_data(n):
    data = pr.open_house_votes_data()
    data = cv.feature_shuffling(data, pr.HOUSE_VOTES_DATA_ATTR_COLS)
    run(n, data, cl.classify_house_data, pr.HOUSE_VOTES_DATA_FILE_NAME)


# Runs a ten percent attribute shuffle split n-times on the iris data, returning the averages of our loss metrics.
def run_iris_shuffle_data(n):
    data = pr.open_iris_data()
    data = cv.feature_shuffling(data, pr.IRIS_DATA_ATTR_COLS)
    run(n, data, cl.classify_iris_data, pr.IRIS_DATA_FILE_NAME)


# Runs a ten percent attribute shuffle split n-times on the soy bean data,
# returning the averages of our loss metrics.
def run_soy_bean_shuffle_data(n):
    data = pr.open_soybean_small()
    data = cv.feature_shuffling(data, pr.SOYBEAN_SMALL_ATTR_COLS)
    run(n, data, cl.classify_soybean_data, pr.SOYBEAN_SMALL_DATA_NAME)


# ----------------------
# MISC. HELPER FUNCTIONS
# ----------------------


# Runs a 10-fold cross validation given a 2D list of the data, the function that returns the result of the Naive Bayes
# classifier, and the file name of the data set. Outputs the average metrics across all folds to the console.
def run_cross_validation(data, classification_function, filename):
    trials = cv.ten_fold_cross_validation(data)

    accuracy_sum = 0
    mean_square_error_loss = 0
    cross_entropy_loss = 0

    for trial in trials:
        training = trial["training"]
        testing = trial["testing"]

        results = classification_function(training, testing)

        accuracy_sum += calc_accuracy(results)
        mean_square_error_loss += lf.loss_function(results, "MSE")
        cross_entropy_loss += lf.loss_function(results, "Cross_Entropy")

    average_accuracy = accuracy_sum / 10
    average_mse_loss = mean_square_error_loss / 10
    average_cross_entropy_loss = cross_entropy_loss / 10

    print("\n" + str(filename) + " @cross-fold=10")
    print("Avg. Accuracy: " + str(average_accuracy))
    print("Avg. Mean Square Error Loss: " + str(average_mse_loss))
    print("Avg. Cross Entropy Error Loss: " + str(average_cross_entropy_loss))


# Runs the Naive Bayes classification on a random 20/80 test/training split n-times. Requires the number of times to
# run the classification, a 2D list of the data, the function that returns the result of the Naive Bayes classifier,
# and the file name of the data set. Outputs the average metrics across all folds to the console.
def run(n, data, classification_function, filename):
    accuracy_sum = 0
    mean_square_error_loss = 0
    cross_entropy_loss = 0

    for i in range(n):
        training, test = shuffle_and_split(data)
        results = classification_function(training, test)

        accuracy_sum += calc_accuracy(results)
        mean_square_error_loss += lf.loss_function(results, "MSE")
        cross_entropy_loss += lf.loss_function(results, "Cross_Entropy")

    average_accuracy = accuracy_sum / n
    average_mse_loss = mean_square_error_loss / n
    average_cross_entropy_loss = cross_entropy_loss / n

    print("\n" + filename + " @n=" + str(n))
    print("Avg. Accuracy: " + str(average_accuracy))
    print("Avg. Mean Square Error Loss: " + str(average_mse_loss))
    print("Avg. Cross Entropy Error Loss: " + str(average_cross_entropy_loss))


# (1) shuffles the data, and (2) segments it into a training and test set.
def shuffle_and_split(data):
    # Shuffle data so classes aren't grouped.
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Segment data into training and test, using TEST_SET_PROPORTION as the proportion of observations that should
    # belong to the test set.
    n = len(shuffled_data)
    split_point = math.floor(n * TEST_SET_PROPORTION)
    test = shuffled_data[:split_point]
    training = shuffled_data[split_point:]

    return training, test


# Simple metric for analyzing the algorithm. If the class with the highest probability is the actual class, then we
# consider the observation to have been correctly classified. Otherwise, it was incorrectly classified. We then
# calculate accuracy as correct / N where N is the number of test observations.
def calc_accuracy(results):
    correct = 0
    incorrect = 0
    for result in results:
        actual_class = result[0]
        chosen_class = pick_highest(result[1])

        if actual_class == chosen_class:
            correct += 1
        else:
            incorrect += 1
    return correct / (correct + incorrect)


# Given a probability map in the form {*class name*: *probability belongs to class*, ...}, this will return the class
# with the highest associated probability.
def pick_highest(probability_map):
    max_probability = float("-inf")
    max_class = None
    for potential_class in probability_map:
        probability = probability_map[potential_class]
        if float(probability) > max_probability:
            max_probability = float(probability)
            max_class = potential_class
    return max_class


# Runs all 80/20 split, 80/20 shuffle split, and 10-Fold Cross Validation test on loss functions (Mean Square Error Loss
# and Cross Entropy Error Loss) n number of times to determine the average accuracy of each test set.
def main():
    n = 25
    # Run basic test with a 80/20 split.
    print("\n---------------------------------------")
    print("TESTING SET: 80/20 SPLIT")
    print("---------------------------------------")
    run_breast_cancer_data(n)
    run_glass_data(n)
    run_house_data(n)
    run_iris_data(n)
    run_soybean_data(n)
    # Run test with an 80/20 split and 10% feature shuffling.
    print("\n---------------------------------------")
    print("TESTING SET: 80/20 SHUFFLE SPLIT")
    print("---------------------------------------")
    run_breast_cancer_shuffle_data(n)
    run_glass_shuffle_data(n)
    run_house_shuffle_data(n)
    run_iris_shuffle_data(n)
    run_soy_bean_shuffle_data(n)
    # Run 10-fold cross validation.
    print("\n---------------------------------------")
    print("TESTING SET: 10 FOLD CROSS VALIDATION")
    print("---------------------------------------")
    run_breast_cancer_data_cross_validation()
    run_glass_data_cross_validation()
    run_house_data_cross_validation()
    run_iris_data_cross_validation()
    run_soybean_data_cross_validation()


main()

