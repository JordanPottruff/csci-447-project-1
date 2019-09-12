import random
import math

import preprocessor as pr
import classify as cl
import cross_validation as cv
import LossFunctions as lf
# Proportion of data that belongs to test set when running algorithm without x-validation.
TEST_SET_PROPORTION = 0.2


def run_breast_cancer_data_cross_fold():
    run_cross_fold(pr.open_breast_cancer_data(), cl.classify_breast_cancer_data, "breast-cancer-wisconsin.data")


def run_glass_data_cross_fold():
    run_cross_fold(pr.open_glass_data(), cl.classify_glass_data, "glass.data")


def run_house_data_cross_fold():
    run_cross_fold(pr.open_house_votes_data(), cl.classify_house_data, "house-votes-84.data")


def run_iris_data_cross_fold():
    run_cross_fold(pr.open_iris_data(), cl.classify_iris_data, "iris.data")


def run_soybean_data_cross_fold():
    run_cross_fold(pr.open_soybean_small(), cl.classify_soybean_data, "soybean-small.data")


def run_breast_cancer_data(n):
    data = pr.open_breast_cancer_data()
    accuracy_sum = 0
    for i in range(n):
        training, test = shuffle_and_split(data)
        results = cl.classify_breast_cancer_data(training, test)
        accuracy_sum += calc_accuracy(results)

    average_accuracy = accuracy_sum / n

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\nbreast-cancer-wisconsin.data @n=" + str(n))
    print("Avg. Accuracy: " + str(average_accuracy))


# Segments the glass data into test and training and attempts to classify the test set using the training set.
def run_glass_data(n):
    data = pr.open_glass_data()
    accuracy_sum = 0
    for i in range(n):
        training, test = shuffle_and_split(data)
        results = cl.classify_glass_data(training, test)
        accuracy_sum += calc_accuracy(results)

    average_accuracy = accuracy_sum / n

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\nglass.data @n=" + str(n))
    print("Avg. Accuracy: " + str(average_accuracy))


# Segments the house data into test and training and attempts to classify the test set using the training set.
def run_house_data(n):
    data = pr.open_house_votes_data()
    accuracy_sum = 0
    for i in range(n):
        training, test = shuffle_and_split(data)
        results = cl.classify_house_data(training, test)
        accuracy_sum += calc_accuracy(results)

    average_accuracy = accuracy_sum / n

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\nhouse-votes-84.data @n=" + str(n))
    print("Avg. Accuracy: " + str(average_accuracy))


# Segments the iris data into test and training and attempts to classify the test set using the training set.
def run_iris_data(n):
    data = pr.open_iris_data()
    accuracy_sum = 0
    for i in range(n):
        training, test = shuffle_and_split(data)
        results = cl.classify_iris_data(training, test)
        accuracy_sum += calc_accuracy(results)

    average_accuracy = accuracy_sum / n

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\niris.data @n=" + str(n))
    print("Avg. Accuracy: " + str(average_accuracy))


# Segments the soybean data into test and training and attempts to classify the test set using the training set.
def run_soybean_data(n):
    data = pr.open_soybean_small()
    accuracy_sum = 0
    for i in range(n):
        training, test = shuffle_and_split(data)
        results = cl.classify_soybean_data(training, test)
        accuracy_sum += calc_accuracy(results)

    average_accuracy = accuracy_sum / n

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\nsoybean-small.data @n=" + str(n))
    print("Avg. Accuracy: " + str(average_accuracy))


def run_cross_fold(data, classification_function, filename):
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


# run_breast_cancer_data(50)
# run_glass_data(50)
# run_house_data(50)
# run_iris_data(50)
# run_soybean_data(50)
run_breast_cancer_data_cross_fold()
# run_glass_data_cross_fold()
# run_house_data_cross_fold()
# run_iris_data_cross_fold()
# run_soybean_data_cross_fold()
