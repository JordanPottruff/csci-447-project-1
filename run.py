import random
import math

import preprocessor as pr
import classify as cl

# Proportion of data that belongs to test set when running algorithm without x-validation.
TEST_SET_PROPORTION = 0.2


def run_breast_cancer_data():
    data = pr.open_breast_cancer_data()
    training, test = shuffle_and_split(data)
    results = cl.classify_breast_cancer_data(training, test)

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\nbreast-cancer-wisconsin.data:")
    print("Accuracy: " + str(calc_accuracy(results)))


# Segments the glass data into test and training and attempts to classify the test set using the training set.
def run_glass_data():
    data = pr.open_glass_data()
    training, test = shuffle_and_split(data)
    results = cl.classify_glass_data(training, test)

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\nglass.data:")
    print("Accuracy: " + str(calc_accuracy(results)))


# Segments the house data into test and training and attempts to classify the test set using the training set.
def run_house_data():
    data = pr.open_house_votes_data()
    training, test = shuffle_and_split(data)
    results = cl.classify_house_data(training, test)

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\nhouse-votes-84.data:")
    print("Accuracy: " + str(calc_accuracy(results)))


# Segments the iris data into test and training and attempts to classify the test set using the training set.
def run_iris_data():
    data = pr.open_iris_data()
    training, test = shuffle_and_split(data)
    results = cl.classify_iris_data(training, test)

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\niris.data:")
    print("Accuracy: " + str(calc_accuracy(results)))


# Segments the soybean data into test and training and attempts to classify the test set using the training set.
def run_soybean_data():
    data = pr.open_soybean_small()
    training, test = shuffle_and_split(data)
    results = cl.classify_soybean_data(training, test)

    # TODO: use loss function on results. Right now we just print accuracy.
    print("\nsoybean-small.data:")
    print("Accuracy: " + str(calc_accuracy(results)))


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


run_breast_cancer_data()
run_glass_data()
run_house_data()
run_iris_data()
run_soybean_data()