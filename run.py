import random
import math

import preprocessor as pr
import classify as cl
import cross_validation as cv
import LossFunctions as lf
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

def run_breast_cancer_ten_shuffle_data(n):
    run(n, pr.open_breast_cancer_data(), cl.classify_breast_cancer_data, )

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

# Shuffle 10 per
def shuffle_10_percent_feature(data, attribute_index_array):
    print(attribute_index_array)
    shuffle = []
    for i in range(len(data)):

        for j in range(len(data[i])):
            attribute = []
            if attribute_index_array[j] is 'Y':
                attribute.append(data[i][j])
            print(attribute)
#            shuffle.append(attribute)
    # print(shuffle)


def run_shuffle_breast_cancer_feature():
    bc_data = pr.open_breast_cancer_data()
    attributes_to_shuffle = []
    num_of_attributes = 11
    for i in range(num_of_attributes):
        probability = random.randrange(num_of_attributes)
        if probability is 0:
            attributes_to_shuffle = attributes_to_shuffle + ['Y']
        else:
            attributes_to_shuffle = attributes_to_shuffle + ['N']
    attributes_to_shuffle[0] = 'N'
    attributes_to_shuffle[-1] = 'N'
    shuffle_10_percent_feature(bc_data, attributes_to_shuffle)


run_shuffle_breast_cancer_feature()

# run_breast_cancer_data(50)
# run_glass_data(50)
# run_house_data(50)
# run_iris_data(50)
# run_soybean_data(50)

# run_breast_cancer_data_cross_validation()
# run_glass_data_cross_validation()
# run_house_data_cross_validation()
# run_iris_data_cross_validation()
# run_soybean_data_cross_validation()

# run_breast_cancer_data_cross_fold()
# run_glass_data_cross_fold()
# run_house_data_cross_fold()
# run_iris_data_cross_fold()
# run_soybean_data_cross_fold()

