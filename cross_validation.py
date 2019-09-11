import preprocessor as pr
import random
import naive_bayes as nb
import numpy as np
import math
from math import floor


def cross_validation(data, folds=10):
    # size fold is just for testing purposes to make sure it splits the data correctly
    size_fold = floor(len(data)/folds)
    print(size_fold)
    # split the data into 10 evenly grouped sets as possible
    arr = np.array_split(data, folds)
    array = np.array(arr)
    print(array)

    for i in range(folds):
        training1 = array[i:folds]
        train = array[:i]
        training_set = np.append(training1, train, axis=0)
        testing_set = array[i]
        real_training_set = np.delete(training_set, 0, axis=0)
        print("Testing set %d: %s" % (i, testing_set))
        print("Training set %d: %s" % (i, real_training_set))


# Returns an array of a dictionary of keys: "testing" and "training" and values: arrays of data.
# Each key is mapped to a value of a 2D array of the data respective to the key.
# Input: Data (as a 2D list) and number of segments or folds to split the data
def ten_fold_cross_validation(data, seg):
    avg = math.ceil(len(data) / seg)
    array_of_segments = []
    array_of_dictionary = []
    last = 0.0

    # Populates array_of_ten_segments by splitting the data
    while last < len(data):
        array_of_segments.append(data[int(last):int(last + avg)])
        last += avg
    # Creates a dictionary that maps testing and training to its respective data
    for i in range(len(array_of_segments)):
        validation = {}
        validation['testing'] = array_of_segments[i]
        copy_of_array = array_of_segments.copy()
        copy_of_array.pop(i)
        training_set = []
        for sets in copy_of_array:
            training_set = training_set + sets[:]
        validation['training'] = training_set
        array_of_dictionary.append(validation)

    # Prints the data to verify
    counter = 1
    for d in array_of_dictionary:
        print("SET: " + str(counter))
        counter += 1
        for key, value in d.items():
            print(key, value)

    return array_of_dictionary


bc_data = pr.open_breast_cancer_data()
# print("BREAST CANCER DATA")
# cross_validation(bc_data, 10)
# bc_array = ten_fold_cross_validation(bc_data, 10)

iris_data = pr.open_iris_data()
# print("IRIS DATA")
# cross_validation(iris_data, 10)
# ten_fold_cross_validation(iris_data, 10)

glass_data = pr.open_glass_data()
# print("GLASS DATA")
# cross_validation(glass_data, 10)
# ten_fold_cross_validation(glass_data, 10)

house_data = pr.open_house_votes_data()
# print("HOUSE DATA")
# cross_validation(house_data, 10)
# ten_fold_cross_validation(house_data, 10)


soybean_data = pr.open_soybean_small()
print("SOYBEAN DATA")
# cross_validation(soybean_data, 10)
ten_fold_cross_validation(soybean_data, 10)




