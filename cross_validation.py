import preprocessor as pr
import random
import math
import pandas as pd
import numpy as np

# Returns an array of a dictionary of keys: "testing" and "training" and values: arrays of data.
# Each key is mapped to a value of a 2D array of the data respective to the key.
# Input: Data (as a 2D list)
def ten_fold_cross_validation(data):
    # Shuffles the data to prevent data from being grouped
    random.shuffle(data)
    seg = 10
    avg = math.ceil(len(data) / seg)
    array_of_segments = []
    array_of_dictionary = []
    last = 0.0

    # Populates array_of_segments by splitting the data to 10
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
    return array_of_dictionary


def feature_shuffling(data):
    data_set = pd.DataFrame(data)
    num_shuffle = math.ceil(0.10 * data_set.shape[1])
    for i in range(0, num_shuffle):
        rand_col = random.randint(0, data_set.shape[1])
        to_shuffle = data_set[rand_col]
        shuffled = np.random.permutation(data_set[rand_col].values)
    return shuffled


# TEST EACH DATA SET
# --------------------------------------
# bc_data = pr.open_breast_cancer_data()
# ten_fold_cross_validation(bc_data)
# feature_shuffling(bc_data)

# iris_data = pr.open_iris_data()
# ten_fold_cross_validation(iris_data)
# feature_shuffling(iris_data)

# glass_data = pr.open_glass_data()
# ten_fold_cross_validation(glass_data)
# feature_shuffling(glass_data)

# house_data = pr.open_house_votes_data()
# ten_fold_cross_validation(house_data)
# feature_shuffling(house_data)

# soybean_data = pr.open_soybean_small()
# ten_fold_cross_validation(soybean_data)
# feature_shuffling(soybean_data)