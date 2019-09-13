
import random
import math


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


def feature_shuffling(data, attribute_cols):
    new_data = data.copy()
    num_shuffle = math.ceil(0.10 * len(attribute_cols))
    columns = []
    count = 0
    while count < num_shuffle:
        col = random.choice(attribute_cols)
        if columns.count(col) == 0:
            columns.append(col)
            new_data = shuffle_col(new_data, col)
            count += 1
    return new_data


def shuffle_col(data, col):
    column = []
    for line in data:
        column.append(line[col])

    random.shuffle(column)

    new_data = []
    for i in range(len(data)):
        line = data[i].copy()
        line[col] = column[i]
        new_data.append(line)
    return new_data
