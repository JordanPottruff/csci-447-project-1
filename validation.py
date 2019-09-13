# validation.py
#
# Defines functions for validating naive bayes by manipulating our original data. This includes an implementation of
# 10-fold cross validation that returns the 10 pairs of training and test sets. It also includes a function for feature
# shuffling.

import random
import math


# ----------------------------------
# TOP-LEVEL FUNCTIONS FOR VALIDATION
# ----------------------------------


# Returns an array of a dictionary of keys: "testing" and "training" and values: arrays of data.
# Each key is mapped to a value of a 2D array of the data respective to the key.
# Input: Data (as a 2D list)
def ten_fold_cross_validation(data):
    # Shuffles the data to prevent classes from being grouped
    random.shuffle(data)
    # Average segment length uses ceiling to ensure that we never need more than 10 segments.
    avg_seg_length = math.ceil(len(data) / 10)
    array_of_segments = []
    folds = []
    last = 0.0

    # Populates array_of_segments by splitting the data into 10 segments, with the last one potentially having less
    # members than the average if its not an even split.
    while last < len(data):
        array_of_segments.append(data[int(last):int(last + avg_seg_length)])
        last += avg_seg_length
    # Creates a dictionary that maps testing and training to its respective data
    for i in range(len(array_of_segments)):
        # A fold is composed of the testing and training data. Here we set the ith segment as the testing set.
        fold = {'testing': array_of_segments[i]}
        # We then create a list of the remaining segments.
        remaining_segments = array_of_segments.copy()
        remaining_segments.pop(i)
        # Lastly, we combine these segments into our training set.
        training_set = []
        for segment in remaining_segments:
            training_set = training_set + segment[:]
        fold['training'] = training_set
        # This fold then becomes one of the 10 folds used for cross validation.
        folds.append(fold)
    return folds


# Returns the original data but with 10% of the features shuffled. Specifically, if there are n columns, then
# ceil(n/10) features are shuffled.
def feature_shuffling(data, attribute_cols):
    new_data = data.copy()
    # Taking the ceiling ensures that we always shuffle at least one feature.
    num_shuffle = math.ceil(0.10 * len(attribute_cols))
    columns = []
    # We then continue drawing random column indices until we have found 'num_shuffle' number of columns.
    count = 0
    while count < num_shuffle:
        col = random.choice(attribute_cols)
        # If the column has not been selected yet...
        if columns.count(col) == 0:
            # ...append it to the list of visited columns, shuffle it, and increase the count of shuffled columns.
            columns.append(col)
            new_data = shuffle_col(new_data, col)
            count += 1
    return new_data


# -------------------------------
# HELPER FUNCTIONS FOR VALIDATION
# -------------------------------


# Helper function that shuffles a given column in the data.
def shuffle_col(data, col):
    # Extract the column from the data set as a list.
    column = []
    for line in data:
        column.append(line[col])

    # Shuffle the values in the column.
    random.shuffle(column)

    # Replace the column in the output data.
    new_data = []
    for i in range(len(data)):
        line = data[i].copy()
        line[col] = column[i]
        new_data.append(line)
    return new_data
