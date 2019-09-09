import csv
import math
import random

BREAST_CANCER_DATA_FILE_NAME = "data/breast-cancer-wisconsin.data"
GLASS_DATA_FILE_NAME = "data/glass.data"
HOUSE_VOTES_DATA_FILE_NAME = "data/house-votes-84.data"
IRIS_DATA_FILE_NAME = "data/iris.data"
SOYBEAN_SMALL_DATA_NAME = "data/soybean-small.data"

##
#  The functions below should be used to get the preprocessed data for each data set.
##

# Returns the 2D list of the breast cancer data. Missing values removed, no categorizing necessary.
def open_breast_cancer_data():
    original = get_original_data(BREAST_CANCER_DATA_FILE_NAME)
    return remove_missing_rows(original)


# Returns the 2D list of the glass data. Missing values removed, relevant columns categorized.
def open_glass_data():
    data = get_original_data(GLASS_DATA_FILE_NAME)
    data = remove_missing_rows(data)

    bins = ['LOW', 'MED', 'HIGH']

    for col in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        data = discretize(data, col, bins)

    return data


# Returns the 2D list of house votes data. Missing values replaced based on frequency, no categorizing necessary.
def open_house_votes_data():
    data = get_original_data(HOUSE_VOTES_DATA_FILE_NAME)
    return data


# Returns the 2D list of iris data. Missing values removed, relevant columns categorized.
def open_iris_data():
    data = get_original_data(IRIS_DATA_FILE_NAME)
    data = remove_missing_rows(data)

    min_val = 0
    max_val = 10

    data = discretize_equal_width(data, 0, min_val, max_val, 15)
    data = discretize_equal_width(data, 1, min_val, max_val, 10)
    data = discretize_equal_width(data, 2, min_val, max_val, 4)
    data = discretize_equal_width(data, 3, min_val, max_val, 12)

    return data


# Returns the 2D list of soybean data. Missing values removed (not needed), and no categorizing necessary.
def open_soybean_small():
    data = get_original_data(SOYBEAN_SMALL_DATA_NAME)
    return remove_missing_rows(data)

##
# Functions below are used for doing the actual preprocessing.
##


# Returns the data from the file as a 2D list (table).
def get_original_data(filename):
    with open(filename) as csvfile:
        data = list(csv.reader(csvfile))
    empty_removed = []
    for line in data:
        if line:
            empty_removed.append(line)
    return empty_removed


# Returns the number of rows that have missing feature values.
def num_missing_rows(data):
    missing = 0
    for line in data:
        if '?' in line:
            missing += 1
    return missing


# Returns the input data but with any rows w/ missing feature values removed.
def remove_missing_rows(data):
    new_data = []
    for line in data:
        if '?' not in line:
            new_data.append(line)
    return new_data


# Returns a new 2D list with missing values for each column replaced based on the frequency of values in that column.
def replace_missing_rows(data, cols, class_col):
    new_data = []
    for i in range(len(data)):
        new_data.append(data[i].copy())
        for col in cols:
            if data[i][col] == '?':
                class_name = data[i][class_col]
                new_data[i][col] = get_replacement(data, col, class_col, class_name)

    return new_data


# Randomly selects a non-missing value from the column.
def get_replacement(data, col, class_col, class_name):
    non_missing = []
    for line in data:
        if line[col] != '?' and line[class_col] == class_name:
            non_missing.append(line[col])

    return random.choice(non_missing)


# Prints the data (2D list) line-by-line.
def print_data(data):
    for line in data:
        print(line)


# Transforms continuous (real) values in the specified column to be 'discrete' by categorizing them with the given bin
# names. For example, if bins=["Low", "Medium", "High"], the bottom 33% of values in the specified column will become
# "Low", the next 33% will become "Medium", and the last 33% will become "High". This works for any number of bins.
def discretize(data, col, bins):
    # new_data will be the final version of our data that is returned.
    new_data = []
    # column will be a list of the values for the column being discretized.
    column = []
    for line in data:
        new_data.append(line.copy())
        column.append(float(line[col]))

    # sorting the column lets us find the values for each "percentile" by using indices.
    sorted_column = column.copy()
    sorted_column.sort()

    # find the maximum value (cutoff) that can belong to each bin.
    bin_width = len(column)/len(bins)
    bin_cutoffs = []
    for i in range(len(bins)-1):
        val = sorted_column[math.ceil((i + 1) * bin_width)]
        bin_cutoffs.append(val)

    # the last bin should have a maximum equal to the largest item.
    bin_cutoffs.append(sorted_column[-1])

    # replace the continuous values with their corresponding bin names based on our cutoffs.
    for i in range(len(data)):
        for b in range(len(bins)):
            if float(new_data[i][col]) <= bin_cutoffs[b]:
                new_data[i][col] = bins[b]
                break

    # again, we didn't modify the original data -- only this new_data variable.
    return new_data


def discretize_equal_width(data, col, start, end, count):
    bin_cutoffs = []
    width = (end-start) / count
    for i in range(count):
        bin_cutoffs.append(start + (i+1) * width)

    bin_cutoffs[-1] = float('inf')

    new_data = []

    for line in data:
        new_line = line.copy()
        for i in range(len(bin_cutoffs)):
            cutoff = bin_cutoffs[i]

            if float(line[col]) <= cutoff:
                new_line[col] = i
                break

        new_data.append(new_line)
    return new_data
