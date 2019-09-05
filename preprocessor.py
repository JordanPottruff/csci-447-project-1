import csv

BREAST_CANCER_DATA_FILE_NAME = "data/breast-cancer-wisconsin.data"
GLASS_DATA_FILE_NAME = "data/glass.data"
HOUSE_VOTES_DATA_FILE_NAME = "data/house-votes-84.data"
IRIS_DATA_FILE_NAME = "data/iris.data"
SOYBEAN_SMALL_DATA_NAME = "data/soybean-small.data"

# Returns the data from the file as a 2D list (table).
def get_original_data(filename):
    with open(filename) as csvfile:
        data = list(csv.reader(csvfile))
    return data

# Returns the number of rows that have missing feature values.
def num_missing_rows(data):
    missing = 0
    for line in data:
        if '?' in line:
            missing += 1
    return missing

# Returns the input data but with any rows w/ missing feature values removed.
def remove_missing_rows(data):
    newData = []
    for line in data:
        if '?' not in line:
            newData.append(line)
    return newData

def print_data(data):
    for line in data:
        print(line)

# Returns the 2D list of the breast cancer data.
def open_breast_cancer_data():
    return get_original_data(BREAST_CANCER_DATA_FILE_NAME)

