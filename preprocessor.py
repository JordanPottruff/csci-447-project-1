import csv
import math

BREAST_CANCER_DATA_FILE_NAME = "data/breast-cancer-wisconsin.data"
GLASS_DATA_FILE_NAME = "data/glass.data"
HOUSE_VOTES_DATA_FILE_NAME = "data/house-votes-84.data"
IRIS_DATA_FILE_NAME = "data/iris.data"
SOYBEAN_SMALL_DATA_NAME = "data/soybean-small.data"

BREAST_CANCER_ATTR_COLS = list(range(1, 10))
GLASS_DATA_ATTR_COLS = list(range(1, 10))
HOUSE_VOTES_DATA_ATTR_COLS = list(range(1, 17))
IRIS_DATA_ATTR_COLS = list(range(0, 4))
SOYBEAN_SMALL_ATTR_COLS = list(range(0, 35))


# -----------------------------------------------------
# FUNCTIONS FOR OPENING CLEANED DATA SETS AS A 2D LIST.
# -----------------------------------------------------

# Returns the 2D list of the breast cancer data. Missing values removed, no categorizing necessary.
def open_breast_cancer_data():
    original = get_original_data(BREAST_CANCER_DATA_FILE_NAME)
    return remove_missing_rows(original)


# Returns the 2D list of the glass data. Missing values removed, relevant columns categorized.
def open_glass_data():
    data = get_original_data(GLASS_DATA_FILE_NAME)
    data = remove_missing_rows(data)

    # Equal-frequency discretization of refractive index attribute.
    data = discretize_equal_freq(data, 1, 10)
    # Equal-width discretization of remaining weight percentage attributes.
    bin_start = 0       # Bin 0 will begin at 0.
    bin_width = 1       # Each bin will capture a range of length 1.
    for col in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        data = discretize_equal_width(data, col, bin_start, bin_width)
    return data


# Returns the 2D list of house votes data. No missing values, no categorizing necessary.
def open_house_votes_data():
    data = get_original_data(HOUSE_VOTES_DATA_FILE_NAME)
    return data


# Returns the 2D list of iris data. Missing values removed, relevant columns categorized.
def open_iris_data():
    data = get_original_data(IRIS_DATA_FILE_NAME)
    data = remove_missing_rows(data)

    # Discretize all four attribute columns using equal-width discretization.
    bin_start = 0
    data = discretize_equal_width(data, 0, bin_start, 1)
    data = discretize_equal_width(data, 1, bin_start, 1)
    data = discretize_equal_width(data, 2, bin_start, 1)
    data = discretize_equal_width(data, 3, bin_start, 1)

    return data


# Returns the 2D list of soybean data. Missing values removed (not needed), and no categorizing necessary.
def open_soybean_small():
    data = get_original_data(SOYBEAN_SMALL_DATA_NAME)
    return remove_missing_rows(data)


# ---------------------------------------
# HELPER FUNCTIONS FOR PREPROCESSING DATA
# ---------------------------------------


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


# Prints the data (2D list) line-by-line.
def print_data(data):
    for line in data:
        print(line)


# Discretization using bins of equal-frequency that store increasingly higher values. This means that roughly the same
# proportion of the rows will be discretized to each bin for the given column. For example, 3 bins would lead to bin 0
# to represent the lowest third of values, bin 1 the next third, and bin 2 the top third.
def discretize_equal_freq(data, col, num_bins):
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
    bin_width = len(column)/num_bins
    bin_cutoffs = []
    for i in range(num_bins-1):
        val = sorted_column[math.ceil((i + 1) * bin_width)]
        bin_cutoffs.append(val)

    # the last bin should have a maximum equal to the largest item.
    bin_cutoffs.append(sorted_column[-1])

    # replace the continuous values with their corresponding bin names based on our cutoffs.
    for i in range(len(data)):
        for bin in range(num_bins):
            if float(new_data[i][col]) <= bin_cutoffs[bin]:
                new_data[i][col] = bin
                break

    # again, we didn't modify the original data -- only this new_data variable.
    return new_data


# Discretization using bins that represent ranges of values that are of equal width. The continuous will be replaced
# with an integer representing the bin number. The origin specifies the value that should be the lowest possible value
# for bin 0. If any values are below the origin, they will take on a negative bin number. The width determines how large
# each bin's range is. For example, bin 0 is (origin, origin + width),  is (origin, origin + 2*width), etc.
def discretize_equal_width(data, col, origin, width):
    # Don't discretize in place; return a new data set with discretization.
    discretized_data = []

    for line in data:
        # Create copy of line to ensure we don't manipulate original data.
        line_cpy = line.copy()

        # Determine the "bin", represented by a number. The range (origin, origin + width) is 0, then (origin, origin +
        # 2 * width) is 1, etc (with negatives for below the origin).
        continuous_value = float(line_cpy[col])
        bin = math.floor((continuous_value - origin) / width)
        line_cpy[col] = bin

        # Add discretized line to the data set copy we are building.
        discretized_data.append(line_cpy)

    return discretized_data