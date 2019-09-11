
import preprocessor as pr
import random


# Calculates Q(C = ci).
def calc_Q(data, class_col, class_name):
    n = len(data)

    class_count = calc_class_count(data, class_col, class_name)

    return class_count / n


# Calculates the number of observations in the data that belong to the specified class.
def calc_class_count(data, class_col, class_name):
    class_n = 0
    for line in data:
        if line[class_col] == class_name:
            class_n += 1

    return class_n


# Calculates the number of observations in the data that belong to the specified class and have the specified attribute
# value for the specified attribute.
def calc_attribute_count(data, attribute_col, attribute_value, class_col, class_name):
    attribute_n = 0
    for line in data:
        if line[class_col] == class_name and line[attribute_col] == attribute_value:
            attribute_n += 1

    return attribute_n


# Calculates F(Aj = ak, C = ci).
def calc_F(data, attribute_col, attribute_value, class_col, class_name, num_attributes):
    attr_count = calc_attribute_count(data, attribute_col, attribute_value, class_col, class_name)
    class_n = calc_class_count(data, class_col, class_name)

    return (attr_count + 1) / (class_n + num_attributes)


# Calculates the probability that the specified observation belongs to the specified class based on the 'data', i.e. the
# training set.
def calc_C(data, observation, attribute_cols, class_col, class_name):
    product = 1

    for attribute_col in attribute_cols:
        product *= calc_F(data, attribute_col, observation[attribute_col], class_col, class_name, len(attribute_cols))

    return calc_Q(data, class_col, class_name) * product


def pick_highest(classes):
    max_val = float("-inf")
    max_class = ""
    for c in classes:
        if(c[1] > max_val):
            max_val = c[1]
            max_class = c[0]

    return max_class


# A test using the house data.
def test_house_data():
    data = pr.open_house_votes_data()
    random.shuffle(data)

    training = data[:300]
    test = data[300:]

    attribute_cols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    class_col = 0

    correct = 0
    incorrect = 0
    for line in test:
        rep = ('republican', calc_C(training, line, attribute_cols, class_col, 'republican'))
        dem = ('democrat', calc_C(training, line, attribute_cols, class_col, 'democrat'))

        choice = pick_highest([rep, dem])

        actual = line[class_col]

        if choice != actual:
            incorrect += 1
        else:
            correct += 1

    print("House data set: ")
    print("Accuracy: " + str(correct / (correct + incorrect)))


# A test using the iris data.
def test_iris_data():
    data = pr.open_iris_data()

    random.shuffle(data)

    training = data[:130]
    test = data[130:]

    attribute_cols = [0, 1, 2, 3]
    class_col = 4

    correct = 0
    incorrect = 0
    for line in test:
        setosa = ("Iris-setosa", calc_C(training, line, attribute_cols, class_col, 'Iris-setosa'))
        veriscolor = ("Iris-versicolor", calc_C(training, line, attribute_cols, class_col, 'Iris-versicolor'))
        virginica = ("Iris-virginica", calc_C(training, line, attribute_cols, class_col, 'Iris-virginica'))

        choice = pick_highest([setosa, veriscolor, virginica])
        actual = line[class_col]

        if choice != actual:
            incorrect += 1
        else:
            correct += 1

    print("Iris data set: ")
    print("Accuracy: " + str(correct / (correct + incorrect)))


def test_soybean_data():
    data = pr.open_soybean_small()

    random.shuffle(data)

    training = data[:40]
    test = data[40:]

    attribute_cols = list(range(35))
    class_col = 35

    correct = 0
    incorrect = 0
    for line in test:
        d1 = ('D1', calc_C(training, line, attribute_cols, class_col, 'D1'))
        d2 = ('D2', calc_C(training, line, attribute_cols, class_col, 'D2'))
        d3 = ('D3', calc_C(training, line, attribute_cols, class_col, 'D3'))
        d4 = ('D4', calc_C(training, line, attribute_cols, class_col, 'D4'))

        choice = pick_highest([d1, d2, d3, d4])

        actual = line[class_col]

        if choice != actual:
            incorrect += 1
        else:
            correct += 1

    print("Soybean data set: ")
    print("Accuracy: " + str(correct / (correct + incorrect)))


# test_house_data()
# test_iris_data()
# test_soybean_data()
