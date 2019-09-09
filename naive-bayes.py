<<<<<<< HEAD
=======
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
        rep = calc_C(training, line, attribute_cols, class_col, 'republican')
        dem = calc_C(training, line, attribute_cols, class_col, 'democrat')

        if rep > dem:
            if line[0] == 'republican':
                correct += 1
            else:
                incorrect += 1
        else:
            if line[0] == 'republican':
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
        setosa = calc_C(training, line, attribute_cols, class_col, 'Iris-setosa')
        veriscolor = calc_C(training, line, attribute_cols, class_col, 'Iris-versicolor')
        virginica = calc_C(training, line, attribute_cols, class_col, 'Iris-virginica')

        actual = line[class_col]

        if setosa > veriscolor and setosa > virginica:
            if actual == 'Iris-setosa':
                correct += 1
            else:
                incorrect += 1

        if veriscolor > setosa and veriscolor > virginica:
            if actual == 'Iris-veriscolor':
                correct += 1
            else:
                incorrect += 1

        if virginica > veriscolor and virginica > setosa:
            if actual == 'Iris-virginica':
                correct += 1
            else:
                incorrect += 1

    print("Iris data set: ")
    print("Accuracy: " + str(correct / (correct + incorrect)))

test_house_data()
test_iris_data()
>>>>>>> 4e164422d5a1105cc98ae84b718055abd039720c
