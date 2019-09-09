import preprocessor as pr
import math
import random


def calc_instances_of_each_class(data):
    class_dictionary = {}
    for instance in data:
        if instance[-1] not in class_dictionary.keys():
            class_dictionary[instance[-1]] = 1
        else:
            class_dictionary[instance[-1]] += 1
    return class_dictionary


def training(data):
    training_set = []
    for i in range(0, math.floor(len(data) * 0.75)):
        training_set.append(data[i])
    return training_set


def testing(data):
    testing_set = []
    # for i in range(math.floor(len(data) / 2), len(data)):
    # for i in range(math.floor(len(data) / 2) + 1, math.floor(len(data) / 2) + 2):
    for i in range(math.floor(len(data) * 0.75), len(data)):
        testing_set.append(data[i])
    print("TESTING: " + str(testing_set))
    return testing_set


# Calculates Q (C = ci)
def calc_probability(set_training, set_full_data):
    total_number_in_training_set = sum(set_training.values())
    probability = {}
    for key, value in set_full_data.items():
        probability[key] = value / total_number_in_training_set
    # print("The occurence of each specified class divided by the total number of examples in training set")
    print(str(probability))
    return probability


def filter_by_class(data_set, class_name):
    classes = []
    for d in data_set:
        if d[-1] is class_name:
            classes.append(d)
    return classes


# Calculates F(Aj = ak, C = ci).
def calc_attribute_probability(class_col, class_name, num_attribute, training_data):
    dict_attribute = {}
    class_array = filter_by_class(training_data, class_name)
    denominator_total_examples_and_attributes = len(class_array) + num_attribute + 1

    for attribute in class_array:
        if attribute[class_col] not in dict_attribute:
            dict_attribute[attribute[class_col]] = 1
        else:
            dict_attribute[attribute[class_col]] = dict_attribute[attribute[class_col]] + 1

    # Add one to each value
    for key, value in dict_attribute.items():
        d = {key: value + 1}
        dict_attribute.update(d)

    for key, value in dict_attribute.items():
        d = {key: value / denominator_total_examples_and_attributes}
        dict_attribute.update(d)

    # print("The occurence of each attribute belonging to the specified class (" + class_name + ") + 1 / total number of examples in the class + the number of attributes")
    print(str(dict_attribute))
    return dict_attribute


def classify_data(testing_set, c1_attribute_probability, c2_attribute_probability):
    class1 = 1
    class2 = 1
    check = []

    for testing_data in testing_set:
        example = testing_data[1:10]
        for num in range(0, len(example)):
            try:
                class1 *= c1_attribute_probability[num][example[num]]
                class2 *= c2_attribute_probability[num][example[num]]
            finally:
                "KeyError"

            if class1 > class2:
                check.append("2")
            elif class2 < class1:
                check.append("4")
            print(check)

    """
    for testing_data in testing_set:
        example = testing_data[1:10]

    for testing_data in testing_set:
        example = testing_data[1:10]
        for test in example:
            for dict in c1_attribute_probability:
                for key, value in dict.items():
                    if test == key:
                        c.append([key, value])
                        num = num + value
        print("Class 1: " + str(num))
    """


def test_data(data):
    training_set = training(data)
    testing_set = testing(data)

    full_dataset_dictionary = calc_instances_of_each_class(data)
    training_set_dictionary = calc_instances_of_each_class(training_set)

    """
    print("Full Data: " + str(full_dataset_dictionary))
    print("Number in Examples Total: " + str(sum(full_dataset_dictionary.values())))
    print("Number in Training Set: " + str(sum(training_set_dictionary.values())))
    calc_probability(training_set_dictionary, full_dataset_dictionary)
    """
    # args: class_col, class_name, num_attributes, data (training set)

    c1_prob = []
    c2_prob = []
    for i in range(1, 10):
        print("Attribute: " + str(i))
        c1_prob.append(calc_attribute_probability(i, "2", 9, training_set))
        c2_prob.append(calc_attribute_probability(i, "4", 9, training_set))

    classify_data(testing_set, c1_prob, c2_prob)


data = pr.open_breast_cancer_data()
test_data(data)







