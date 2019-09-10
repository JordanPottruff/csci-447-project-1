import math as m
import numpy as np


def cross_entropy_loss(element):  # Taking in arrays of probabilities for each element
    """Loss Function algorithm that determines our algorithms performs on the data-set provided."""

    # We are taking in the truth and predicted values for each element in our test set and going to calculate our loss.

    def calculate_loss(truth, predicted):
        return (-1) * (truth * m.log(predicted))

    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    # calculate a separate loss for each class label per observation and sum the result
    # H(p, q) =−∑∀x p(x)log(q(x))
    # p_o_c predicted probability observation o is of class c
    # y_o_c binary indicator if class label c is the correct classification for observation o
    # Sum over binary cross-entropies for individual points in a dataset
    num_of_elements = len(truth)  # Number of elements in our test set
    total_element_loss = 0
    for i in range(num_of_elements - 1):
        total_element_loss += calculate_loss(truth[i], predicted[i])

    return performance


def mean_square_error(example_data):
    # Num of classes is the number of keys -- Jordan will pass predicted value in as a map
    # Mean Square Error (MSE) loss function that will 

    for eachExample in example_data:
        extractData(eachExample);





    # predicted = (Actual class, {Key=Classname : Value=(Probability, Classname: Value)})
    num_of_elements = len(truth)  # Number of elements in our test set
    total_element_loss = 0
    for i in range(num_of_elements - 1):
        total_element_loss += (predicted - true) ** 2  # Calculate the squared error for all the instances
    mean_element_loss = total_element_loss / num_of_elements  # Get average loss for each instance
    return mean_element_loss

def extractData(eachExample):
    """Extracts the truth value, the estimated probability value, the classname, and the number of classes"""
    for truth, probabilities in eachExample:
        current_highest_probability = 0
        for individual_probability in probabilities:
            # We are just going to ignore if they are the same probability

            if individual_probability > current_highest_probability:
                current_highest_probaility = individual_probability
