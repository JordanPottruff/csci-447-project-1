import math as m
import numpy as np


def cross_entropy_loss(training_set):  # Taking in each element in test set
    """Loss Function algorithm that determines our algorithms performs on the data-set provided."""

    # We are taking in an element and are going to provide the loss for this element using Cross Entropy.

    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    # calculate a separate loss for each class label per observation and sum the result
    # H(p, q) =−∑∀x p(x)log(q(x))

    # Cross Entropy function
    def calculate_loss(truth, predicted):
        if predicted == 0:
            print("Error: To accurate? Find me in loss functions as we cannot have a natural log of 0")
        return (-1) * (truth * m.log(predicted))

    total_loss = 0
    # Loop through each example in our test set
    for eachExample in training_set:
        # ... get needed variables from the data for our loss function ...
        actual_class, classes, probabilities, num_of_classes = extractData(eachExample)

        # ... keep track of total loss for each example ...
        example_loss = 0
        # ... Loop through classes
        for i in range(num_of_classes - 1):
            # ... get targets of what the probabilities should of been ...
            if actual_class != classes[i]:
                truth = 0
            else:
                truth = 1
            example_loss += calculate_loss(truth, probabilities[i])

        total_loss += example_loss
    loss_average_of_testset = total_loss/len(training_set)
    print("Our averaged loss value on this training set using cross entropy is [0]", format(loss_average_of_testset))


def mean_square_error(example_data):
    # Num of classes is the number of keys
    # Mean Square Error (MSE) loss function that will
    # https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
    # Mean square error function
    def calculate_loss(truth, predicted):
        return (predicted - truth)**2

    total_loss = 0
    # Loop through each example in our test set
    for eachExample in training_set:
        # ... get needed variables from the data for our loss function ...
        actual_class, classes, probabilities, num_of_classes = extractData(eachExample)

        # ... keep track of total loss for each example ...
        example_loss = 0
        # ... Loop through classes
        for i in range(num_of_classes - 1):
            # ... get targets of what the probabilities should of been ...
            if actual_class != classes[i]:
                truth = 0
            else:
                truth = 1
            example_loss += calculate_loss(truth, probabilities[i])
        total_loss += example_loss
    loss_average_of_testset = total_loss / len(training_set)
    print("Our averaged loss value on this training set using cross entropy is [0]", format(loss_average_of_testset))

def extractData(eachExample):
    """Extracts the truth value, the estimated probability value, the classname, and the number of classes"""
    # Format of data --> eachExample = (Actual class*, {Key=Classname : Value=(Probability, Classname: Value)})
    # From what I see it looks like it will be eachExample = (Actual Class, {classname: probability, classname2: probability2}
    # Declare Variables
    dictionary = eachExample[1]
    actual_class = eachExample[0]
    classes = []
    probabilities = []
    num_of_classes = 0

    error_check = 0
    for key in dictionary:

        # Get classes

        # Get probabilities associated with each class

        error_check += 1

    # Error checking
    if error_check > 1:
        print("Error, we have more then 1 key per example in the LossFunctions input")
    if len(probailities) != len(classes):
        print("Error, we should have equal number of probabilities and classes")

    num_of_classes = len(classes)
    return actual_class, classes, probabilities, num_of_classes

