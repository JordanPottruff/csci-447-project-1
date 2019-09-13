# loss_function.py
#
# Defines the loss functions we use to evaluate the effectiveness of our classifier. We use mean square error and cross
# entropy error and discuss the findings in our report. These two methods are accessed through a common loss_function
# top-level function that requires the caller to specify which function they want to use.

import math as m


# The loss function method takes in the results of our Naive Bayes classifier and determines an error based on the
# difference between our class probabilities and the desired class probabilities. loss_function_type specifies whether
# mean square error (arg: 'MSE') should be used or cross entropy error (arg: 'Cross_Entropy').
def loss_function(classifier_results, loss_function_type):
    # The classifier results are a list of tuple, representing the result for a specific test example. The first index
    # is the name of the actual class of the test example. The second index is a map of each class name to the
    # probability that the test example belongs to that class according to the classifier.

    # We average our loss functions across all the test example results.
    total_loss = 0
    # Loop through each example in our test set
    for result in classifier_results:
        # ... get needed variables from the data for our loss function ...
        actual_class, classes, probabilities, num_of_classes = extract_data(result)
        # ... keep track of total loss for each example ...
        example_loss = 0
        # ... Loop through classes
        for i in range(num_of_classes):
            # ... get targets of what the probabilities should of been ...
            truth = 0
            if actual_class != classes[i]:
                truth = 0
            else:
                truth = 1
            if loss_function_type == 'MSE':
                example_loss += calculate_mean_square_error(truth, probabilities[i], num_of_classes)
            elif loss_function_type == 'Cross_Entropy':
                example_loss += calculate_cross_entropy_error(truth, probabilities[i], num_of_classes)
            else:
                print("Error: There was no loss_function created for the requested type.")

        total_loss += example_loss
    loss_average_of_testset = total_loss / len(classifier_results)
    # print("Our averaged loss value on this training set using cross entropy is [0]", format(loss_average_of_testset))
    return loss_average_of_testset


# Function to calculate the mean square error of our prediction.
def calculate_mean_square_error(truth, predicted, num_of_classes):
    # Mean Square Error (MSE) loss function
    # https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
    # Mean square error function

    # We normalize the mean square error by the num_of_classes here as well.
    return (predicted - truth)**2 / num_of_classes


# Function to calculate the cross entropy error.
def calculate_cross_entropy_error(truth, predicted, num_of_classes):
    # Cross Entropy loss function
    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    # calculate a separate loss for each class label per observation and sum the result
    # H(p, q) =−∑∀x p(x)log(q(x))

    # We want to prevent a log(0) situation by adding some small amount of noise.
    if predicted == 0:
        predicted += .000001
    # We normalize the cross entropy error by the num_of_classes here as well.
    return (-1) * (truth * m.log(predicted))/num_of_classes


# Helper function that extracts all the pieces of our test example results into a more usable format.
def extract_data(test_result):
    # Format of test_result -> (Actual Class, {classname: probability, classname2: probability2})
    # Declare Variables
    probability_dictionary = test_result[1]
    actual_class = test_result[0]
    classes = []
    probabilities = []

    for cls in probability_dictionary:
        # Get classes
        classes.append(cls)
        # Get probabilities associated with each class
        probabilities.append(probability_dictionary[cls])

    # Error checking
    if len(probabilities) != len(classes):
        print("Error, we should have equal number of probabilities and classes")

    num_of_classes = len(classes)

    # Returns the individual components of the test result.
    return actual_class, classes, probabilities, num_of_classes

