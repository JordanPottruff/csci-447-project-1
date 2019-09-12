import math as m


def loss_function(training_set, loss_function_type):
    """ Call the lossFunctions method when trying to find the loss of a model. The loss determines how well your model
        works.   """
    total_loss = 0
    # Loop through each example in our test set
    for eachExample in training_set:
        # ... get needed variables from the data for our loss function ...
        actual_class, classes, probabilities, num_of_classes = extract_data(eachExample)

        # ... keep track of total loss for each example ...
        example_loss = 0
        # ... Loop through classes
        for i in range(num_of_classes - 1):
            # ... get targets of what the probabilities should of been ...
            truth = 0
            if actual_class != classes[i]:
                truth = 0
            else:
                truth = 1

            if loss_function_type == 'MSE':
                print("Probability:" + str(probabilities[i]))
                print("Class:" + str(classes[i]))
                example_loss += calculate_mean_square_error(truth, probabilities[i], num_of_classes)
            elif loss_function_type == 'Cross_Entropy':
                print("Probability:" + str(probabilities[i]))
                print("Class:" + str(classes[i]))
                example_loss += calculate_cross_entropy_error(truth, probabilities[i], num_of_classes)
            else:
                print("Error: There was no loss_function created for the requested type.")

        total_loss += example_loss
    loss_average_of_testset = total_loss / len(training_set)
    print("Our averaged loss value on this training set using cross entropy is [0]", format(loss_average_of_testset))
    return loss_average_of_testset


def calculate_mean_square_error(truth, predicted, num_of_classes):
    """ Method to calculate the mean_square_error of a model.    """
    # Mean Square Error (MSE) loss function
    # https://jamesmccaffrey.wordpress.com/2013/11/05/why-you-should-use-cross-entropy-error-instead-of-classification-error-or-mean-squared-error-for-neural-network-classifier-training/
    # Mean square error function
    return ((predicted - truth)**2/num_of_classes)  # Return average loss for example


def calculate_cross_entropy_error(truth, predicted, num_of_classes):
    """ Loss Function algorithm that determines our algorithms performs on the data-set provided.   """
    # Cross Entropy loss function
    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    # calculate a separate loss for each class label per observation and sum the result
    # H(p, q) =−∑∀x p(x)log(q(x))
    if predicted == 0:
        predicted += .001
    return ((-1) * (truth * m.log(predicted))/num_of_classes)  # Return the average loss for the example


def extract_data(each_example):
    """Extracts the truth value, the estimated probability value, the classname, and the number of classes"""
    # Format of eachExample -> (Actual Class, {classname: probability, classname2: probability2})
    # Declare Variables
    probability_dictionary = each_example[1]
    actual_class = each_example[0]
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
    return actual_class, classes, probabilities, num_of_classes

