import naive_bayes as nb


# Returns the results of classifying the test data using the specified training data. The result is a list of tuples
# that represents the results for each row in the test data. The first item in the tuple is the actual class for the
# observation, and the second item in the tuple is a map of each class to the calculated probability that the
# observation belongs to the class.
def classify_house_data(training, test):
    classes = ['republican', 'democrat']
    attribute_cols = list(range(1, 17))
    class_col = 0

    results = []
    for line in test:
        probabilities = {}
        for cls in classes:
            probabilities[cls] = nb.calc_C(training, line, attribute_cols, class_col, cls)

        results.append((line[class_col], probabilities))

    return results

