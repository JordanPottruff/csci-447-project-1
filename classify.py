import naive_bayes as nb


def classify_house_data(training, test):
    classes = ['republican', 'democrat']
    attribute_cols = list(range(1, 17))
    class_col = 0

    results = []
    for line in test:
        result = {}
        for cls in classes:
            result[cls] = nb.calc_C(training, line, attribute_cols, class_col, cls)

    return results