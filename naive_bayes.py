

# Calculates C(x), or the probability that the specified observation belongs to the specified class based on the
# training data.
def calc_classification_probability(training_data, observation, attribute_cols, class_col, class_name):
    attribute_count = len(attribute_cols)
    product = 1
    # For each attribute...
    for col in attribute_cols:
        # ...find the probability of that attribute value for the given class and multiply it with our running product.
        observed = observation[col]
        product *= calc_attribute_probability(training_data, col, observed, class_col, class_name, attribute_count)
    # Finally, multiply the attribute probability product by the probability of the class occurring.
    return calc_class_probability(training_data, class_col, class_name) * product


# NOTE:
# The functions below are used for calculating the classification probability, and should not need to be referenced
# outside of this file.


# Calculates F(Aj = ak, C = ci), or (roughly) the probability of examples belonging to the specified class, with the
# specified attribute value for the specified attribute.
def calc_attribute_probability(training_data, attribute_col, attribute_value, class_col, class_name, num_attributes):
    attr_count = calc_attribute_count(training_data, attribute_col, attribute_value, class_col, class_name)
    class_n = calc_class_count(training_data, class_col, class_name)
    return (attr_count + 1) / (class_n + num_attributes)


# Calculates Q(C = ci), or the probability that an example will belong to the specified class.
def calc_class_probability(training_data, class_col, class_name):
    n = len(training_data)
    class_count = calc_class_count(training_data, class_col, class_name)
    # Essentially, just the proportion of examples belonging to the class.
    return class_count / n


# Calculates the number of examples in the data that belong to the specified class.
def calc_class_count(training_data, class_col, class_name):
    class_n = 0
    for line in training_data:
        # If the example matches the specified class, then increase our count by one.
        if line[class_col] == class_name:
            class_n += 1
    return class_n


# Calculates the number of examples in the data that belong to the specified class and have the specified attribute
# value for the specified attribute.
def calc_attribute_count(data, attribute_col, attribute_value, class_col, class_name):
    attribute_count = 0
    for line in data:
        # If the example matches the specified class and attribute value, then increase our count by one.
        if line[class_col] == class_name and line[attribute_col] == attribute_value:
            attribute_count += 1
    return attribute_count

