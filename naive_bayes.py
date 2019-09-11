

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
