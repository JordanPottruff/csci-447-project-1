import preprocessor as pr
import random
import naive_bayes as nb
import numpy as np
from math import floor


def cross_validation(data, folds=10):
    # size fold is just for testing purposes to make sure it splits the data correctly
    size_fold = floor(len(data)/folds)
    print(size_fold)
    # split the data into 10 evenly grouped sets as possible
    arr = np.array_split(data, folds)
    array = np.array(arr)
    print(array)

    for i in range(folds):
        training1 = array[i:folds]
        train = array[:i]
        training_set = np.append(training1, train, axis=0)
        testing_set = array[i]
        real_training_set = np.delete(training_set, 0, axis=0)
        print("Testing set %d: %s" % (i, testing_set))
        print("Training set %d: %s" % (i, real_training_set))


# bc_data = pr.open_breast_cancer_data()
# cross_validation(bc_data, 10)

# iris_data = pr.open_iris_data()
# cross_validation(iris_data, 10)

# glass_data = pr.open_glass_data()
# cross_validation(glass_data, 10)

# house_data = pr.open_house_votes_data()
# cross_validation(house_data, 10)

# soybean_data = pr.open_soybean_small()
# cross_validation(soybean_data, 10)

