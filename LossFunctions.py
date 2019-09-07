import math as m
import numpy as np

def cross_entropy_loss(amt_of_classes, truth, predicted): #Taking in arrays of probabilities for each class?
    """Loss Function algorithm that determines our algorithms performs on the data-set provided."""
    # Takes in 2 distributions, the true distribution p(x) and the estimated distribution q(x), defined
    # over the discrete (cannot be infinite) variable x.
    # We reward/penalise probabilities of correct classes only.

    # Time needed to reach a certain level of growth ln(x) -- > time needed to grow to x

    # Think of it as amount of information you have to communicate, if more bits of information are needed
    # then the entropy is greater -- same with probability as if we had 8 fruits the binary encoding of those.
    # fruits would be log_2_(n), so 3 in this case.

    # This entropy tells us about the uncertainty involved with certain probability distributions, the more
    # uncertainty involded with certain probability distributions; the more uncertainty/variation in a probablity
    # distribution, the larger is the entropy (e.g for 1024 fruits it would be 10)

    # In cross-entropy, as the name suggests, we focus on the number of bits required to explain the differences
    # in 2 different probabilty distributions -- so best case is if they are identical, as the least amount of bits
    # are required. In mathmatical terms H(y,y^)=−∑i yiloge(y^i) where "y^" is the predicted probabilty vector
    # (softmax output) and y is the ground-truth vector (eg. one-hot). The reason we use natural log is because it
    # is easy to differentiate (ref. calculating gradients) and the reason we do not take log of the ground-truth
    # vector is because it contains a lot of 0's which simplify the summation.

    # In layman terms, one could think of cross-entropy as the distance between 2 probability distributions in
    # terms of the amount of information (bits) needed to explain that distance. It is a neat way of defining a
    # loss which goes down as the probabilty vectors get closer to one another.

    # 1. What is the probabiltity of the example being democrat?
    # 2. 1 if it is democrat 0 if it is republican
    # 3. Given 17 features we need to predict it's label
    # 4. Fit a model to perform this classification -- it will predict a probability of being democrat.
    # 5. How good or bad are our predicted probabilities ---LOSS function comes into play

    # predicted -- is the predicted probability our model outputted that the label will be democrat
    # truth -- is the actual label for the point (1 if democrat and 0 if republican)
    # For each democrat point we add log probability of it being democrat. For each republican point
    # we add log probability of it being republican.
    # Get mean of all of these losses.

    def calculate_loss_foreach_class(truth, predicted):
        return ((-1) * (truth * m.log(predicted)))


    if amt_of_classes > 2:
        # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
        # calculate a separate loss for each class label per observation and sum the result
        # H(p, q) =−∑∀x p(x)log(q(x))
        # p_o_c predicted probability observation o is of class c
        # y_o_c binary indicator if class label c is the correct classification for observation o

        total_sum = 0
        for i in range(len(predicted)-1):
            total_sum += calculate_loss_foreach_class(truth[i], predicted[i])

    else:
        # For binary class labels
        amt_of_useful_info =  (-1)*(truth * m.log(predicted) + (1-truth) * m.log(1-predicted))
    return amt_of_useful_info


def hinge_loss(amt_of_classes, truth, predicted):
    # https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy
    return np.max(0, (1-predicted) * truth)