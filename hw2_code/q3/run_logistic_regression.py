from check_grad import check_grad
from utils import *
from logistic import *

import matplotlib.pyplot as plt
import numpy as np


def run_logistic_regression():
    #train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs, test_targets = load_test()
    N, M = train_inputs.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations, and the way in which you initialize the weights.   #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.1,
        #"weight_regularization": 0.,
        "num_iterations": 10000
    }
    weights = np.random.randn(M+1, 1)/1000
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # and compute test error.                                           #
    #####################################################################
    ces_train = []
    ces_valid = []
    ce_train, frac_train = None, None
    ce_valid, frac_valid = None, None
    ce_test, frac_test = None, None
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, train_inputs, train_targets, hyperparameters)
        y_valid = logistic_predict(weights, valid_inputs)
        y_test = logistic_predict(weights, test_inputs)
        ce_train, frac_train = evaluate(train_targets, y)
        ce_valid, frac_valid = evaluate(valid_targets, y_valid)
        ce_test, frac_test = evaluate(test_targets, y_test)
        weights -= hyperparameters['learning_rate'] * df / N
        ces_train.append(ce_train)
        ces_valid.append(ce_valid)
    print("Iteration: ", hyperparameters["num_iterations"], " ce_train: ", ce_train,
          " frac_train: ", frac_train, " ce_valid: ",ce_valid, "frac_valid: ", frac_valid,
          " ce_test: ", ce_test, " frac_test: ", frac_test )
    x = np.arange(0, hyperparameters['num_iterations'], 1)
    ces_train = np.asarray(ces_train)
    ces_valid = np.asarray(ces_valid)
    import pandas as pd
    pd.DataFrame(ces_train).to_csv("train_small.csv")
    pd.DataFrame(ces_valid).to_csv("valid_small.csv")
    plt.plot(x, ces_train)
    plt.plot(x, ces_valid)
    plt.xlabel("iteration")
    plt.ylabel("cross entropy")

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def run_check_grad(hyperparameters):
    """ Performs gradient check on logistic function.
    :return: None
    """
    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions + 1, 1)
    data = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,
                      weights,
                      0.001,
                      data,
                      targets,
                      hyperparameters)

    print("diff =", diff)


if __name__ == "__main__":
    run_logistic_regression()
    #run_pen_logistic_regression()
