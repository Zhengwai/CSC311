from utils import *

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    log_lklihood = 0.
    users = data['user_id']
    ques = data['question_id']
    results = data['is_correct']
    for i in range(len(users)):
        t = theta[users[i]]
        b = beta[ques[i]]
        log_lklihood += results[i]*(t-b)-np.log(1+np.exp(t-b))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    dtheta = np.zeros(len(theta))
    dbeta = np.zeros(len(beta))
    users = data['user_id']
    ques = data['question_id']
    results = data['is_correct']
    for i in range(len(users)):
        t = theta[users[i]]
        b = beta[ques[i]]
        dtheta[users[i]] += results[i] - sigmoid(t-b)
        dbeta[ques[i]] -= results[i] - sigmoid(t - b)
    theta += lr*dtheta
    beta += lr*dbeta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    theta = np.ones(max(data['user_id'])+1)
    beta = np.ones(max(data['question_id'])+1)
    neg_lld_lst = []
    neg_lld_val_lst = []
    val_acc_lst = []

    for i in range(iterations):
        # neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        # neg_lld_lst.append(neg_lld)
        # neg_lld_val = neg_log_likelihood(val_data, theta=theta, beta=beta)
        # neg_lld_val_lst.append(neg_lld_val)
        #score = evaluate(data=val_data, theta=theta, beta=beta)
        #val_acc_lst.append(score)
        #print("Iterations: {} \t Train NLLK: {} \t Validation NLLK: {} \t Score on validation data: {}".format(i, neg_lld, neg_lld_val, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, neg_lld_lst, neg_lld_val_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iter_num = 70
    lr = 0.002
    theta, beta, val_acc_lst, neg_lld_lst, neg_lld_val_lst = irt(train_data, val_data, lr, iter_num)
    plt.plot(np.arange(iter_num) + 1, neg_lld_lst, label='train negative log likelihood')
    plt.xlabel("iteration #")
    plt.ylabel("negative_log_likelihood")
    plt.legend()
    plt.show()
    plt.close()
    plt.plot(np.arange(iter_num) + 1, neg_lld_val_lst, label='validation negative log likelihood')
    plt.xlabel("iteration #")
    plt.ylabel("negative_log_likelihood")
    plt.legend()
    plt.show()
    plt.close()
    print("validation accuracy:" + str(evaluate(data=val_data, theta=theta, beta=beta)))
    print("test accuracy:" + str(evaluate(data=test_data, theta=theta, beta=beta)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j_values = (600, 1000, 1300)
    x = [[], [], []]
    y = [[], [], []]
    users = train_data['user_id']
    ques = np.array(train_data['question_id'])
    for i in range(3):
        j = j_values[i]
        indices = np.where(ques == j)[0]
        for index in indices:
            t = theta[users[index]]
            b = beta[j]
            x[i].append(t)
            y[i].append(sigmoid(t-b))
        plt.plot(np.sort(np.array(x[i])), np.sort(np.array(y[i])), label="j=" + str(j), markersize=10)
    plt.xlabel('theta value')
    plt.ylabel('p(c=1|theta, beta)')
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()