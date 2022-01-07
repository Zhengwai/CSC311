from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    results = train_data["is_correct"]
    users = train_data["user_id"]
    ques = train_data["question_id"]
    du = np.zeros(u.shape)
    dz = np.zeros(z.shape)
    for i in range(len(users)):
        user_id = users[i]
        q_id = ques[i]
        c = results[i]
        du[user_id] -= (c - u[user_id].T@z[q_id])*z[q_id]
    for k in range(len(users)):
        user_id = users[k]
        q_id = ques[k]
        c = results[k]
        dz[q_id] -= (c - u[user_id].T@z[q_id])*u[user_id]
    u -= lr*du
    z -= lr*dz
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, val_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    train_sel = []
    val_sel = []
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)
        # train_sel.append(squared_error_loss(train_data, u, z))
        # val_sel.append(squared_error_loss(val_data, u, z))
    mat = u@z.T
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, train_sel, val_sel


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    val = []
    k_values = [8, 18, 28, 38, 58]
    for k in k_values:
        re_matrix = svd_reconstruct(train_matrix, k)
        val.append(sparse_matrix_evaluate(val_data, re_matrix))
        print(val[-1])
    k = k_values[val.index(max(val))]
    print("k: {} \t validation accuracy: {} \t test accuracy: {}".format(k, max(val), sparse_matrix_evaluate(test_data, svd_reconstruct(train_matrix, k))))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    lr = 0.003
    num_iteration = 59
    val = []
    k_values = [38,48,58,68,78]
    for k in k_values:
        re_matrix, train_sel, val_sel = als(train_data, val_data, k, lr, num_iteration)
        val.append(sparse_matrix_evaluate(val_data, re_matrix))
        print(val[-1])
    k = k_values[val.index(max(val))]
    print("k: {} \t validation accuracy: {} \t test accuracy: {}".format(k, max(val), sparse_matrix_evaluate(test_data, als(train_data, val_data, k, lr, num_iteration)[0])))
    # re_matrix, train_sel, val_sel = als(train_data, val_data, k, lr, num_iteration)
    # print(sparse_matrix_evaluate(val_data, re_matrix))
    # print(sparse_matrix_evaluate(test_data, re_matrix))
    # x = np.arange(num_iteration)+1
    # plt.plot(x, train_sel, label='training square error loss')
    # plt.xlabel("Iteration #")
    # plt.ylabel("square error loss")
    # plt.legend()
    # plt.show()
    # plt.close()
    # plt.plot(x, val_sel, label='validation square error loss')
    # plt.xlabel("Iteration #")
    # plt.ylabel("square error loss")
    # plt.legend()
    # plt.show()

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
