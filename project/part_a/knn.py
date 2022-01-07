from sklearn.impute import KNNImputer
from utils import *
from matplotlib import pyplot as plt

def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    #print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix.T)
    acc = sparse_matrix_evaluate(valid_data, mat.T)
    #print("Validation Accuracy: {}".format(acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc, mat.T


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)


    #####################################################################
    # TODO:                                                             #
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    for k in [i for i in range(30, 100)]:
        a, b = knn_impute_by_item(sparse_matrix, val_data, k)
        print(k)
        print(a)
        print(sparse_matrix_evaluate(test_data, b))
    k_values = [1, 6, 11, 16, 21, 26]
    user_acc = []
    item_acc = []
    for k in k_values:
        user_acc.append(knn_impute_by_user(sparse_matrix, val_data, k))
        item_acc.append(knn_impute_by_item(sparse_matrix, val_data, k)[0])
        print("mode: knn by user; k: " + str(k) + "; validation accuracy: " + str(user_acc[-1]))
        print("mode: knn by item; k: " + str(k) + "; validation accuracy: " + str(item_acc[-1]))
    max_user_acc = max(user_acc)
    max_item_acc = max(item_acc)
    k_star_user = k_values[user_acc.index(max_user_acc)]
    k_star_item = k_values[item_acc.index(max_item_acc)]
    print("k* for knn by user: " + str(k_star_user))
    print("mode: knn by user; k=k*; test accuracy: " + str(knn_impute_by_user(sparse_matrix, test_data, k_star_user)))
    print("k* for knn by item: " + str(k_star_item))
    print(
        "mode: knn by item; k=k*; test accuracy: " + str(knn_impute_by_item(sparse_matrix, test_data, k_star_item)[0]))
    plt.plot(np.array(k_values), np.array(user_acc), 'o-', label='knn by user')
    plt.plot(np.array(k_values), np.array(item_acc), 'o-', label='knn by item')
    plt.xlabel("k value")
    plt.ylabel("accuracy of prediction")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
