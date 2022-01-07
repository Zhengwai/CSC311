from sklearn.impute import KNNImputer
from utils import *
import pandas as pd
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.utils import resample
from matplotlib import pyplot as plt

def knn_impute_by_item_new(ques_matrix, sub_matrix, k, b):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k, metric=dist2)
    matrix = np.concatenate((ques_matrix.T, sub_matrix*b), axis=1)
    mat = nbrs.fit_transform(matrix)
    print('hi')
    return mat.T[:542]


def str_to_list(string):
    string = string[1:-1].split(', ')
    l = [int(char) for char in string]
    return l

def lst_to_binary(lst, n):
    result = [0]*n
    for item in lst:
        result[item] = 1
    return result
def experiment(ques_matrix, sub_matrix, num):
    u_dist = []
    sub_dist = []
    ques = ques_matrix.T
    n = len(ques)
    indices = [i for i in range(n)]
    indices = resample(indices, n_samples=num, replace=False)
    for i in range(num):
        k = indices[i]
        for j in range(i+1, num):
            h = indices[j]
            u_dist.append(dist1(ques[k], ques[h]))
            sub_dist.append(np.linalg.norm(sub_matrix[k]-sub_matrix[h]))
    plt.plot(sub_dist, u_dist, '.')
    plt.xlabel('distance in subject dimensions')
    plt.ylabel('distance in user dimensions')
    plt.show()
    plt.close()
    dict = {}
    for i in range(len(sub_dist)):
        if not np.isnan(u_dist[i]):
            if sub_dist[i] in dict.keys():
                dict[sub_dist[i]][0] += 1
                dict[sub_dist[i]][1] += u_dist[i]
            else:
                dict[sub_dist[i]] = [1, u_dist[i]]
    x = []
    y = []
    for key in dict:
        x.append(key)
        y.append(dict[key][1]/dict[key][0])
    plt.plot(x, y, '.')
    plt.xlabel('distance in subject dimensions')
    plt.ylabel('average distance in user dimensions')
    plt.show()

def main():
    sparse_matrix = load_train_sparse("../starter_code/data").toarray()
    val_data = load_valid_csv("../starter_code/data")
    test_data = load_public_test_csv("../starter_code/data")

    ques_meta = pd.read_csv('../starter_code/data/question_meta.csv', sep=',').values
    ques_id = ques_meta[:,0]
    subjects = ques_meta[:,1:]
    subjects = [str_to_list(subjects[i][0]) for i in range(len(subjects))]

    n = 388
    m = sparse_matrix.shape[1]
    subject_matrix = np.zeros((m, n))
    for i in range(m):
        subject_matrix[ques_id[i]] = lst_to_binary(subjects[i], n)
    subject_matrix = np.array(subject_matrix)
    experiment(sparse_matrix, subject_matrix, 1742)

    # for k in [5*i+6 for i in range(15)]:
    #     for w in [0.1*i+0.1 for i in range(30)]:
    #         pred = knn_impute_by_item_new(sparse_matrix, subject_matrix, k, w)
    #         print(k, w)
    #         print(sparse_matrix_evaluate(val_data, pred))
    #         print(sparse_matrix_evaluate(test_data, pred))
    # matrix = np.concatenate((sparse_matrix.T, subject_matrix * 0), axis=1)
    # print(dist1(sparse_matrix.T[0], sparse_matrix.T[1]))
    # print(dist2(sparse_matrix.T[0], sparse_matrix.T[1]))

def dist(X,Y, **kws):
    a = nan_euclidean_distances(X[:542].reshape((1,542)), Y[:542].reshape((1,542)))[0][0]
    return a

def dist1(X,Y):
    a = nan_euclidean_distances(X.reshape((1, 542)), Y.reshape((1, 542)))[0][0]
    return a

def dist2(X,Y, **kws):
    x = X[:542]
    y = Y[:542]
    d = x-y
    n = np.count_nonzero(~np.isnan(d))
    d = np.nan_to_num(d)
    if n == 0:
        return np.nan
    return np.sqrt(542/n)*np.linalg.norm(d) + np.linalg.norm(X[542:]-Y[542:])

if __name__ == "__main__":
    main()
