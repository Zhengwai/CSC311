# TODO: complete this file.
from knn import *
from item_response import *
from matrix_factorization import *
from sklearn.utils import resample
def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    n, m = train_matrix.shape
    data_lst = []
    indices_lst = []
    for i in range(2):
        d, indices = resample_matrix(train_matrix, i)
        data_lst.append(d)
        indices_lst.append(indices)
    d, indices = generate_last_resample(train_matrix, indices_lst, n, 2)
    data_lst.append(d)
    indices_lst.append(indices)
    print('resampling done!')

    theta1, beta1, val_acc_lst, neg_lld_lst, neg_lld_val_lst = irt(matrix_to_dict(data_lst[0]), val_data, 0.002, 70)
    ir_pred1 = np.zeros((n, m))
    for i in range(len(ir_pred1)):
        for j in range(len(ir_pred1[0])):
            ir_pred1[i][j] = sigmoid(theta1[i] - beta1[j])
    print("item response 1 done!")

    # theta2, beta2, val_acc_lst, neg_lld_lst, neg_lld_val_lst = irt(matrix_to_dict(data_lst[1]), val_data, 0.002, 70)
    # ir_pred2 = np.zeros((n, m))
    # for i in range(len(ir_pred2)):
    #     for j in range(len(ir_pred2[0])):
    #         ir_pred2[i][j] = sigmoid(theta2[i] - beta2[j])
    # print("item response 2 done!")

    # theta3, beta3, val_acc_lst, neg_lld_lst, neg_lld_val_lst = irt(matrix_to_dict(data_lst[2]), val_data, 0.002,
    #                                                               iter)
    # ir_pred3 = np.zeros((n, m))
    # for i in range(len(ir_pred2)):
    #     for j in range(len(ir_pred2[0])):
    #         ir_pred2[i][j] = sigmoid(theta2[i] - beta2[j])
    # print("item response 3 done!")
    m_pred1, train_sel, val_sel = als(matrix_to_dict(data_lst[1]), val_data, 78, 0.003, 59)
    print("matrix factorization 1 done!")
    m_pred2, train_sel, val_sel = als(matrix_to_dict(data_lst[2]), val_data, 78, 0.003, 59)
    print("matrix factorization 2 done!")

    # knn_pred1 = knn_impute_by_item(train_matrix, val_data, 21)[1]
    # print('knn done!')

    pred_matrices = [ir_pred1, m_pred1, m_pred2]
    pred = np.zeros((n, m))
    count = [0] * n
    for i in range(3):
        matrix = pred_matrices[i]
        indices = indices_lst[i]
        for j in range(n):
            pred[indices[j]] += matrix[j]
            count[indices[j]] += 1
    for i in range(n):
        pred[i] = pred[i] / count[i]
    print(sparse_matrix_evaluate(val_data, pred))
    print(sparse_matrix_evaluate(test_data, pred))

def generate_last_resample(matrix, ind_lst, n, rand_state):
    all = [i for i in range(n)]
    u = set()
    for indices in ind_lst:
        u = set.union(u, set(indices))
    elements = set(all).difference(u)
    ind = list(resample(np.array(all), n_samples=n-len(elements), random_state=rand_state))+list(elements)
    new_matrix = np.zeros(matrix.shape)
    for i in range(len(new_matrix)):
        new_matrix[i] = matrix[ind[i]]
    return new_matrix, ind

def resample_matrix(matrix, rand_state):
    indices = np.array([i for i in range(len(matrix))])
    u1 = resample(indices, random_state=rand_state)

    new_matrix = np.zeros(matrix.shape)
    for i in range(len(new_matrix)):
        new_matrix[i] = matrix[u1[i]]
    return new_matrix, list(u1)

def cover_all(indices_lst, n):
    u = set()
    for indices in indices_lst:
        u = u.union(set(indices))
    return len(u) == n

def matrix_to_dict(matrix):
    users = []
    ques = []
    answers = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if not np.isnan(matrix[i][j]):
                users.append(i)
                ques.append(j)
                answers.append(matrix[i][j])
    return {'user_id': users, 'question_id': ques, 'is_correct': answers}

if __name__ == '__main__':
    main()