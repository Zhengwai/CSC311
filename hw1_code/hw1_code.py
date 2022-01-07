from sklearn.feature_extraction.text import CountVectorizer
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def load_data():
    data_real = open("clean_real.txt", 'r').read().split('\n')[:-1]
    data_fake = open("clean_fake.txt", 'r').read().split('\n')[:-1]
    data = data_real + data_fake
    outputs = np.array([True]*len(data_real) + [False]*len(data_fake))
    vec = CountVectorizer()
    data_train, x_test, outputs_train, y_test = train_test_split(vec.fit_transform(data).toarray(), outputs,
                                                                          test_size=0.3, random_state=1)
    names = vec.get_feature_names()
    data_test, data_valid, outputs_test, outputs_valid = train_test_split(x_test, y_test, test_size=0.5, random_state=2)
    dataset = [data_train, data_valid, data_test]

    outputset = [outputs_train, outputs_valid, outputs_test]
    return dataset, outputset, names


def select_model(data_train, outputs_train, data_valid, outputs_valid):
    max_depth_choices = [3, 5, 7, 9, 11, 13, 15]
    modes = ["gini", "entropy"]
    for mode in modes:
        print("=======================================\n" + "mode: "+str(mode))
        for depth in max_depth_choices:
            clf = DecisionTreeClassifier(criterion=mode,
                                         random_state=100, max_depth=depth)
            clf.fit(data_train, outputs_train)
            predictions = clf.predict(data_valid)
            print("depth: "+str(depth)+", accuracy: "+ str(accuracy_score(outputs_valid, predictions)))


def max_tree(data_train, outputs_train, data_valid, outputs_valid):
    clf = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=15)
    clf.fit(data_train, outputs_train)
    export_graphviz(clf, out_file="tree.txt", feature_names=names, max_depth=2)


def compute_information_gain(data_train, output_train, names):
    keywords = ["donald", "the", "hillary","turnbull", "and", "trump"]
    real_len = len(np.where(output_train==True)[0])
    train_size = len(output_train)
    fake_len = train_size-real_len
    h_root = -(real_len / train_size) * math.log(real_len / train_size, 2) - (fake_len / train_size) * math.log(
        fake_len / train_size, 2)
    for name in keywords:
        column = data_train[:, names.index(name)]
        left_indices = np.where(column <= 0.5)[0]
        right_indices = np.where(column > 0.5)[0]
        left_len = len(left_indices)
        right_len = len(right_indices)

        left_output = output_train[left_indices]
        right_output = output_train[right_indices]
        left_real = len(np.where(left_output==True)[0])
        left_fake = len(np.where(left_output==False)[0])
        right_real = len(np.where(right_output == True)[0])
        right_fake = len(np.where(right_output == False)[0])
        if left_real == 0 or left_fake == 0:
            h_left = 0
        else:
            h_left = (left_len / train_size) * (-(left_real / left_len) * math.log(left_real / left_len, 2) - (left_fake / left_len) * math.log(left_fake / left_len, 2))
        if right_real == 0 or right_fake ==0:
            h_right = 0
        else:
            h_right = (right_len / train_size) * (-(right_real / right_len) * math.log(right_real / right_len, 2) - (right_fake / right_len) * math.log(right_fake / right_len, 2))
        print("The information gain of a split at '"+str(name)+"' is", h_root - h_left - h_right)


dataset, outputset, names = load_data()
select_model(dataset[0], outputset[0], dataset[1], outputset[1])
max_tree(dataset[0], outputset[0], dataset[1], outputset[1])
compute_information_gain(dataset[0], outputset[0], names)
