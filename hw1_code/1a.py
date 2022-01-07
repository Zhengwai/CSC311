import matplotlib.pyplot as plt
import numpy as np
import itertools

def rand_pts(dim, num):
    return np.random.rand(num, dim)
def euc_sqr_dist(p1, p2):
    return (np.linalg.norm(p1-p2))**2
def avg_dist(pts):
    i = 0
    sum = 0
    for comb in itertools.combinations(pts, 2):
        sum += euc_sqr_dist(comb[0], comb[1])
        i += 1
    return sum/i
def dist_array(pts):
    array = []
    for comb in itertools.combinations(pts, 2):
        array.append(euc_sqr_dist(comb[0], comb[1]))
    return np.array(array)


def std_dev(pts, mean):
    i = 0
    sum = 0
    for comb in itertools.combinations(pts, 2):
        dis = euc_sqr_dist(comb[0], comb[1])
        sum += (dis-mean)**2
        i += 1
    return sum/i

dims = []
pts_array = []
for i in range(11):
    dims.append(2**i)
    pts_array.append(rand_pts(2**i,100))

x = np.array([2**i for i in range(11)])
y = np.array([avg_dist(pts_array[i]) for i in range(11)])

z = np.array([dist_array(pts_array[i]).std() for i in range(11)])
print(y)

plt.plot(x, y, label="Average Distance", marker='o')
plt.plot(x, z, label="Standard Deviation", marker='o')
plt.xlabel("dimension")
plt.legend()
plt.show()