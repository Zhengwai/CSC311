'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    for i in range(10):
        subset = data.get_digits_by_label(train_data, train_labels, i)
        means[i,] = np.mean(subset, axis=0)
    #print(means)
    return means

def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    for i in range(10):
        subset = data.get_digits_by_label(train_data, train_labels, i)
        means = subset.mean(0)
        cov = np.zeros((64, 64))
        num = len(subset)
        for j in range(num):
            x_mu = subset[j, ] - means
            cov += np.dot(x_mu.reshape((64,1)), x_mu.reshape((1,64)))
        covariances[i, :, :] = cov / num
    return covariances

def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    n, d = digits.shape
    c1 = -d / 2 * np.log(2 * np.pi)
    llh = np.zeros((n, 10))
    for i in range(n):
        for j in range(10):
            covs = covariances[j, :, :]
            x_mu = digits[i,:] - means[j, :]
            c2 = - 0.5 * np.log(np.linalg.det(covs)) - 0.5 * (np.dot(x_mu.reshape(1, 64), np.dot(np.linalg.inv(covs), x_mu.reshape(64,1))))[0][0]
            llh[i][j] = c1 + c2
    return llh


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    cllh = np.zeros((len(digits), 10))
    gllh = generative_likelihood(digits, means, covariances)
    for i in range(len(digits)):
        llh = gllh[i, :]
        # marginalization for p(x|mu, sigma)
        p2 = llh[0]
        for k in range(1, 10):
            p2 = np.logaddexp(p2, llh[k])
        p2 += np.log(1/10)
        for j in range(10):
            cllh[i][j] = llh[j] + np.log(1 / 10) - p2
    return cllh

def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute as described above and return
    num = len(digits)
    sum = 0
    for i in range(num):
        sum += cond_likelihood[i][int(labels[i])]
    return sum / num


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def accuracy(preds,labels):
    return np.mean(preds == labels)


def plot_eigenvectors(cov):
    for i in range(10):
        eigen_values, eigen_vectors = np.linalg.eig(cov[i, : ])
        lead_ev = eigen_vectors[ : , np.argmax(eigen_values)]
        plt.subplot(2, 5, i + 1)
        plt.imshow(lead_ev.reshape((8,8)), cmap='gray')
    plt.show()

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    train_avg = avg_conditional_likelihood(train_data, train_labels, means, covariances)
    test_avg = avg_conditional_likelihood(test_data, test_labels, means, covariances)

    print('The average conditional log-likelihood on train set is {}.\n'.format(train_avg))

    print('The average conditional log-likelihood on test set is {}.\n'.format(test_avg))
    # Evaluation
    train_preds = classify_data(train_data, means, covariances)
    train_accuracy = accuracy(train_preds, train_labels)
    test_preds = classify_data(test_data, means, covariances)
    test_accuracy = accuracy(test_preds, test_labels)
    print('Training accuracy is {} \n'.format(train_accuracy))
    print('Test accuracy is {} \n'.format(test_accuracy))

    plot_eigenvectors(covariances)

if __name__ == '__main__':
    main()
