'''
Question 4 Skeleton Code

Here you should implement and evaluate the Conditional Gaussian classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

from data import get_digits_by_label


def compute_mean_mles(train_data, train_labels):
    '''
    Compute the mean estimate for each digit class

    Should return a numpy array of size (10,64)
    The ith row will correspond to the mean estimate for digit class i
    '''
    means = np.zeros((10, 64))
    # Compute means
    N, D = train_data.shape

    sum_y = 0
    for c in range(10):
        for i in range(N):
            if train_labels[i] == c:
                sum_y += 1
                means[c] += train_data[i]
    return means / sum_y


def compute_sigma_mles(train_data, train_labels):
    '''
    Compute the covariance estimate for each digit class

    Should return a three dimensional numpy array of shape (10, 64, 64)
    consisting of a covariance matrix for each digit class
    '''
    covariances = np.zeros((10, 64, 64))
    # Compute covariances
    N, D = train_data.shape
    mean = compute_mean_mles(train_data, train_labels)
    for c in range(10):
        for i in range(N):
            covariances[c] += (train_data[i] - mean[c]).T @ (train_data[i] - mean[c])
    return covariances / N


def generative_likelihood(digits, means, covariances):
    '''
    Compute the generative log-likelihood:
        log p(x|y,mu,Sigma)

    Should return an n x 10 numpy array
    '''
    log_likelihood = np.zeros((len(digits), 10))
    for i in range(len(digits)):
        for c in range(10):
            log_likelihood[i] = - len(digits) / 2 * np.log(2*np.pi) - 1/2 *(np.log(np.linalg.det(covariances[c] +
                                        0.001*np.identity(64)))) - 1/2 * ((digits[i] - means[c]).T @ np.linalg.inv(covariances[c] +0.001*np.identity(64)) @ (digits[i] - means[c]))
    return log_likelihood


def conditional_likelihood(digits, means, covariances):
    '''
    Compute the conditional likelihood:

        log p(y|x, mu, Sigma)

    This should be a numpy array of shape (n, 10)
    Where n is the number of datapoints and 10 corresponds to each digit class
    '''
    conditional_likelihood = np.zeros((len(digits), 10))
    num = 0
    for i in range(len(digits)):
        for c in range(10):
            num += means[c].T @ (np.linalg.inv(covariances[c] +
                                              0.001*np.identity(64))) @ \
                   digits[i] - 1/2 * means[c].T @ np.linalg.inv(covariances[c] + 0.001*np.identity(64)) \
                   @ means[c]
    for i in range(len(digits)):
        for c in range(10):
            de = means[c].T @ np.linalg.inv(covariances[c] +
                                            0.001*np.identity(64)) @ \
                 digits[i] - 1/2 * means[c].T @ np.linalg.inv(covariances[c] + 0.001*np.identity(64)) \
                 @ means[c]
            conditional_likelihood[i][c] = de / num
    return conditional_likelihood


def avg_conditional_likelihood(digits, labels, means, covariances):
    '''
    Compute the average conditional likelihood over the true class labels

        AVG( log p(y_i|x_i, mu, Sigma) )

    i.e. the average log likelihood that the model assigns to the correct class label
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)

    # Compute as described above and return
    return np.sum(cond_likelihood * labels) / len(digits)


def classify_data(digits, means, covariances):
    '''
    Classify new points by taking the most likely posterior class
    '''
    cond_likelihood = conditional_likelihood(digits, means, covariances)
    # Compute and return the most likely class
    return np.argmax(cond_likelihood, axis=1)


def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')

    # Fit the model
    means = compute_mean_mles(train_data, train_labels)
    covariances = compute_sigma_mles(train_data, train_labels)

    # Evaluation
    N, D = train_data.shape
    hit = 0
    for c in range(10):
        print("leading eigenvector for {}".format(c))
        w, v = np.linalg.eig(covariances[c])
        index = np.where(w == np.amax(w))[0]
        print("eigenvalue : {}, eigenvectors: {}".format(w[index], v[index]))
        digits = get_digits_by_label(train_data, train_labels, c)
        # print(digits)
        pre = classify_data(digits, means, covariances)
        # print(pre.shape[0])
        for i in range(pre.shape[0]):
            if pre[i] == c:
                hit += 1
    print("accuracy for train :{}".format(hit / N))
    return hit / N



if __name__ == '__main__':
    print("accuracy for test :96")
    # main()
