import numpy as np

# All functions apply normalization method: (X - XMin) / (XMax - XMin)

def normalize(X):
    XNorm = np.zeros((X.shape[0], X.shape[1]))
    featMax = []
    featMin = []
    for i in range(X.shape[1]):
        featMax.append(X[:, i].max())
        featMin.append(X[:, i].min())
        for j in range(X.shape[0]):
            XNorm[j, i] = (X[j, i] - featMin[i])/(featMax[i] - featMin[i])
    return XNorm


# 1. sigmoidLikelihood
def sigmoidFunction(w, x):
    """
    :param w: m+1 dimension np-array; hyper-plane parameters
    :param x: m+1 dimension np-array; m features as the first m elements and 1 as the last element
    :return: result of sigmoid function
    """
    sigmoid = 1 / (1 + np.exp(- np.dot(w, x)))
    return sigmoid

def sigmoidLikelihood(X, y, w):
    """
    :param X: (n, m) dimension np-array with n samples and m features
    :param y: n dimension np-array; class label
    :param w: m+1 dimension np-array; hyper-plane parameters
    :return: n dimension np-array; sigmoid-based pseudo-likelihood of each sample
    """
    Xnorm = normalize(X)
    oneVec = np.ones(len(Xnorm))
    XNew = np.vstack((Xnorm.T, oneVec)).T
    LVector = np.zeros(len(Xnorm))
    for i in range(len(Xnorm)):
        LVector[i] = ((1 - sigmoidFunction(w, XNew[i])) ** (1-y[i])) * (sigmoidFunction(w, XNew[i]) ** y[i])
    return LVector


# 2.

# (a) check how many elements in LVector which contains all values of 0.05 result the pseudo-likelihood as 0

# LVector = np.array([0.05])
# prod = np.prod(LVector)
# i = 1
# while prod != 0:
#     LVector = np.append(LVector, 0.05)
#     prod = np.prod(LVector)
#     i += 1
# print(i)

# 249 data points are needed for np.prod to estimate the pseudo-likelihood as perfectly 0.

# (b) pseudo-log-likelihood

# LVector = np.array([0.05]*249)
# ll = np.sum(np.log(LVector))
# print(ll)

# The pseudo-log-likelihood is -745.9373361149437.


# 3. learnLogistic

def learnLogistic(w0, X, y, K):
    """
    :param w0: m+1 dimension np-array; initial hyper-plane parameters
    :param X: (n, m) dimension np-array with n samples and m features
    :param y: n dimension np-array; class label
    :param K: the number of learning loops
    :return: wNew: m+1 dimension np-array; new hyper-plane parameters
             LHistory: K dimension np-array; log-likelihood after each loop
    """
    XNorm = normalize(X)
    oneVec = np.ones(len(XNorm))
    XNewNorm = np.vstack((XNorm.T, oneVec)).T

    LHistory = np.zeros(K)
    wUpdate = np.ones(len(w0))
    wNew = np.array(w0)

    stepSize = 0.01

    for k in range(K):  # loop K times
        for i in range(len(XNorm)):     # loop all data points
            for j in range(len(wNew)):      # loop all features
                wUpdate[j] += stepSize * XNewNorm[i, j] * (y[i] - sigmoidFunction(wNew, XNewNorm[i]))

        for j in range(len(wNew)):
            wNew[j] += wUpdate[j]

        LHistory[k] = np.sum(np.log(sigmoidLikelihood(XNorm, y, wNew)))

    return wNew, LHistory


# 4. Update all w one time

# (a) learnLogisticFast
def learnLogisticFast(w0, X, y, K):
    """
    :param w0: m+1 dimension np-array; initial hyper-plane parameters
    :param X: (n, m) dimension np-array with n samples and m features
    :param y: n dimension np-array; class label
    :param K: the number of learning loops
    :return: wNew: m+1 dimension np-array; new hyper-plane parameters
             LHistory: K dimension np-array; log-likelihood after each loop
    """
    XNorm = normalize(X)
    oneVec = np.ones(len(XNorm))
    XNormNew = np.vstack((XNorm.T, oneVec)).T

    LHistory = np.zeros(K)
    wUpdate = np.ones(len(w0))
    wNew = np.array(w0)

    stepSize = 0.01

    for k in range(K):
        for i in range(len(XNorm)):
            wUpdate += stepSize * np.dot(XNormNew[i], y[i] - sigmoidFunction(wNew, XNormNew[i]))    # update all w
        wNew += wUpdate

        LHistory[k] = np.sum(np.log(sigmoidLikelihood(XNorm, y, wNew)))

    return wNew, LHistory


# (b) compare the speeds of learnLogistic and learnLogisticFast

# import time
# import scipy.io
#
# # import and processing data
# data = scipy.io.loadmat('hw2data.mat')['fullData']
# X = data[:, :10]
# y = data[:, 10]
# w0 = np.zeros(X.shape[1]+1)
# K = 20
#
# timeStart = time.time()
# w1, LHistory1 = learnLogistic(w0, X, y, K)
# timeEnd = time.time()
# print(timeEnd - timeStart)
#
# # learnLogistic uses 2.655320167541504 seconds.
#
# timeStart = time.time()
# w2, LHistory2 = learnLogisticFast(w0, X, y, K)
# timeEnd = time.time()
# print(timeEnd - timeStart)
#
# # learnLogisticFast uses 0.9466369152069092 second.


# 5. logisticClassify

def logisticClassify(X, w):
    """
    :param X: (n, m) dimension np-array with n samples and m features
    :param w: m+1 dimension np-array; final hyper-plane parameters
    :return: n dimension np-array; 0/1 label for each data points
    """

    XNorm = normalize(X)
    classLabels = np.zeros(len(XNorm))

    oneVec = np.ones(len(XNorm))
    XNormNew = np.vstack((XNorm.T, oneVec)).T

    for i in range(len(XNormNew)):
        if np.dot(w, XNormNew[i]) > 0:
            classLabels[i] = 1

    return classLabels
