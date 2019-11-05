import numpy as np

# Neural Networks
# 1. 3-Layer Neural Networks: Calculate outputs of each layer 
# n data points with k inputs; qn units in nth layer

def sigmoidMat(matrix):
  sigmoid = lambda x: 1 / (1 + np.exp(-x))
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      matrix[i][j] = sigmoid(matrix[i][j])
  return matrix


def feedforward(x, layer1W, layer2W, layer3W):
  """
  :param x: (n, k) np array
  :param layer1W: weights of the first layer; (q1, k+1) np array
  :param layer2W: weights of the second layer; (q2, q1+1) np array
  :param layer3W: weights of the third layer; (q3, q2+1) np array
  :return: outDict: a dictionary containing the outputs of each layer for each input
  """
  outDict = {}
  xNew = np.hstack((x, np.ones(np.shape(x)[0]).reshape(np.shape(x)[0], 1)))
  outDict['lay1'] = sigmoidMat(np.matmul(xNew, layer1W.T))
  layer2Input = np.hstack((outDict['lay1'], np.ones(np.shape(outDict['lay1'])[0]).reshape(np.shape(outDict['lay1'])[0], 1)))
  outDict['lay2'] = sigmoidMat(np.matmul(layer2Input, layer2W.T))
  layer3Input = np.hstack((outDict['lay2'], np.ones(np.shape(outDict['lay2'])[0]).reshape(np.shape(outDict['lay2'])[0], 1)))
  outDict['lay3'] = sigmoidMat(np.matmul(layer3Input, layer3W.T))

  return outDict
  
  
# Principal Component Analysis(PCA)

# Given function commonDirection
import numpy.linalg as LN

def commonDirection(allData):
  covMat = np.matmul(allData.T, allData)
  w, v = LN.eig(covMat)
  maxInd = np.argwhere(w == max(w))
  component = v[:, maxInd[0]].real
  return component

