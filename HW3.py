import numpy as np

# Neural Networks
# 1. 3-Layer Neural Networks: Calculate outputs of each layer 
# n data points with k inputs; qn units in nth layer

import numpy as np

def sigmoidMat(matrix):
  sigmoid = lambda x: 1 / (1 + np.exp(-x))
  for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
      matrix[i][j] = sigmoid(matrix[i][j])
  return matrix


def feedforward(x, layer1W, layer2W, layer3W):
  """
  :param x: (n, 4) np array
  :param layer1W: weights of the first layer; (2, 5) np array
  :param layer2W: weights of the second layer; (4, 3) np array
  :param layer3W: weights of the third layer; (1, 5) np array
  :return: outDict: a dictionary containing the outputs of each layer for each input
  """
  outDict = {}
  xNew = np.hstack((x, np.ones(np.shape(x)[0]).reshape(np.shape(x)[0], 1)))
  outDict['lay1'] = sigmoidMat(np.matmul(layer1W, xNew.T))
  layer2Input = np.vstack((outDict['lay1'], np.ones(np.shape(outDict['lay1'])[1])))
  outDict['lay2'] = sigmoidMat(np.matmul(layer2W, layer2Input))
  layer3Input = np.vstack((outDict['lay2'], np.ones(np.shape(outDict['lay2'])[1])))
  outDict['lay3'] = sigmoidMat(np.matmul(layer3W, layer3Input))

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


# Loading data
'''
import csv
import numpy
reader = csv.reader(open('mnist_train.csv', 'rb'), delimiter=',')
x=list(reader)
digits = numpy.array(x).astype('float')
'''

import pandas as pd
df = pd.read_csv('https://storm.cis.fordham.edu/leeds/cisc5800/mnist_train.csv', header=None)
allData = np.array(df)[:, 1:].astype('float')


# 2. Calculate a single z for one datapoints given one component (i.e. the datapoint's projection on the given component)

def dataProjection(dataVector, component):
  z = np.dot(component, dataVector)
  return z


# 3. Update one datapoint by removing the principal component

def componentRemoval(xOld, component, z):
  xNew = xOld - z * component
  return xNew

# 4. Find n principal components

def findComponents(allData, n):

  uMatrix = np.ones((n, allData.shape[1]))
  
  # set data to be zero-average for each feature
  for i in range(allData.shape[1]):
    allData[:, i] = allData[:, i] - np.mean(allData[:, i])

  dataPoints = allData
  for i in range(n):
    # find most common direction of variance  
    u = commonDirection(dataPoints).reshape(dataPoints.shape[1])
    uMatrix[i, :] = u / np.linalg.norm(u)
    zVector = np.array([dataProjection(dataPoints[m, :], uMatrix[i, :]) for m in range(dataPoints.shape[0])])
    for m in range(dataPoints.shape[0]):
      dataPoints[m, :] = componentRemoval(dataPoints[m, :], uMatrix[i, :], zVector[m])

  return uMatrix

# 5. visualize the first 3 components
comp3 = findComponents(allData, 3)
for i in range(comp3.shape[0]):
  plt.imshow(comp3[i, :].reshape((28, 28)))
  plt.show()
