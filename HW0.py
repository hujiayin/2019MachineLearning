import numpy as np

# 1. threshClassify
# Two thresholds to classify 

def threshClassify(featureArray, thresholds):
    classList = []
    for array in featureArray:
        if array[0] < thresholds[0]:
            if array[1] < thresholds[1]:
                classList.append(0)
            else:
                classList.append(1)
        else:
            if array[1] < thresholds[2]:
                classList.append(0)
            else:
                classList.append(1)
    return classList

featureArray = np.array([[2, 1], [4, 2], [8, 12], [6, 5], [3, 5]])
thresholds = np.array([5, 3, 9])
print(threshClassify(featureArray, thresholds))


# 2. findAccuracy

def findAccuracy(classifierOutput, trueLabels):
    num = len(classifierOutput)
    correct = 0
    for i in range(num):
        if classifierOutput[i] == trueLabels[i]:
            correct = correct + 1
    accuracy = correct/num
    return accuracy


classifierOutput = [1, 1, 1, 0, 1, 0, 1, 1]
trueLabels = [1, 1, 0, 0, 0, 0, 1, 1]
findAccuracy(classifierOutput, trueLabels)

# 3. fillBlanks

def fillBlanks(featArray):
    import numpy as np
    featArrayCorrected = np.where(featArray == 0, 5, featArray)
    return featArrayCorrected


featArray = np.array([[0, 5], [0, 3], [1, 8], [10, 0]])
featArrayCorrected = fillBlanks(featArray)
print(featArrayCorrected)

# 4. fillBlanksAve

def fillBlanksAve(featArray):
    import numpy as np
    index = np.where(featArray == 0)
    featArrayCorrected = featArray.copy().astype('float')
    for row, col in zip(*index):
        featArrayCorrected[row, col] = np.mean(featArrayCorrected[np.where(featArrayCorrected[:, col] != 0), col])
    return featArrayCorrected


featArray=np.array([[0,5],[0,3],[1,8],[10,0]])
featArrayCorrected = fillBlanksAve(featArray)
print(featArrayCorrected)



