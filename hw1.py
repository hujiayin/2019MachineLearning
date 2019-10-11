# 1. function learnMeans
import numpy as np

def learnMeans(inputList):
    meansOut = dict()
    inputArray1 = np.array(inputList[0])
    inputArray2 = np.array(inputList[1])

    for class_name in inputList[0]:
        if class_name in meansOut:
            continue
        else:
            index = np.where(inputArray1[:] == class_name)
            meansOut[class_name] = np.mean(inputArray2[index])

    return meansOut

# 2.  MAP learning of Gaussian means
# use Gaussian distribution as prior

# (a)
# MAP estimate mu = (sigGen^2 * sum(Xi) + sig^2 * muGen) / (n * muGen^2 + sig^2)

# b. function LearnMeanMAP


def learnMeanMAP(inputList, muGen, sigGen):
    meansOut = dict()
    inputArray1 = np.array(inputList[0])
    inputArray2 = np.array(inputList[1])
    sig = 2

    for class_name in inputList[0]:
        if class_name in meansOut:
            continue
        else:
            index = np.where(inputArray1[:] == class_name)
            meansOut[class_name] = (sigGen ** 2 * np.sum(inputArray2[index]) + sig ** 2 * \
                                    muGen) / (sigGen ** 2 * np.shape(inputArray2[index])[0] + sig ** 2)

    return meansOut


# 4. function learnPriors


def learnPriors(inputList):
    probDict = dict()
    for class_name in inputList[0]:
        if class_name in probDict:
            continue
        else:
            probDict[class_name] = inputList[0].count(class_name)

    for key in probDict:
        probDict[key] = probDict[key]/len(inputList[0])

    return probDict


# 5. function labelGB


def gaussDist(x, mu, sig):
    return 1/np.sqrt(2 * np.pi * (sig ** 2)) * np.power(np.e, -(((x - mu) ** 2) / (2 * (sig ** 2))))

def labelGB(amountAlcohol, meansDict, priorsDict):
    """
    :param amountAlcohol: a number
    :param meansDict: a dictionary containing the mean values for each classes (output by learnMeans)
    :param priorsDict:  a dictionary containing the shopper prior probabilities (output by learnPriors)
    :return: class label with highest probability
    """

    sig = 2
    probDict = dict()

    for key in priorsDict:
        probDict[key] = gaussDist(amountAlcohol, meansDict[key], sig) * priorsDict[key]

    shopper = max(probDict, key=probDict.get)

    return shopper


# 6. function evaluateGB


def evaluateGB(testData, meansDict, priorsDict):
    predList = []
    correctNum = 0
    totalNum = len(testData[0])

    for i in range(totalNum):
        amountAlcohol = testData[1][i]
        label = labelGB(amountAlcohol, meansDict, priorsDict)
        predList.append(label)
        if predList[i] == testData[0][i]:
            correctNum += 1

    return correctNum / totalNum

# 7. function crossValidOrder

def crossValidOrder(dataIn):
    k = 5
    num = len(dataIn[0])
    accuracyList = []
    meanListDict = {data: [] for data in dataIn[0]}
    priorListDict = {data: [] for data in dataIn[0]}

    for i in range(k):
        start = int(i * num / k)
        end = int((i + 1) * num / k)
        test = [dataIn[0][start: end],
                dataIn[1][start: end]]
        train = [dataIn[0][:start] + dataIn[0][end:],
                 dataIn[1][:start] + dataIn[1][end:]]

        meansDict = learnMeans(train)
        priorsDict = learnPriors(train)
        accuracyList.append(evaluateGB(test, meansDict, priorsDict))
        for key in meanListDict:
            if key in meansDict:
                meanListDict[key].append(meansDict[key])

        for key in priorListDict:
            if key in priorsDict:
                priorListDict[key].append(priorsDict[key])

    accuracy = np.mean(accuracyList)
    for key in meanListDict:
        meanListDict[key] = np.mean(meanListDict[key])
    for key in priorListDict:
        priorListDict[key] = np.mean(priorListDict[key])

    return accuracy, meanListDict, priorListDict


# 8. function crossValidStoch

def crossValidStoch(dataIn):

    randnum = np.random.randint(100)
    dataInShuffle = [dataIn[0][:], dataIn[1][:]]
    np.random.seed(randnum)
    dataInShuffle[0] = np.random.permutation(dataInShuffle[0]).tolist()
    np.random.seed(randnum)
    dataInShuffle[1] = np.random.permutation(dataInShuffle[1]).tolist()

    k = 5
    num = len(dataInShuffle[0])
    accuracyList = []
    meanListDict = {data: [] for data in dataIn[0]}
    priorListDict = {data: [] for data in dataIn[0]}

    for i in range(k):
        start = int(i * num / k)
        end = int((i + 1) * num / k)
        test = [dataInShuffle[0][start: end],
                dataInShuffle[1][start: end]]
        train = [dataInShuffle[0][:start] + dataInShuffle[0][end:],
                 dataInShuffle[1][:start] + dataInShuffle[1][end:]]

        meansDict = learnMeans(train)
        priorsDict = learnPriors(train)
        accuracyList.append(evaluateGB(test, meansDict, priorsDict))
        for key in meanListDict:
            if key in meansDict:
                meanListDict[key].append(meansDict[key])

        for key in priorListDict:
            if key in priorsDict:
                priorListDict[key].append(priorsDict[key])

    accuracy = np.mean(accuracyList)
    for key in meanListDict:
        meanListDict[key] = np.mean(meanListDict[key])
    for key in priorListDict:
        priorListDict[key] = np.mean(priorListDict[key])

    return accuracy, meanListDict, priorListDict


# 9. Comparison between crossValidOrder and crossValidStoch
# crossValidOrder gives a fixed mean, prior and accuracy due to the fixed training data and test data. The accuracy is 0.455.
# crossValidStoch gives different results each time. The accuracy is higher than that of crossValidOrder.



