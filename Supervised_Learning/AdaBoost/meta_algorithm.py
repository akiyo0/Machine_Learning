import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def stumpClassify(dataMat, column, threshIneq, threshold):
    array = np.ones((np.shape(dataMat)[0], 1))
    #print(array.shape)
    if threshIneq == 'lt':
        array[dataMat[:, column] <= threshold] = -1.0 #小于阈值时为-1
    else:
        array[dataMat[:, column] > threshold] = -1.0 #大于阈值时为-1
    return array

def buildStump(dataMat, labels, D):
    labels = labels.T.reshape(-1, 1)
    #print(labels)
    
    ddim_1, ddim_2 = np.shape(dataMat)
    numSteps = 10.0; bestStump = {}; bestClasEst = np.zeros((ddim_1, 1))
    minError = float('inf')

    for i in range(ddim_2): #遍历所有特征
        dmin, dmax = dataMat[:, i].min(), dataMat[:, i].max()
        #print(dmin, dmax) #当前特征的最大最小值
        stepSize = (dmax - dmin) / numSteps
        #print("stepSize", stepSize)

        for j in range(-1, int(numSteps) + 1):
            #大于和小于的情况均遍历。lt:less_than/gt:greater_than
            for inequal in ['lt', 'gt']:
                threshold = (dmin + float(j) * stepSize)
                #print(i, threshold)
                predicted = stumpClassify(dataMat, i, inequal, threshold)

                errArr = np.ones((ddim_1, 1))
                errArr[predicted == labels] = 0

                weightedError = np.dot(D.T, errArr)
                print("weightedError", weightedError[0][0])

                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predicted.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['inequal'] = inequal

    return bestStump, minError, bestClasEst                

def main():
    dataMat = pd.DataFrame([
        [1., 2.1], [1.5, 1.6], [1.3, 1.], [1., 1.], [2., 1.]
    ], columns = ['X', 'Y'])

    labels = [1.0, 1.0, -1.0, -1.0, 1.0]


    sns.set()

    ax = sns.scatterplot(x='X', y='Y', data=dataMat, hue=labels, palette="Paired")
    ax.set(xlim=(0.8, None))
    ax.set(ylim=(0.8, None))
    plt.show()


    D = np.ones((5, 1)) / 5
    dataMat = np.array(dataMat)
    labels = np.array(labels)
    a, b, c = buildStump(dataMat, labels, D)

    print(a)
    print(b)
    print(c)

if __name__ == "__name__":
    pdb.set_trace()
    main()
    