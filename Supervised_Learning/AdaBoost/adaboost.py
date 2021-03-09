import numpy as np
import pandas as pd
import meta_algorithm

def adaboostTrainDS(dataMat, labels, iteration):
    weakClassGroup = []
    length = dataMat.shape[0]
    D = np.ones((length, 1)) / length
    aggClassEst = np.zeros((length, 1))

    for i in range(iteration):
        bestStump, error, classEst = meta_algorithm.buildStump(dataMat, labels, D)
        alpha = float(0.5 * np.log((1. - error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassGroup.append(bestStump)
        #print(bestStump)

        expon = np.multiply(-1 * alpha * labels.T.reshape(-1, 1), classEst)
        
        D = np.multiply(D, np.exp(expon))
        D = D/D.sum()

        aggClassEst += alpha * classEst
        aggErrors = np.multiply(np.sign(aggClassEst) != labels.T, np.ones((length, 1)))
        errorRate = aggErrors.sum()/length
        #print(errorRate)

        if errorRate == 0.0:
            break
    
    return weakClassGroup

dataMat = pd.DataFrame([
    [1., 2.1], [1.5, 1.6], [1.3, 1.], [1., 1.], [2., 1.]
], columns = ['X', 'Y'])

dataMat = np.array(dataMat)
labels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])

weakClassGroup = adaboostTrainDS(dataMat, labels, 40)
for i in weakClassGroup:
    print(i)