import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

def stumpClassify(dataMat, column, threshIneq, threshold):
    result_array = np.ones((dataMat.shape[0], 1))
    #print(array.shape)
    if threshIneq == 'lt':
        result_array[dataMat[:, column] <= threshold] = -1.0 #小于阈值时为-1
        result_array[dataMat[:, column] > threshold] = 1.0 #小于阈值时为-1
    else:
        result_array[dataMat[:, column] >= threshold] = -1.0
        result_array[dataMat[:, column] < threshold] = -1.0 #大于阈值时为-1
    return result_array

def buildStump(X_train, labels, W):
    labels = labels.T.reshape(-1, 1)
    
    ddim_1, ddim_2 = X_train.shape
    BestStumpParams = {}; best_pred = np.zeros((ddim_1, 1))
    minError = float('inf')

    for i in range(ddim_2): #遍历所有特征
        dmin, dmax = X_train[:, i].min(), X_train[:, i].max() #当前特征的最大最小值
        #print(dmin, dmax) 

        for threshold in np.arange(dmin, dmax, 0.1):
            for inequal in ['lt', 'gt']: #大于和小于的情况均遍历。lt:less_than/gt:greater_than
                predicted = stumpClassify(X_train, i, inequal, threshold)

                loss = np.ones((ddim_1, 1))
                loss[predicted == labels] = 0

                this_weighted_loss = np.dot(W.T, loss)
                print("weightedError %.1f" % this_weighted_loss[0][0])

                if this_weighted_loss < minError:
                    minError = this_weighted_loss
                    best_pred = predicted.copy()
                    BestStumpParams['dim'] = i
                    BestStumpParams['threshold'] = threshold
                    BestStumpParams['inequal'] = inequal

    return BestStumpParams, loss, best_pred


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

if __name__ == "__main__":
    print("init")
    #pdb.set_trace()
    main()
