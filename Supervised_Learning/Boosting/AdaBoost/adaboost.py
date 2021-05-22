#import pdb
import numpy as np
import pandas as pd
import basic_algorithm

class Adaboost():
    def __init__(self, labels, iteration):
        self.labels = labels #标签
        self.iteration = iteration #迭代次数
        self.weakClassGroup = []

    def fit(self, X_data):
        length = X_data.shape[0]
        y_true = self.labels.T.reshape(-1, 1)

        W = np.ones((length, 1)) / length #初始化权重
        strongClass = np.zeros((length, 1)) #类别估计累计值

        for i in range(self.iteration):
            BestStumpParams, loss, y_pred = basic_algorithm.buildStump(X_data, y_true, W)
            alpha = float(0.5 * np.log((1. - loss) / max(loss, 1e-16))) #弱学习算法权重alpha
            BestStumpParams['alpha'] = alpha
            self.weakClassGroup.append(BestStumpParams)
            #print(BestStumpParams)
            
            exp = np.exp(np.multiply(-1 * alpha * y_true, y_pred))
            W = np.multiply(W, exp)
            W = W/W.sum()

            strongClass += alpha * y_pred
            em = np.multiply(np.sign(strongClass) != y_true, np.ones((length, 1)))
            lossRate = em.sum() / length

            if lossRate == 0.0:
                break

            return self.weakClassGroup
        
    def predict(self, X_train):
        length = X_train.shape[0]
        strongClass = np.zeros((length, 1))
        for i in self.weakClassGroup: #遍历分类器，进行分类
            y_pred = basic_algorithm.stumpClassify(X_train, i['dim'], i['inequal'], i['threshold'])			
            strongClass += i['alpha'] * y_pred
            print(strongClass)
        return np.sign(strongClass)

def main():
    dataMat = pd.DataFrame([
        [1., 2.1], [1.5, 1.6], [1.3, 1.], [1., 1.], [2., 1.]
    ], columns = ['X', 'Y'])

    dataMat = np.array(dataMat)
    labels = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    X_test = np.array([[0,0],[5,5]])
    trainer = Adaboost(labels, 40)
    
    weakClassGroup = trainer.fit(dataMat)
    print(trainer.predict(X_test))

    for i in weakClassGroup:
        print(i)

if __name__ == "__main__":
    main()
    #pdb.set_trace()