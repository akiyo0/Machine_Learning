import numpy as np
import math
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd

class DecisionStump():
    def __init__(self):
        self.polarity = 1 #-1 or 1
        self.feature_index = None #用于分类的特征索引
        self.threshold = None
        self.alpha = None

class Adaboost():
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
    
    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        #print(n_samples, 1 / n_samples)
        w = np.full((n_samples, 1), (1 / n_samples))
        self.clfs = []

        for _ in range(self.n_clf):
            clf = DecisionStump()
            min_error = float('inf')

            for i_feature in range(n_features):
                #feature_values = np.expand_dims(X_train[:, i_feature], axis=1)
                feature_values = X_train[:, i_feature].reshape(-1, 1)
                unique_values = np.unique(feature_values) #取唯一值
                
                for threshold in unique_values: #遍历可能阈值
                    p = 1
                    prediction = np.ones(y_train.shape)
                    prediction[X_train[:, i_feature] < threshold] = -1
                    error = np.sum(w[y_train != prediction])

                    if error > 0.5:
                        error = 1 - error
                        p = -1
                    
                    if error < min_error:
                        clf.polarity = p
                        clf.threshold = threshold
                        clf.feature_index = i_feature
                        min_error = error

            clf.alpha = 0.5 * math.log((1.0 - min_error) / (min_error + 1e-10))
            predictions = np.ones(y_train.shape)
            negative_idx = (clf.polarity * X_train[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1

            w *= np.exp(-clf.alpha * y_train * predictions)
            w /= np.sum(w)

            self.clfs.append(clf)

    def predict(self, X_test):
        n_samples = X_test.shape[0]
        y_pred = np.zeros((n_samples, 1))

        for clf in self.clfs:
            predictions = np.ones((y_pred.shape))
            negative_idx = (clf.polarity * X_test[:, clf.feature_index] < clf.polarity * clf.threshold)
            predictions[negative_idx] = -1
            y_pred += clf.alpha * predictions

        y_pred = np.sign(y_pred).flatten()
        return y_pred
    
    

def main():
    dataset = datasets.load_digits()
    X, y = dataset.data, dataset.target
    digit = [1, 8]
    idx = np.append(np.where(y == digit[0])[0], np.where(y == digit[1])[0])
    X, y = X[idx], y[idx].reshape(-1, 1)

    y[y == digit[0]], y[y == digit[1]] = -1, 1 # Change labels to {-1, 1}

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    clf = Adaboost(n_clf=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)





if __name__ == "__main__":
    main()
