import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, datasets, ensemble
#from sklearn.datasets import make_gaussian_quantiles

x1, y1 = datasets.make_gaussian_quantiles(
    cov = 2., n_samples=500, n_features=2, n_classes=2, shuffle=True, random_state=1)
x2, y2 = datasets.make_gaussian_quantiles(
    mean=(3, 3), cov=1.5, n_samples=400, n_features=2, n_classes=2, shuffle=True, random_state=1)

X = np.concatenate((x1, x2))
y = np.concatenate((y1, 1-y2))

#plt.scatter(X[:,0], X[:,1], c=y)
#plt.show()

weakClassifier = tree.DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=20,
    min_samples_leaf=5)

clf = ensemble.AdaBoostClassifier(
    base_estimator=weakClassifier,
    algorithm="SAMME",
    n_estimators=200,
    learning_rate=0.8)

clf.fit(X, y)

x1_min = X[:,0].min() - 1
x1_max = X[:,0].max() + 1
x2_min = X[:,1].min() - 1
x2_max = X[:,1].max() + 1

x_, y_ = np.meshgrid(
    np.arange(x1_min, x1_max, 0.02),
    np.arange(x2_min, x2_max, 0.02))

y_pre = clf.predict(np.c_[x_.ravel(), y_.ravel()])
y_pre = y_pre.reshape(x_.shape)

plt.contourf(x_, y_, y_pre, cmap=plt.cm.Paired)
plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

print("Score_1:", clf.score(X,y))

weakClassifier = tree.DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=20,
    min_samples_leaf=5)

clf = ensemble.AdaBoostClassifier(
    base_estimator=weakClassifier,
    algorithm="SAMME",
    n_estimators=300, 
    learning_rate=0.8)

clf.fit(X, y)
print("Score_2:", clf.score(X,y))

weakClassifier = tree.DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=20,
    min_samples_leaf=5)
    
clf = ensemble.AdaBoostClassifier(
    base_estimator=weakClassifier,
    algorithm="SAMME",
    n_estimators=300, 
    learning_rate=0.5)

clf.fit(X, y)
print("Score_3:", clf.score(X,y))

weakClassifier = tree.DecisionTreeClassifier(
    max_depth=2,
    min_samples_split=20,
    min_samples_leaf=5)
    
clf = ensemble.AdaBoostClassifier(
    base_estimator=weakClassifier,
    algorithm="SAMME",
    n_estimators=600, 
    learning_rate=0.8)

clf.fit(X, y)
print("Score_4:", clf.score(X,y))

