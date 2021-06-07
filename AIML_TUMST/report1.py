from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

boston = load_boston()
x = boston.data[:,-1]; y = boston.target

x_mean = x.mean(axis=0) ##(13,)
x_variant = (x ** 2).mean(axis=0) - x_mean ** 2 ##(13,)
x_std = np.array((x - x_mean) / np.sqrt(x_variant)) ## (506, 13)

x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size=0.25)

xc_train = sm.add_constant(x)
lr = sm.OLS(y, xc_train).fit()

x_axis = np.linspace(0, 100, 100)
xc_test = sm.add_constant(x_axis)
y_pred = lr.predict(xc_test)

logx = -np.log(x)
logy = -np.log(y)
xc_train = sm.add_constant(logx)
lr = sm.OLS(logy, xc_train).fit()

xc_test = sm.add_constant(x_axis)
y_pred_2 = lr.predict(xc_test)

#plt.xlim(0, 40); plt.ylim(0, 51)
plt.scatter(x, y, s=10, c='blue')
plt.plot(x_axis, y_pred, 'r-', lw=3, label="Linear Regression")
plt.plot(-np.exp(x_axis), -np.exp(y_pred_2), 'g-', lw=3, label="log Regression")
plt.legend()
plt.show()