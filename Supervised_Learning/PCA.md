## PCA 主成分分析
PCA（Principal Component Analysis） 主成分分析是最常用的一种降维方法，可用于提取数据的主要特征分量。[^definition]

[^definition]: 机器学习：PCA降维 https://blog.csdn.net/zouxiaolv/article/details/100590725

对于正交矩阵空间中的样本点，如何用一个超平面对所有样本进行恰当的表达。容易想到，如果这样的超平面存在，那么他大概应该具有下面的性质。

+ 最大可分性：样本点在这个超平面上的投影尽可能分开
+ 最近重构性：样本点到超平面的距离都足够近

PCA 的数学推导可以从**最大可分型**和**最近重构性**两方面进行，前者的优化条件为划分后方差最大，后者的优化条件为点到划分平面距离最小。

基于最近重构性和最大可分性，能分别得到主成分分析的两种等价推导。[^2]

[^2]: 主成分分析：https://endymecy.gitbooks.io/spark-ml-source-analysis/content/%E9%99%8D%E7%BB%B4/PCA/pca.html?q=

基于最近重构性的 PCA 其与 LDA（线性判别法）的数学推导有异曲同工之处。



