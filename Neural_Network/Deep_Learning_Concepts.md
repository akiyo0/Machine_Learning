# 传统神经网络
## 构成要件
### 训练
Epoch, Batch, Iteration
基本概念：
Epoch：使用训练集的全部数据对模型进行一次完整训练，被称之为“一代训练”；或一整次完全训练
Batch：使用训练集中的一小部分样本对模型权重进行一次反向传播的参数更新，这一小部分样本被称为“一批数据”；
Literation：使用一个 Batch 数据对模型进行一次参数更新的过程，被称之为“一次训练”
换算关系：
$$\text{Number of Batches}=\frac{\text { TrainingSet Size }}{\text{Batch Size}}$$
实际上，梯度下降的几种方式的根本区别就在于上面公式中的 Batch Size 不同。

| 梯度下降方式 | TrainingSet Size | Batch Size | Number of Batches |
| --- | --- | --- | --- |
| BGD | $N$ | $N$ | $1$ |
| SGD | $N$ | $1$ | $N$ |
| Mini-Batch | $N$ | $B$ | $\frac{N}{B+1}$ |

注：上表中 Mini-Batch 的 Batch 个数为 $\frac{N}{B+1}$ 是针对未整除的情况。整除则是 $\frac{N}{B}$。

e.g. CIFAR10 数据集有 50000 张训练图片，10000 张测试图片。现在选择 Batch Size = 256 对模型进行训练。
1. 每代 Epoch 要训练的图片数量：$50000$
2. 训练集具有的 Batch 个数：$50000 / 256 \approx 195+1=196$
3. 每个 Epoch 需要完成的 Batch 个数：$196$
4. 每个 Epoch 具有的 Iteration 个数：$196$
5. 每个 Epoch 中发生模型权重更新的次数：$196$
6. 训练 10 代后，模型权重更新的次数：$196\times10=1960$
7. 不同代的训练，其实用的是同一个训练集的数据。第 1 代和第 10 代虽然用的都是训练集的全部图片，但是对模型的权重更新值却是完全不同的。因为不同代的模型处于代价函数空间上的不同位置，模型的训练代越靠后，越接近谷底，其代价越小。
https://www.jianshu.com/p/22c50ded4cf7
### 梯度

**梯度消失及梯度爆炸💥问题
(Gradient vanishing and gradient exploding problem)**

**梯度爆炸问题**和**梯度消失问题**一般随着网络层数的增加会变得越来越明显。

例如，对于下面一个含有3个隐藏层的神经网络（`HDL1`, `HDL2`, `HDL3`）
`INT` -> `HDL1` -> `HDL2` -> `HDL3` -> `OUT`
当**梯度消失问题**发生时，接近于输出层(`OUT`)的`HDL3`及`HDL2`的权值更新相对正常，但接近于输出层(`INT`)的`HDL1`的权值更新会变得很慢，<u>以至于这些层权值没有发生显著改变，接近于初始化的权值</u>，导致前面的层只对所有的输入做了一个同一映射（即映射层）。也解释了为何更深层的神经网络的学习结果仅等价于后几层浅层网络的学习。
 
**梯度消失问题产生的具体过程**
以下图的反向传播为例（假设每一层只有一个神经元且对于每一层$y_{i}=\sigma\left(z_{i}\right)=\sigma\left(w_{i} x_{i}+b_{i}\right)$，其中 $\sigma$ 为sigmoid函数）
`START` --> `b1` -$w_2$-> `b2` -$w_3$-> `b3` -$w_4$-> `b4` --> `C`
$$\begin{aligned}
\frac{\partial C}{\partial b_{1}} &=\frac{\partial C}{\partial y_{4}} \frac{\partial y_{4}}{\partial z_{4}} \frac{\partial z_{4}}{\partial x_{4}} \frac{\partial x_{4}}{\partial z_{3}} \frac{\partial z_{3}}{\partial x_{3}} \frac{\partial x_{3}}{\partial z_{2}} \frac{\partial z_{2}}{\partial x_{2}} \frac{\partial x_{2}}{\partial z_{1}} \frac{\partial z_{1}}{\partial b_{1}} \\
&=\frac{\partial C}{\partial y_{4}} \sigma^{\prime}\left(z_{4}\right) w_{4} \sigma^{\prime}\left(z_{3}\right) w_{3} \sigma^{\prime}\left(z_{2}\right) w_{2} \sigma^{\prime}\left(z_{1}\right)
\end{aligned}$$

根据 $\sigma^{\prime}(x)$ 函数在定义域 $[-5,5]$ 图像可知。$\sigma^{\prime}(x)$ 的最大值为 $\frac{1}{4}$，而我们初始化的网络权值 $|w|$ 通常都小于1，因此 $\left|\sigma^{\prime}(z) w\right| \leq \frac{1}{4}$，因此对于上面的链式求导，层数越多，求导结果 $\frac{\partial C}{\partial b_{1}}$ 越小，因而导致梯度消失的情况出现。

这样，梯度爆炸问题的出现原因就显而易见了，即 $\left|\sigma^{\prime}(z) w\right|>1$，也就是 $\sigma^{\prime}(z)$ 比较大的情况。但对于使用sigmoid激活函数来说，这种情况比较少。因为 $\sigma^{\prime}(z)$ 的大小也与 $w$ 有关（$z=wx+b$），除非该层的输入值 $x$ 在一直一个比较小的范围内。

如果此部分大于1，那么层数增多的时候，最终的求出的梯度更新将以指数形式增加，即发生梯度爆炸，如果此部分小于1，那么随着层数增多，求出的梯度更新信息将会以指数形式衰减，即发生了梯度消失。
https://blog.csdn.net/qq_25737169/article/details/78847691

其实梯度爆炸和梯度消失问题都是因为网络太深，网络权值更新不稳定造成的，本质上是因为梯度反向传播中的连乘效应。对于更普遍的梯度消失问题，可以考虑用ReLU激活函数取代sigmoid激活函数。另外，LSTM的结构设计也可以改善RNN中的梯度消失问题。

https://zhuanlan.zhihu.com/p/25631496




## 损失函数 目标函数 Loss
Cost/Loss(Min) Objective(Max) Functions

### 最大似然估计 [A-a-1]
1. Many cost functions are the result of applying Maximum Likelihood. For instance, the Least Squares cost function can be obtained via Maximum Likelihood. Cross-Entropy is another example.

2. The likelihood of a parameter value (or vector of parameter values), $θ$, given outcomes $x$, is equal to the probability (density) assumed for those observed outcomes given those parameter values, that is $$\mathcal{L}(\theta | x)=P(x | \theta)$$

3. The natural logarithm of the likelihood function, called the log-likelihood, is more convenient to work with. Because the logarithm is a monotonically increasing function, the logarithm of a function achieves its maximum value at the same points as the function itself, and hence the log-likelihood can be used in place of the likelihood in maximum likelihood estimation and related techniques.

4. In general, for a ﬁxed set of data and underlying statistical model, the method of maximum likelihood selects the set of values of the model parameters that maximizes the likelihood function. Intuitively, this maximizes the "agreement" of the selected model with the observed data, and for discrete random variables it indeed maximizes the probability of the observed data under the resulting distribution. Maximum-likelihood estimation gives a uniﬁed approach to estimation, which is well-deﬁned in the case of the normal distribution and many other problems.
$$f\left(x_{1}, x_{2}, \ldots, x_{n} | \theta\right)=f\left(x_{1} | \theta\right) \times f\left(x_{2} | \theta\right) \times \cdots \times f\left(x_{n} | \theta\right)$$

$$\mathcal{L}\left(\theta ; x_{1}, \ldots, x_{n}\right)=f\left(x_{1}, x_{2}, \ldots, x_{n} | \theta\right)=\prod_{i=1}^{n} f\left(x_{i} | \theta\right)$$

$$\ln \mathcal{L}\left(\theta ; x_{1}, \ldots, x_{n}\right)=\sum_{i=1}^{n} \ln f\left(x_{i} | \theta\right)$$

$$\hat{\ell}(\theta ; x)=\frac{1}{n} \sum_{i=1}^{n} \ln f\left(x_{i} | \theta\right)$$

$$\left\{\hat{\theta}_{\text {mle}}\right\} \subseteq\left\{\underset{\theta \in \Theta}{\arg \max} \hat{\ell}\left(\theta ; x_{1}, \ldots, x_{n}\right)\right\}$$

### 交叉熵 (Cross-Entropy) [A-a-2]
Cross entropy can be used to deﬁne the loss function in machine learning and optimization. The true probability $p_i$ is the true label, and the given distribution $q_i$ is the predicted value of the current model.
$$\begin{aligned}
&H(p, q)=\mathrm{E}_{p}\left[l_{i}\right]=\mathrm{E}_{p}\left[\log \frac{1}{q\left(x_{i}\right)}\right]\\
&H(p, q)=\sum_{x_{i}} p\left(x_{i}\right) \log \frac{1}{q\left(x_{i}\right)}\\
&H(p, q)=-\sum_{x} p(x) \log q(x)
\end{aligned}
$$

Cross-entropy error function and logistic regression
$$
L(\mathbf{w})=\frac{1}{N} \sum_{n=1}^{N} H\left(p_{n}, q_{n}\right)=-\frac{1}{N} \sum_{n=1}^{N}\left[y_{n} \log \hat{y}_{n}+\left(1-y_{n}\right) \log \left(1-\hat{y}_{n}\right)\right]
$$


## 最优化
### 优化器
#### 动量（Momentum）
mini-batch SGD 算法虽然有很快的训练速度，但结果并不总是全局最优。另外需要挑选合适的超参（学习率），不合适的超参会导致收敛速度过慢（震荡收敛？）或跳过最优区间。
Momentum基于梯度的移动指数加权平均，可以解决mini-batch SGD优化算法更新幅度摆动大的问题，同时可以使得网络的收敛速度更快。
假设在当前的迭代步骤第 $t$ 步中，那么基于Momentum优化算法可以写成下面的公式： 
$$\begin{array}{c}
v_{d w}=\beta v_{d w}+(1-\beta) d W \\
v_{d b}=\beta v_{d b}+(1-\beta) d b \\
W=W-\alpha v_{d w} \\
b=b-\alpha v_{d b}
\end{array}$$
其中，在上面的公式中 $v_{dw}$ 和 $v_{db}$ 分别是损失函数在前 $t−1$ 轮迭代过程中累积的梯度动量，$β$ 是梯度累积的一个指数，一般设置值为 $β=0.9$。$dW$ 和 $db$ 分别是损失函数反向传播时候所求得的梯度，下面两个公式是网络权重向量和偏置向量的更新公式，$α$ 是网络的学习率。
所以Momentum优化器的主要思想就是利用了类似与**指数加权移动平均法[^EWMA]**的方法来对网络的参数进行平滑处理的，让梯度的摆动幅度变得更小。 
* [ ] ???为什么

[^EWMA]: **指数加权移动平均法(EWMA)**：Exponentially weighted averages。根据同一个移动段内不同时间的数据对预测值的影响程度，分别给予不同的权数，然后再进行平均移动以预测未来值。加权移动平均法，是对观察值分别给予不同的权数，按不同权数求得移动平均值，并以最后的移动平均值为基础，确定预测值的方法。采用加权移动平均法，是因为观察期的近期观察值对预测值有较大影响，它更能反映近期变化的趋势。指数移动加权平均法，是指各数值的加权系数随时间呈指数式递减，越靠近当前时刻的数值加权系数就越大。指数移动加权平均较传统的平均法来说，一是不需要保存过去所有的数值；二是计算量显著减小。

#### RMSprop
在上面的Momentum优化算法中，虽然初步解决了损失函数在更新中存在的摆动幅度[^摆动幅度]过大问题。为了进一步优化，并且进一步加快函数的收敛速度，Geoffrey E. Hinton在Coursera课程中提出的一种新的优化算法。
[^摆动幅度]: 摆动幅度：指在优化中经过更新之后参数的变化范围。
RMSProp算法的全称叫 Root Mean Square Prop，如下图所示，RMSProp算法拥有较Momentum算法更快的收敛速度。
![](https://img-blog.csdn.net/20170923134334368?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvd2lsbGR1YW4x/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
蓝色的为Momentum优化算法收敛路径，绿色的为RMSProp优化算法收敛路径。 

与Momentum算法不同的是，RMSProp算法对梯度计算了**微分平方加权平均数**。这样利于消除摆动幅度大的方向，修正摆动幅度，使得各个维度的摆动幅度降低。另一方面也使得网络函数收敛更快。

RMSProp算法对权重 $W$ 和偏置 $b$ 的梯度使用了**微分平方加权平均数**。 其中，假设在第 $t$ 轮迭代过程中： 
$$\begin{aligned}
s_{d w} &=\beta s_{d w}+(1-\beta) d W^{2} \\
s_{d b} &=\beta s_{d b}+(1-\beta) d b^{2} \\
W &=W-\alpha \frac{d W}{\sqrt{s_{d b}+\varepsilon}} \\
b &=b-\alpha \frac{d b}{\sqrt{s_{d b}+\varepsilon}}
\end{aligned}$$

在上面的公式中 $s_{dw}$ 和 $s_{db}$ 分别是损失函数在前 $t−1$ 轮迭代过程中累积的梯度梯度动量，$β$ 是梯度累积的一个指数。（当 $dW$ 或者 $db$ 中有一个值比较大的时候，那么在更新权重或者偏置的时候除以它之前累积的梯度的平方根，这样就可以使得更新幅度变小）。为了防止分母为零，使用了一个很小的数值 $ϵ$ 来进行平滑，一般取值为 $10^{−8}$。

#### Adam
以上两种优化算法，一种是类似于物理中的动量来累积梯度，另一种使得收敛速度更快同时减小损失函数的波动。Adam（Adaptive Moment Estimation）算法是将Momentum算法和RMSProp算法结合起来的一种算法。
使用的参数基本和结合之前保持一致，在训练的开始前需要初始化梯度累积量和平方累积量。 
$$v_{d w}=0, v_{d b}=0 ; s_{d w}=0, s_{d b}=0$$


