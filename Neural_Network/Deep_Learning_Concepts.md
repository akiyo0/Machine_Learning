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




## 梯度

### 梯度消失及梯度爆炸💥问题（gradient vanishing and gradient exploding problem）
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

如果此部分大于1，那么层数增多的时候，最终的求出的梯度更新将以指数形式增加，即发生梯度爆炸，如果此部分小于1，那么随着层数增多，求出的梯度更新信息将会以指数形式衰减，即发生了梯度消失。https://blog.csdn.net/qq_25737169/article/details/78847691

其实梯度爆炸和梯度消失问题都是因为网络太深，网络权值更新不稳定造成的，本质上是因为梯度反向传播中的连乘效应。对于更普遍的梯度消失问题，可以考虑用ReLU激活函数取代sigmoid激活函数。另外，LSTM的结构设计也可以改善RNN中的梯度消失问题。

https://zhuanlan.zhihu.com/p/25631496
 
 

 
 
