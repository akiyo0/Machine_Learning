## 对 卷积神经网络(CNN) 的若干点补充

### Batch_Size
Batch_Size（批尺寸）是机器学习中一个重要参数。Batch 的选择，首先决定的是下降的方向。
如果数据集比较小，完全可以采用全数据集 (Full Batch Learning) 的形式，这样做至少有 2 个好处：
+ 由全数据集确定的方向能够更好地代表样本总体，从而更准确地朝向极值所在的方向。
+ 由于不同权重的梯度值差别巨大，因此选取一个全局的学习率很困难。 Full Batch Learning 可以使用 Rprop 只基于梯度符号并且针对性单独更新各权值。

对于更大的数据集，以上 2 个好处又变成了 2 个坏处：
+ 随着数据集的海量增长和内存限制，一次性载入所有的数据进来变得越来越不可行。
+ 以 Rprop 的方式迭代，会由于各个 Batch 之间的采样差异性，各次梯度修正值相互抵消，无法修正。这才有了后来 RMSProp 的妥协方案。

既然 Full Batch Learning 并不适用大数据集，那么走向另一个极端怎么样？所谓另一个极端，就是每次只训练一个样本，即 Batch_Size = 1。即在线学习 (Online Learning)。
线性神经元在均方误差代价函数的错误面是一个抛物面，横截面是椭圆。对于多层神经元、非线性网络，在局部依然近似是抛物面。使用在线学习，每次修正方向以各自样本的梯度方向修正，横冲直撞各自为政，难以达到收敛。

可不可以选择一个适中的 Batch_Size 值呢？
当然可以，这就是批梯度下降法（Mini-batches Learning）。因为如果数据集足够充分，那么用一半（甚至少得多）的数据训练算出来的梯度与用全部数据训练出来的梯度是几乎一样的。
引用元：https://blog.csdn.net/ycheng_sjtu/article/details/49804041


### 2
```python
import torch

x = torch.randn(2,1,7,3)
conv = torch.nn.Conv2d(1,8,(2,3))
res = conv(x)

print(res.shape)    # shape = (2, 8, 6, 1)
```
这其中，输入函数
```python
torch.randn([batch_size, channels, height, width])
```

| 参数名 | 解释 | 输入值 |
| --- | --- | --- |
| batch_size | 一个batch中样例的个数 | 2 |
| channels | 通道数，也就是当前层的深度 | 1 |
| height *1 | 图片的高 | 7 |
| width *1 | 图片的宽 | 3 |

```python
torch.nn.Conv2d([channels, output, height, width])
```

| 参数名 | 解释 | 输入值 |
| --- | --- | --- |
| channels | 通道数(同上的解释) | 1 |
| output | 输出的深度 | 8 |
| height *2 | 过滤器filter的高 | 2 |
| width *2 | 过滤器filter的宽 | 3 |

得到结果
```python
res -> [batch_size, output, height, width]
```
| 参数名 | 解释 | 输入值 |
| --- | --- | --- |
| batch_size | 一个batch中样例的个数，同上 |2|
| output | 输出的深度 | 8 |
| height *3 | 卷积结果的高度 | 6 | 
| width *3 | 卷积结果的宽度 | 1 |

 height *3 = height *1 - height *2 + 1 = 7 - 2 + 1
width *3 = width *1 - width *2 + 1 = 3 - 3 + 1 

### modules()
```python
for name,sub_module in net.named_children():
    if name in ['conv1']:
        print(sub_module)
```
```python
params = list(net.parameters())
print(len(params))
```
```python
for name,parameters in net.named_parameters():
    print(name, ':', parameters.size())
```
```python
for module in model.modules():
    print(module)
```
```python
for name, module in model.named_children():
    if name in ['conv4', 'conv5']:
        print(module)
```
```python
for param in model.parameters():
    print(type(param.data), param.size())
```
输出结果
`<class 'torch.FloatTensor'> (20L,)
<class 'torch.FloatTensor'> (20L, 1L, 5L, 5L)`

# 卷积神经网络（CNN）的层级结构
+ 数据输入层 (Input layer)
+ 卷积计算层 (CONV layer)
+ ReLU激励层 (ReLU layer)
+ 池化层 (Pooling layer)
+ 全连接层 (FC layer)

## 数据输入层
该层要做的处理主要是对原始图像数据进行预处理，其中包括：
+ 去均值：把输入数据各个维度都中心化为0，如下图所示，其目的就是把样本的中心拉回到坐标系原点上。
+ 归一化：幅度归一化到同样的范围，如下所示，即减少各维度数据取值范围的差异而带来的干扰，比如，我们有两个维度的特征A和B，A范围是0到10，而B范围是0到10000，如果直接使用这两个特征是有问题的，好的做法就是归一化，即A和B的数据都变为0到1的范围。
+ PCA[^pca]/白化：用PCA降维；白化是对数据各个特征轴上的幅度归一化

[^pca]: PCA(主成分分析, Principal Component Analysis) 是一种常见的数据分析方式，常用于高维数据的降维，可用于提取数据的主要特征分量。

## 卷积层
### 卷积

1. 卷积的定义[^define_of_C]
训练神经网络生成图片，将低分辨率的图片转换为高分辨率的图片时，通常会使用插值方法进行处理。
   + 最近邻插值 (Nearest neighbor interpolation)
   + 双线性插值 (Bi-Linear interpolation)
   + 双立方插值 (Bi-Cubic interpolation)
如果我们想要我们的网络可以学习到最好地上采样的方法，我们这个时候就可以采用转置卷积。这个方法不会使用预先定义的插值方法，它具有可以学习的参数。

卷积操作中输入值和输出值之间存在位置上的连接关系。
卷积操作是多对一(many-to-one)的映射关系，而转置卷积为一对多(one-to-many)的映射关系。

1. 转置卷积
对输入矩阵的操作：将其摊平(flatten)为列向量。
对卷积核的操作：
$$\left(\begin{array}{cccccccccccccccc}
w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{1,1} & w_{1,2} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 & 0 & 0 & 0 & 0 \\
0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{1,1} & w_{1,2} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{1,1} & w_{1,2} & 0 & w_{2,0} & w_{2,1} & w_{2,2} & 0 \\
0 & 0 & 0 & 0 & 0 & w_{0,0} & w_{0,1} & w_{0,2} & 0 & w_{1,0} & w_{1,1} & w_{1,2} & 0 & w_{2,0} & w_{2,1} & w_{2,2}
\end{array}\right)$$
然后对操作后的 卷积核 和 数据矩阵 进行矩阵乘法。
 
4. 反卷积


[^define_of_C]: 参考文献1：[如何通俗易懂地解释卷积？马同学](https://www.zhihu.com/question/22298352/answer/228543288)<br>参考文献2：[一文搞懂反卷积，转置卷积](https://blog.csdn.net/LoseInVain/article/details/81098502)

### 卷积层关键操作
+ 局部关联：每个神经元看做一个滤波器(filter)
+ 窗口滑动：filter对局部数据计算
+ 感受野：receptive field

关键词
+ 深度/depth（解释见下图）
+ 步长/stride （窗口一次滑动的长度）
+ 填充值/zero-padding

### torch.nn.Conv1d
```python
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```
一维卷积层，输入的尺度是 $\left(N, C_{\mathrm{in}}, L\right)$，输出尺度 $\left(N, C_{\text{out}}, L_{\text {out}}\right)$ 的计算方式：
$$ \operatorname{out}\left(N_{i}, C_{\mathrm{out}_{j}}\right)=\operatorname{bias}\left(C_{\mathrm{out}_{j}}\right)+\sum_{k=0}^{C_{in}-1} \operatorname{weight}\left(C_{\mathrm{out}_{j}}, k\right) \star \operatorname{input}\left(N_{i}, k\right) $$

where $\star$ is the valid cross-correlation operator(有效互相关演算子), $N$ is a batch size, $C$ denotes a number of channels, $L$ is a length of signal sequence.

+ `in_channels` (python:int) – Number of channels in the input image
+ `out_channels` (python:int) – Number of channels produced by the convolution
+ `kernel_size` (python:int or tuple) – Size of the convolving kernel
+ `stride` (python:int or tuple, optional) – Stride of the convolution. Default: 1
+ `padding` (python:int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
+ `padding_mode` (string, optional) – zeros
+ `dilation` (python:int or tuple, optional) – Spacing between kernel elements. Default: 1
+ `groups` (python:int, optional) – Number of blocked connections from input channels to output channels. Default: 1
+ `bias` (bool, optional) – If `True`, adds a learnable bias to the output. Default: `True`

### torch.nn.Conv2d
Applies a 2D convolution over an input signal composed of several input planes.

In the simplest case, the output value of the layer with input size $\left(N, C_{\mathrm{in}}, H, W\right)$ and output $\left(N, C_{\text{out}}, H_{\text {out}}, W_{\text{out}}\right)$  can be precisely described as:
```python
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```
$$\operatorname{out}\left(N_{i}, C_{\text{out}_{j}}\right)=\operatorname{bias}\left(C_{\text{out}_{j}}\right)+\sum_{k=0}^{C_{\text{in}}-1} \text {weight}\left(C_{\text{out}_{j}}, k\right) \star \text{input}\left(N_{i}, k\right)$$

where $\star$ is the valid 2D cross-correlation operator, $N$ is a batch size, $C$ denotes a number of channels, $H$ is a height of input planes in pixels, and $W$ is width in pixels.

## 参数共享机制

权值共享
图片的底层特征是与特征在图片中的位置无关的。
输出层的每一个像素，是由输入层对应位置的 $F*F$ 的局部图片，与相同的一组 $F*F$ 的参数（或称权值）做内积，再经过非线性单元计算而来的。

在卷积层中每个神经元连接数据窗的权重是固定的，每个神经元只关注一个特性。神经元就是图像处理中的滤波器，比如边缘检测专用的Sobel滤波器，即卷积层的每个滤波器都会有自己所关注一个图像特征，比如垂直边缘，水平边缘，颜色，纹理等等，这些所有神经元加起来就好比就是整张图像的特征提取器集合。
需要估算的权重个数减少: AlexNet 1亿 => 3.5w
一组固定的权重和不同窗口内数据做内积: 卷积


## 仿射层/全连接层
### torch.nn.Linear
Applies a linear transformation to the incoming data: $y=x A^{T}+b$ 
```python
class torch.nn.Linear(in_features, out_features, bias=True)
```
+ `in_features` – size of each input sample
+ `out_features` – size of each output sample
+ `bias` – If set to `False`, the layer will not learn an additive bias. Default: `True`

## 激活层
如果网络中的所有隐含单元的激活函数都取线性函数，那么对于任何这种网络，我们总可以找到一个等价的无隐含单元的网络。<u>这是由于连续的线性变换的组合本身是一个线性变换</u>。然而，如果隐含单元的数量小于输入单元的数量或者小于输出单元的数量，那么网络能够产生的变换不是最一般的从输入到输出的线性变换，因为在隐含单元出的维度降低造成了信息丢失。（来源PRML）
https://blog.csdn.net/xingchengmeng/article/details/56289427
https://zhuanlan.zhihu.com/p/92412922

## 池化层
池化层夹在连续的卷积层中间， 用于<u>压缩数据和参数的量</u>，<u>减小过拟合</u>。简而言之，如果输入是图像的话，那么池化层的最主要作用就是压缩图像。

下采样层也叫池化层，其具体操作与卷积层的操作基本相同，只不过下采样的卷积核为只取对应位置的最大值、平均值等（最大池化、平均池化），即矩阵之间的运算规律不一样，并且不经过反向传播的修改。

池化操作就是图像的resize。池化层的具体作用包括：
1. <b>特征不变性(invariance)</b>。在图像处理中也称为特征的尺度不变性。这种不变性包括translation(平移)，rotation(旋转)和scale(尺度)。图像压缩时去掉无关紧要的信息，留下的信息依然包含最重要的特征。则可认为具有尺度不变性的特征，即最能表达图像的特征。
2. <b>特征降维</b>，即将图像信息之类中的冗余信息去除，抽取重要特征。
3. 保留主要的特征同时减少参数(降维，效果类似PCA)和计算量，防止过拟合，提高模型泛化能力
4. 一定程度上<b>防止过拟合</b>，方便优化。

手段主要包括：
`Max Pooling` 和 `Average Pooling`。实际中应用较多的是 `Max pooling`。
对于每个 $2*2$ 的窗口选出最大的数作为输出矩阵的相应元素的值，比如输入矩阵第一个 $2*2$ 窗口中最大的数是6，那么输出矩阵的第一个元素就是6，如此类推。

参考链接：
[1] https://blog.csdn.net/weixin_38145317/article/details/89310404

## 全连接层
两层之间所有神经元都有权重连接，通常全连接层在卷积神经网络尾部。也就是跟传统的神经网络神经元的连接方式是一样的：

### 一般CNN结构依次为
1. `INPUT`
2. [[`CONV`->`RELU`] * N -> `POOL`?] * M 
3. [`FC`->`RELU`] * K
4. `FC`

### 训练算法
1. 同一般机器学习算法，先定义Loss function，衡量和实际结果之间差距。
2. 找到最小化损失函数的 $W$ 和 $b$， CNN中用的算法是SGD（随机梯度下降）。

### 典型CNN
+ LeNet，这是最早用于数字识别的CNN
+ AlexNet， 2012 ILSVRC比赛远超第2名的CNN，比LeNet更深，用多层小卷积层叠加替换单大卷积层。
+ ZF Net， 2013 ILSVRC比赛冠军
+ GoogLeNet， 2014 ILSVRC比赛冠军
+ VGGNet， 2014 ILSVRC比赛中的模型，图像识别略差于GoogLeNet，但是在很多图像转化学习问题(比如object detection)上效果奇好

### fine-tuning
fine-tuning就是使用已用于其他目标、预训练好模型的权重或者部分权重，作为初始值开始训练。
那为什么我们不用随机选取选几个数作为权重初始值？原因很简单，第一，自己从头训练卷积神经网络容易出现问题；第二，fine-tuning能很快收敛到一个较理想的状态，省时又省心。
具体做法:
+ 复用相同层的权重，新定义层取随机权重初始值
+ 调大新定义层的的学习率，调小复用层学习率


## 特征选择 AND 特征提取[^feature]
**特征选择**（feature selection）和 **特征提取**（Feature extraction）都属于(或可以统称为)降维（Dimension reduction）。
这两者都是试图减少特征数据集中的属性（或者称为特征）的数目；但是两者采用不同的方法。
**特征提取**通过属性间的关系，组合不同的属性得到新的属性，这样改变了原来的特征空间。主要手段包括 PCA、LDA和SVD[^SVD]。
CNN特征提取[^Extraction]：
[^Extraction]: [三大特征提取器（RNN/CNN/Transformer）](https://www.cnblogs.com/sandwichnlp/p/11612596.html#卷积神经网络cnn)
**特征选择**是从原始特征数据集中选择出子集，是一种包含的关系，没有更改原始的特征空间。主要手段包含以下：
1. Filter
    主要思想是：对每一维的特征“打分”，即给每一维的特征赋予权重，这样的权重就代表着该维特征的重要性，然后依据权重排序。
    主要的方法有：Chi-squared test(卡方检验)，ID3(信息增益)，correlation coefficient scores(相关系数)
2. Wrapper
    其主要思想是：将子集的选择看作是一个搜索寻优问题，生成不同的组合，对组合进行评价，再与其他的组合进行比较。这样就将子集的选择看作是一个是一个优化问题，这里有很多的优化算法可以解决，尤其是一些启发式的优化算法，如GA，PSO，DE，ABC等，详见“优化算法——人工蜂群算法(ABC)”，“优化算法——粒子群算法(PSO)”。
    主要方法有：recursive feature elimination algorithm(递归特征消除算法)
3. Embedded
    主要思想是：在模型既定的情况下学习出对提高模型准确性最好的属性。这句话并不是很好理解，其实是讲在确定模型的过程中，挑选出那些对模型的训练有重要意义的属性。
    主要方法：正则化。如岭回归就是在基本线性回归的过程中加入了正则项。

[^feature]: 参考文献：[特征选择与特征提取](https://blog.csdn.net/qq_41996090/article/details/88076031)
[^SVD]: SVD：奇异值分解 (Singular Value Decomposition) 本质上是一种数学的方法，在机器学习领域中被广泛使用。

## 输入空间、特征空间、输出空间[^input space]
[^input space]: 参考文献：[机器学习（一）--输入空间、特征空间、输出空间](https://blog.csdn.net/hz_jhx/article/details/80727431)
**输入空间**：输入 $X$ 的所有可能取值的集合为输入空间 (input space)。输入空间可以是有限集合空间(finite topological space?)，也可以是整个欧氏空间(euclidean space)。输出 $Y$ 可能取值的集合是**输出空间** (output space) 也是同样。

**特征空间**：对于输入空间每个具体的输入称为**一个实例**(an instance)，这个实例是由特征向量(feature vector)表示。下式中 $x$ 为输入空间 $X$ 中的一个输入实例，由 $n$ 维特征向量组成
$$x=\left(x^{(1)}, x^{(2)}, \cdots, x^{(\mathrm{n})}\right)^{T} (1.1)$$

一般的，用大写的 $X$ OR $Y$ 代表输入输出空间，小写的 $x$ OR $y$ 代表一个具体的输入输出实例（标量或者向量）。向量表示时默认为列向量，对于输入向量，通常用行向量转置的方式来表示列向量。
$$x_{i}=\left(x_{i}^{(1)}, x_{i}^{(2)}, \cdots, x_{i}^{(\mathrm{n})}\right)^{T} (1.2)$$

$(1.1)$ 与 $(1.2)$ 的不同之处在于，$x_i$ 代表多个输入变量 $x$ 中的第 $i$ 个。式 $(1.2)$ 表示，第 $i$ 个输入变量中，由 $n$ 个特征组成。

