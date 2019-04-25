## [Object Detection with Discriminatively Trained Part Based Models](Object Detection with Discriminatively Trained Part Based Models)

[论文和对应的代码](http://www.cs.berkeley.edu/~rbg/latent/), rbg真牛,先是DPM，然后是RCNN，保持了detection上的领先. 下面的结构是根据原文结构进行描述的。

## [INTRODUCTION]()

1. 介绍了[detection的难点]()，以及本模型主要应对的是[intraclass variability]()大的问题——[通过part的组合]()来适应这个变化。
2. 介绍了[使用的特征是HOG]()，是06年冠军使用的特征。后来HOG得到广泛的使用说明这确实是一个很好的人工设计的特征了。
3. 介绍了[Latent-SVM]()，一个[半凸的优化问题]()
4. 介绍了data-mining for [hard negative](http://blog.csdn.net/masibuaa/article/details/16113373)，因为背景太大，不可能全部用上，所以需要“[挖掘]()”到比较难的背景来训练分类器。

## [RELATE WORK]()

略

## [MODELS]()

总的来说，模型是若干个[线性滤波器]()组成的

$$
\sum_{x',y'}F[x',y'].G[x+x',y+y']
$$

(本质上跟卷积核一样，$F$理解成权重，$G$理解成特征）

为了能够应对多尺度的问题，使用了[特征金字塔](http://blog.csdn.net/qustqustjay/article/details/46786075)，另外，[part所在的层]()的分辨率是[root所在的层]()的分辨率的两倍。

<p align="center"><img src="http://i.imgur.com/WO1AJ28.png" width="400" ></p>

## [Deformable Part Models]()

[单个的DPM]()可以理解成是一个[root滤波器]()加入若干个[part滤波器]()，然后[减去part的形变损失]()。

$$
\text{score}(p_0,p_1,...,p_n)=\sum_{i=0}^nF_i'.\phi(H,p_i)-\sum_{i=1}^nd_i.\phi_d(dx_i,dy_i)+b
$$

上面的公式中:

1. [第一项]()代表了所有的滤波器（[root是0]()，[part从1到n共有n个]()，本质上上它们都是线性滤波器）;
2. 第二项是[形变损失]()（即模型的每个part节点都有一个标准位置，比如手在上半身而不是在下半身）;
      附带一提形变损失的计算公式是：
      $$
      (dx_i,dy_i)=(x_i,y_i)-(2(x_0,y_0)+v_i) \ (1)
      $$
      gives the displacement of the i-th part relative to its anchor position and : $\phi_d(dx,dy)=(dx,dy,dx^2,dy^2)$
3. 第三项是[bias]()，用于在[多个模型之间实现可比性]()。

上面公式(3)中的“2”是由于[root节点的分辨率只有part节点的一半]()，所以需要[映射到2倍大小]()。
公式(4)表明形变损失[考虑到了1范距离（曼哈顿距离）和2范距离（欧氏距离）]()。

## [Matching]()

当训练好一个模型时候，inference就是一个matching的问题了，即需要针对窗口给出分数：

$$
\text{score}(p_0)=\max_{p_1,...,p_n}\text{score}(p_0,...,p_n) \ (7)
$$

上式中的[p0~pn]()表示的是filter的位置。
穷举复杂度过高，所以需要用到动态规划，这里不展开了，作者给了一张大图

<p align="center"><img src="http://i.imgur.com/yKO7EoD.png" width="800" ></p>


（还是挺直观的，对于root和part的每个滤波器，都卷积出来一个map，然后叠加这些feature的结果）

## [Mixture Models]()

说明了如何从[单个model拓展到多个model]()。
原因是因为[单个model的描述能力不够]()——车有正视图、侧视图等。那么就可以根据[不同视角建立不同的model来表]()示。

## [LATENT SVM]()

由于本文的model都是[线性filter]()，所以其实可以将[整个model的所有的参数拉成一个长向量]()，这样就可以用常规的[线性优化方法]()来求解了。

<p align="center"><img src="http://i.imgur.com/Eiat9tE.png" width="600" ></p>

又由于[训练数据（VOC PASCAL）只有root的标注而没有part的标注]()，所以相对位置是未知的，需要作为[latent项]()进行学习，所以就要用到[Latent SVM]()（以下也遵循原文简称为LSVM）了。

## [Semi-convexity]()

作者提出了一个叫“[半凸]()”的说法，因为LSVM的损失函数：

- 对于[负样本]()是凸的
- 对于[正样本]()是非凸的，但是如果[固定住latent项]()，那么就变成[凸的]()了

**于是作者把这个情况叫做“半凸”**


## [Optimization]()

既然LSVM是半凸的，那么可以想到转换成[凸函数]()来进行求解，作者提出了一种叫坐标下降（[coordinate descent]()）的算法，分成两步走：

- 先对于[正样本]()优化[latent]()项，然后固定住，那么损失函数就变凸了;
- 然后用SVM的[常规优化方法]()来进行优化即可

迭代重复上面两个步骤，就可以实现LSVM的优化了。
注意到，**这里只是针对正样本的latent项单独优化并且fix住，而负样本的优化是没有fix住latent项的**，作者给出的解释是：

<p align="center"><img src="http://i.imgur.com/cK4GC7q.png" width="400" ></p>

## [Stochastic gradient descent]()

这个章节讨论的是上面坐标下降中第二步的优化，即[求解SVM部分]()。

注意到这里面正样本的latent已经被固定住了，所以[损失函数已经变成了凸函数，可以用常规方法求解]()。

一般求解SVM可以用[二次规划的方法]()，作者这里用的是随机梯度下降（SGD）。至于为什么，倒没有说明。

梯度的计算公式：
$$
\begin{cases}
\nabla L_D(\beta)=\beta + C\sum_{i=1}^n h(\beta,x_i,y_i) \ \ (16)\\
h(\beta,x_i,y_i)=\begin{cases}0, \text{ if }y_if_{\beta}(x_i)\ge 1 \\ -y_i\Phi(x_i,z_i(\beta)), \text{ otherwise }\end{cases} \ \ (17)
\end{cases}
$$

这里对于(17)可以给出较为直观的解释:
- 当$y_if_{\beta}(x_i)\ge 1$时候，其实也就是样本被[正确分类了]()，所以对应部分的梯度为β+0
- 否则，样本类别错了，或者类别对了但是落在了分隔面之内，那么就需要更新([β+对应的梯度]())了。

更为具体的过程是：
<p align="center"><img src="http://i.imgur.com/GNXRWNf.png" width="500" ></p>

具体解释一下：
1) 设置学习率，一般设置成1/t就可以了。
2) 选择随机的一个样本
3) 求解最佳隐变量
4)和5) 都是更新梯度了。

另外需要注意到，作者没有使用[mini-batch的下降法](http://blog.csdn.net/llx1990rl/article/details/44001921)，而是用是单个样本的梯度（直接乘上一个n）来估计全体样本

## [Data-mining hard examples, SVM version]()

上文也提到了，由于负样本太大，无法全部用上，所以需要挖掘出hard negative来帮助分类。
虽然说得data-mining，但实际上思路很简单。

首先，定义[easy negative]()和[hard negative]()为：

<p align="center"><img src="http://i.imgur.com/E5pLuBz.png" width="600" ></p>

其实，就是[能够正确分类的就是easy negative]()，[不能正确分类或者分类正确但是在决策面之内的叫做hard negative]()
具体的过程，如下图：

<p align="center"><img src="http://i.imgur.com/20M9spa.png" width="600" ></p>

简单解释下。
1) 选择一个子集C，训练SVM得到参数β
3) 排除掉C中的easy negative
4) 填充hard negative到C
2) 是结束条件，如果没有可以添加的hard negative，就退出。

后面是算法有效性的证明，跳过了。

## Data-mining hard examples, LSVM version

LSVM的Data-mining算法是类似的。
正样本的latent项被固定住后，本质上也是一个凸优化问题。
通常地，通过维护一个cache，迭代地排除easy negative，加入hard negative。

## TRAINING MODELS
## Learning parameters

这里说了几件事：

1. 如何生成正样本：根据gt bbox生成，控制latent项的overlap在50%之内
2. 如何生成负样本：没看懂，感觉是在小范围内密集地采样
3. 训练过程的伪代码：基本是将上述过程（搜索并固定正样本的latent项，data-mining）简述了一遍

## Initialization

这里提到了对于有latent项的模型需要较好地初始化，不然会陷入局部最优（略没懂，不是说latent SVM是半凸函数吗？）
大致分成3个步骤：

1. Initialzation Root Filters:
  根据aspect ratio分组，来训练mix model
  使用一个线性SVM进行权值的训练
2. Merging Components
  没看懂，大致是聚类，将相同的component聚在一起
3. Initailzing Part Filters
  使用启发的贪心算法来寻找能量最大的位置（该位置的中心限制在中轴线或者以中轴线对称）

<p align="center"><img src="http://i.imgur.com/1S6gAYy.png" width="600" ></p>

## FEATURES
略，介绍了HOG，和做了PCA

## POST PROCESSING
略，包括了，bbox回归，NMS，Context

## EMPIRICAL RESULTS
略，当时VOC的state-of-the-art

## DISCUSSION
略
