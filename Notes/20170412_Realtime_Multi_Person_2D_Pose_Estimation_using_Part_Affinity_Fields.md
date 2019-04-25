# [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf)

Pose Estimation的三个关键性挑战：
1. 我们不知道一张图像中包含多少人，这些人的姿态和尺寸如何也都是未知的；
2. 多个人体之间的接触、遮挡等使得情况更加复杂；
3. 是实时性的要求，随着图像中人数的增多计算也越来越复杂。

> 现有的一些方法存在的问题：现有的一些方法主要由person detector和single-person pose estimation两部分组成，这些top-down的方法在性能上很依赖于这两个部分，如果person detector检测失败了（当多个人挨得很近的时候），这些方法可能就不奏效了。另外一个就是时间的问题，这些方法都没有一个很好的处理。

## Simultaneous Detection and Association

![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_2.png)

上图给出了模型的整个处理过程：
1. 将一张尺寸为 $w\times h$的图像输入进去，然后模型同时得到人体部位位置的confidence maps集合 $S$ 和一个用来说明关节点连接关系的part affinities集合 $L$; 其中 $S=(S_1,S_2,\cdots, S_J)$ 中包含J个confidence map，每个表示一个关键点，$S_j \in \mathbb{R}^{\mathit{w} \times \mathit{h}}, j \in [1…J]$. 集合 $L=(L_1, L_2, …, L_C)$ 中包含C个vector field，每一个都表示一个limb（即两个关键点之间的连线），其中 $L_c \in \mathbb{R}^{\mathit{w} \times \mathit{h} \times 2}, c \in [1…C]$, $L_c$ 中的每个图像位置都编码成一个2D的向量。
2. 最后，confidence maps和affinity fields被拿来推理出greedy inference，得到一张图像中所有人的2D关键点。

![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_3.png)
上图给出了模型示意图: <u>图像输入进去，然后同时预测出confidence maps和affinity fields。网络分成两个部分，上边米色的那部分预测出confidence maps，下边蓝色的那部分预测出affinity fields。每个分支都是一个迭代预测结构，整个模型包含了T个stage，每个stage都加入中间监督（intermediate supervision）</u>。

1. 图像先经过微调过的VGG19的前十层得到一组feature maps F，将其输入到每一个分支的第一个stage中;
2. 在第一个stage中，网络输出一组detection confidence maps $S^1 = \rho^1(F)$ 和一组part affinity fields $L^1 = \phi^1(F)$, 其中 $\rho^1$ 和 $\phi^1$ 表示第一个stage的CNN结构;
3. 在随后的每个stage中，我们将前一个阶段的输出和F给concatenate到一块儿输入进去，输出的是refined predictions。

$$
S^t = \rho^t(F, S^{t-1}, L^{t-1}), \forall t \geq 2, \qquad \qquad \qquad \nonumber{(1)}
$$

$$
L^t = \phi^t(F, S^{t-1}, L^{t-1}), \forall t \geq 2, \qquad \qquad \qquad \nonumber{(2)}
$$
4. 我们使用L2损失来评估每个阶段的预测结果，下面给出了损失函数的公式，
$$
f^{t}_{S} = \sum\limits_{j=1}^{J} \sum_{p}W(p) \cdot \left|S_{j}^{t}(p) - S_{j}^{*}(p)\right|_{2}^{2}, \qquad \qquad \qquad \nonumber{(3)}
$$
$$
f^{t}_{L} = \sum\limits_{c=1}^{C} \sum_{p}W(p) \cdot \left|L_{c}^{t}(p) - L_{c}^{*}(p)\right|_{2}^{2}, \qquad \qquad \qquad \nonumber{(4)}
$$
其中 $S_{j}^{*}$ 是groundtruth part confidence map， $L_{c}^{*}$ 是groundtruth part affinity vector field，W是一个二进制标志（当图像中位置p的标注数据缺失时 $W(p)=0$ ，避免在训练过程中惩罚true positive predictions）。中间监督通过阶段性地补充梯度，能够有效得解决一部分梯度消失的问题。整个模型的目标为：$f = \sum_{t=1}^{T}(f_S^t + f_L^t)$。

## Confidence Maps for Part Detection
下边给出根据标注数据计算groundtruth confidence maps $S^{\ast}$ 的方法，每个confidence map都是一个2D表示。

- 理想情况下，当图像中只包含一个人时，如果一个关键点是可见的话对应的confidence map中只有一个峰值；
- 当图像中有多个人时，对于每一个人k的每一个可见关键点j在对应的confidence map中都会有一个峰值。

首先给出每一个人k的单个confidence maps $S_{j,k}^*$, $x_{j,k} \in \mathbb{R}^2$ 表示图像中人 $k$ 对应的位置 $j$ 对应的groundtruth position，数值如下所示，
$$
S_{j,k}^{*}(p) = exp(- \frac{|p-x_{j,k}|^2_2}{\sigma^2}), \qquad \qquad \qquad \qquad \qquad \nonumber{(6)}
$$
其中 $\sigma$ 用来控制峰值在confidence map中的传播范围。对应多个人的confidence map见公式如下：
$$
S_{j}^{*}(p) = \max_{k}S_{j,k}^{*}(p), \qquad \qquad \qquad \qquad \qquad \qquad \nonumber{(7)}
$$
这里用最大值而不是平均值能够更准确地将同一个confidence map中的峰值保存下来。

## Part Affinity Fields for Part Association
![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_5.png)

给定一组关键点，如Figure 5(a)所示，我们如何把它们组装成未知数量的人的整个身体的pose呢？

我们需要一个置信方法来确定每队关键点之间的连接，即它们属于同一个人。一个可能的方法是找到一个位于每一对关键点之间的一个中间点，后检查中间点是真正的中间点的概率，如Figure 5(b)所示。

但是当人们挤在一块儿的时候，通过这样的中间点可能得出错误的连接线，如Figure 5(b)中绿线所示。出现这种情况的原因有两个：(1)<u>这种方式只编码了位置信息，而没有方向</u>；(2)<u>躯体的支撑区域已经缩小到一个点上</u>。

为了解决这些限制，作者提出了称为<b>part affinity fields的特征</b>表示来保存躯体的支撑区域的位置信息和方向信息，如Figure 5(c)所示。对于每一条躯干来说，<u>the part affinity是一个2D的向量区域</u>。在属于一个躯干上的每一个像素都对应一个2D的向量，这个向量表示躯干上从一个关键点到另一个关键点的方向。

考虑下图中给出的一个躯干（手臂），令 $X_{j_1,k}$ 和 $x_{j_2,k}$ 表示图中的某个人k的两个关键点 $j_1$ 和 $j_2$ 对应的真实像素点，如果一个像素点p位于这个躯干上，值 $L_{c,k}^{\ast}(p)$ <font color="Red">表示一个从关键点 $j_1$ 到关键点 $j_2$ 的单位向量，对于不在躯干上的像素点对应的向量则是零向量</font>。下面这个公式给出了the groundtruth part affinity vector，对于图像中的一个点p其值 $L_{c,k}^{*}(p)$ 的值如下：
$$L_{c,k}^{*}(p) = \begin{cases}
v,\qquad \text{if p on limb c, k } \\
0,\qquad \text{otherwise}
\end{cases} \qquad \qquad \qquad \qquad \qquad \qquad \nonumber{(8)}
$$

![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_chatu.png)

其中，$v = (x_{j_2,k} - x_{j_1,k})/|x_{j_2,k} - x_{j_1,k}|_2$ 表示这个躯干对应的单位方向向量。属于这个躯干上的像素点满足下面的不等式，其中 $\sigma_{l}$ 表示像素点之间的距离，躯干长度为 $l_{c,k} = |x_{j_2,k}-x{j_1,k}|_2$ ， $v_{\perp}$ 表示垂直于v的向量。并且有：$0 \leq v \cdot (p-x_{j_1,k}) \leq l_{c,k} \text{ and } |v_{\perp} \cdot (p-x_{j_1,k})| \leq \sigma_l$.

整张图像的the groundtruth part affinity field取图像中所有人对应的affinity field的平均值，其中 $n_c(p)$ 是图像中k个人在像素点p对应的非零向量的个数。

$$
L_{c}^{*}(p) = \frac{1}{n_{c}(p)}\sum_{k}L_{c,k}^{*}(p) \qquad \nonumber{(9)}
$$

在预测的时候，我们用候选关键点对之间的PAF来衡量这个关键点对是不是属于同一个人。具体地，对于两个候选关键点对应的像素点 $d_{j1}$ 和 $d_{j_2}$ ，我们去计算这个PAF，如下式所示。
$$
E = \int_{u=0}^{u=1}L_{c}(p(u)) \cdot \frac{d_{j_2}-d_{j_1}}{|d_{j_2}-d_{j_1}|_2}du \qquad \nonumber{(10)}
$$
其中，$p(u)$表示两个像素点 $d_{j_1}$ 和 $d_{j_2}$ 之间的像素点:
$$
p(u) = (1-u)d_{j_1} + ud_{j_2} \qquad \qquad \nonumber{(11)}
$$

## Multi-Person-Parsing-using-PAFs
借助非最大抑制，我们从预测出的confidence maps得到一组离散的关键点候选位置。因为图像中可能有多个人或者存在false positive，每个关键点可能会有多个候选位置，因此也就组成了很大数量的关键点对，如Figure 6(b)所示。按照公式(10)，我们给每一个候选关键点对计算一个分数。从这些关键点对中找到最优结果，是一个NP-Hard问题。下面给出本文的方法。

![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_6.png)

假定模型得到的所有候选关键点构成集合 $D_J=\{d_{j}^{m}: for \, j \in \{1...J\}, m \in \{1...N_{j}\}\}$ ，其中$N_j$表示关键点j的候选位置数量，$d_j^m \in \mathbb{R}^2$是关键点j的第m个候选位置的像素坐标。

我们需要做的是将属于同一个人的关键点连成躯干（胳膊，腿等），为此我们定义变量 $z_{j_1j_1}^{mn} \in \{0, 1\}$ 表示候选关键点 $d_{j_1}^m$ 和 $d_{j_2}^n$ 是否可以连起来。如此以来便得到了集合 $Z=\{z_{j_1j_2}^{mn}: for \, j_1,j_2 \in \{1...J\},m\in\{1...N_{j_1}\},n\in\{1...N_{j_2}\}\}$ 。现在单独考虑第c个躯干如脖子，其对应的两个关键点应该是 $j_1$　和   $j_2$ ，这两个关键点对应的候选集合分别是　$D_{j_1}$ 和 $D_{j_2}$ ，我们的目标如下所示。
$$\max_{Z_c}E_c = \max_{Z_c}\sum_{m \in D_{j_1}}\sum_{n \in D_{j_2}}E_{mn} \cdot z_{j_1j_2}^{mn}, \qquad \nonumber{(12)}$$

$$s.t. \qquad \qquad \forall m \in D_{j_1}, \sum_{n \in D_{j_2}}z_{j_1j_2}^{mn} \leq 1, \qquad \nonumber{(13)}$$

$$\qquad \qquad \qquad \forall n \in D_{j_2}, \sum_{m \in D_{j_1}}z_{j_1j_2}^{mn}\leq1, \qquad \nonumber{(14)}$$
其中，$E_c$ 表示躯干c对应的权值总和，$Z_c$ 是躯干c对应的Z的子集，$E_{mn}$ 是关键点 $d_{j_1}^m$ 和 $d_{j_2}^n$ 对应的part affinity，公式(13)和公式(14)限制了任意两个相同类型的躯干（例如两个脖子）不会共享关键点。问题扩展到所有C个躯干上，我们优化目标就变成了公式(15)。

$$\max_{Z}E = \sum_{c=1}^{C}\max_{Z_c}E_c, \qquad \nonumber{(15)}$$

## Results
在预测的时候，论文中使用了同一张图像的3个尺寸($\times0.7,\times1,\times1.3$)输入进去来得到比单个尺寸更好的结果。
![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_table_1_2.png)
![](http://7xsbsy.com1.z0.glb.clouddn.com/PAF_table_3.png)
