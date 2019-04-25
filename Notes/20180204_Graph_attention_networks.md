# [Graph attention networks](https://arxiv.org/pdf/1710.10903.pdf)

针对图结构数据，本文提出了一种GAT（graph attention networks）网络。该网络使用masked self-attention层解决了之前基于图卷积（或其近似）的模型所存在的问题。在GAT中，图中的每个节点可以根据邻节点的特征，为其分配不同的权值。GAT的另一个优点在于，无需使用预先构建好的图。因此，GAT可以解决一些基于谱的图神经网络中所具有的问题。实验证明，GAT模型可以有效地适用于（基于图的）归纳学习问题与转导学习问题。

# Model
## Graph Attentional Layer
单个的 graph attentional layer 的输入是一个节点特征向量集：$\mathbf{h}=\{\overrightarrow{h}_1,\overrightarrow{h}_2, \cdots, \overrightarrow{h}_N\}, \overrightarrow{h}_i\in \mathbb{R}^F$, 其中， $N$表示节点集中节点的个数， $F$表示相应的特征向量维度。

每一层的输出是一个新的节点特征向量集：$\mathbf{h}'=\{\overrightarrow{h}_1',\overrightarrow{h}_2', \cdots, \overrightarrow{h}_N'\}, \overrightarrow{h}_i'\in \mathbb{R}^F$. 其中， F' 表示新的节点特征向量维度（可以不等于 F ）。

一个graph attention layer的结构如下图所示：

![](https://pic3.zhimg.com/80/v2-526634b065899482bbe9811af105ab73_hd.jpg)

具体来说，graph attentional layer首先根据输入的节点特征向量集，进行self-attention处理：$e_{i,j}=a(\mathbf{W}\overrightarrow{h}_i,\mathbf{W}\overrightarrow{h}_j)$.

其中，a 是一个 $\mathbb{R}^{F'}\times\mathbb{R}^{F'}\to\mathbb{R}$ 的映射， $W\in\mathbb{R}^{F'\times F}$ 是一个权值矩阵（被所有 $\vec{h}_{i}$ 所共享）。一般来说，self-attention会将注意力分配到图中所有的节点上，这种做法显然会丢失结构信息。为了解决这一问题，本文使用了一种masked attention的方式——仅将注意力分配到节点 i 的邻节点集上，即 $j\in\mathcal{N}_{i}$ （在本文中，节点 $i$ 也是 $\mathcal{N}_{i}$ 的一部分）：

$$
\alpha_{ij}=\text{softmax}_j(e_{ij})=\frac{\exp(e_{ij})}{\sum_{k\in \mathcal{N}_i}\exp (e_{ik})}
$$

在本文中， $a$ 使用单层的前馈神经网络实现。

本文使用了很大的篇幅将GAT与其他的图模型进行了比较：
1. GAT是高效的。相比于其他图模型，GAT无需使用特征值分解等复杂的矩阵运算。单层GAT的时间复杂度为 O(|V|FF'+|E|F') （与GCN相同）。其中，  |V| 与 |E| 分别表示图中节点的数量与边的数量。
2. 相比于GCN，每个节点的重要性可以是不同的，因此，GAT具有更强的表示能力。
3. 对于图中的所有边，attention机制是共享的。因此GAT也是一种局部模型。也就是说，在使用GAT时，我们无需访问整个图，而只需要访问所关注节点的邻节点即可。这一特点的作用主要有：（1）可以处理有向图（若 $j\to i$ 不存在，仅需忽略 $\alpha_{ij}$ 即可）；（2）可以被直接用于进行归纳学习。
4. 最新的归纳学习方法（GraphSAGE）通过从每个节点的邻居中抽取固定数量的节点，从而保证其计算的一致性。这意味着，在执行推断时，我们无法访问所有的邻居。然而，本文所提出的模型是建立在所有邻节点上的，而且无需假设任何节点顺序。
5. GAT可以被看作是MoNet的一个特例。具体来说，可以通过将伪坐标函数（pseudo-coordinate function）设为 $u(x,y)=f(x)||f(y)$ ，其中， f(x) 表示节点 x 的特征， || 表示连接符号；相应的权值函数则变成了 $w_{j}(u)=\text{softmax}(MLP(u))$ 。

## Experiments
本文的实验建立在四个基于图的任务上，这些任务包括三个转导学习（transductive learning）任务以及一个归纳学习（inductive learning）任务。具体如下：

### Transductive Learning

在转导学习任务中，使用了三个标准的引证网络数据集——Cora、Citeseer与Pubmed。在这些数据集中，节点对应于文档，边（无向的）对应于引用关系。节点特征对应于文档的BoW表示。每个节点拥有一个类别标签（在分类时使用softmax激活函数）。每个数据集的详细信息如下表所示：

![](https://pic3.zhimg.com/80/v2-b0b18a2cf53f26fdcf5ec8e82962659c_hd.jpg)

### Inductive Learning

对于归纳学习，本文使用了一个蛋白质关联数据集（protein-protein interaction, PPI），在其中，每张图对应于人类的不同组织。此时，使用20张图进行训练，2张图进行验证，2张图用于测试。每个节点可能的标签数为121个，而且，每个节点可以同时拥有多个标签（在分类时使用sigmoid激活函数）。

归纳学习的实验结果如下表所示，可以看到，GAT模型的效果要远远优于其他模型。
![](https://pic2.zhimg.com/80/v2-837773a898ae95e49843d5c5ce54f7af_hd.jpg)

### Conclusion
本文提出了一种基于self-attention的图模型。总的来说，GAT的特点主要有以下两点：
1. 与GCN类似，GAT同样是一种局部网络。因此，（相比于GNN或GGNN等网络）训练GAT模型无需了解整个图结构，只需知道每个节点的邻节点即可。
2. GAT与GCN有着不同的节点更新方式。GCN使用的是GAT使用self-attention为每个邻节点分配权重，也就是说，GAT的节点更新方式与以下是一个具体的示例。假设有三个节点，每个节点使用二维向量进行表示，则两种网络对应于以上运算。通过对比可以发现，GAT在计算新的节点表示时，相比于GCN，多引入了一个权值矩阵（可以看成将原先的 $A$ 修改成了 $A^{new}$)。


## Reference
- [《Graph Attention Networks》阅读笔记](https://zhuanlan.zhihu.com/p/34232818)