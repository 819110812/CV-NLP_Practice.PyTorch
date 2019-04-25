# [Effective approaches to attention-based neural machine translation](https://arxiv.org/abs/1508.04025)

<center><img src="https://images2015.cnblogs.com/blog/670089/201610/670089-20161012111501343-1669960587.png"></img></center>

## 回顾
首先，我们知道基本的encoder-decoder编码器，将输入句子 $\mathbf{X}$ 进行编码得到一个向量 $c$, 然后对$c$进行解码得到输出序列 $\mathbf{Y}$. 具体做法见下图:

<center><img src="https://pic4.zhimg.com/80/v2-a2872599992a0d3317b7faafb32d3df4_hd.jpg"></img></center>

然后，有paper对此进行了改进，最著名的就是引入了attention机制了，详细的可以见这篇文章《[Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/abs/1409.0473)》。

<center><img src="https://pic1.zhimg.com/80/v2-894bc7e1f247625eb3fd292e8d284884_hd.jpg"></img></center>
相比于之前的encoder-decoder模型，attention模型最大的区别就在于<font color="Red">它不在要求编码器将所有输入信息都编码进一个固定长度的向量之中</font>。相反，此时编码器需要将输入编码成一个向量的序列，<font color="Red">而在解码的时候，每一步都会选择性的从向量序列中挑选一个加权后的向量进行进一步处理</font>。这样，在产生每一个输出的时候，都能够做到充分利用输入序列携带的信息。而且这种方法在翻译任务中取得了非常不错的成果。


## 介绍
这篇论文一篇很具代表性的论文，他们的工作告诉了大家attention在RNN中可以如何进行扩展，这篇论文对后续各种基于attention的模型在NLP应用起到了很大的促进作用。在论文中他们提出了两种attention机制，一种是全局（global）机制，一种是局部（local）机制。

![](http://cnyah.com/2017/08/01/attention-variants/attention-mechanisms.png)

### Global attention

<center><img src="https://images2015.cnblogs.com/blog/670089/201610/670089-20161012111506078-902266845.png"></img></center>

首先我们来看看global机制的attention，其实这和上一篇论文提出的attention的思路是一样的，它都是对源语言对所有词进行处理，不同的是在计算attention矩阵值的时候，他提出了几种简单的扩展版本。这里我假设大家数学Bahdanau etal的工作，我们直接给出这篇文章Global attention的公式 (Luong etal, 2015)：
$$
\begin{align}
\mathbf{a}_t &= \text{align}(h_t, \bar{h}_t)=\frac{\exp (\text{score}(h_t, \bar{h}_s))}{\sum_{s'}\exp (\text{score}(h_t, \bar{h}_{s'}))} \qquad \text{Luong etal, 2015}\\
\mathbf{a}_{i,j} &= \frac{\exp (e_{ij})}{\sum_{k=1}^{T_x}\exp (e_{ik})}, e_{ij}=v_a^T \tanh (\mathbf{W}_a s_{i-1}+U_a h_j) \qquad \text{Bahdanau etal, ICLR 2015}
\end{align}
$$

我们看出，最明显的一点是计算 $e_{ij}$ 变成了 $\text{score}(h_t, \bar{h}_s)$.

所以，Global attention的思想是计算source端上下文向量$c_i$时，考虑Encoder (source)的所有隐藏状态（$h_1,\cdots, h_T$）。其间的对应权重（align probability）通过比较当前<font color="Red">目标端隐层状态 $\bar{h}_s$</font> 和<font color="Red">source端的hidden state $h_t$</font>得到。注意，<font color="green">在Bahdanau的版本中，他们只有source的hidden state!</font>

至于计算 $\text{score}$ 的方法，他们给出了3种：
$$
\text{score}(h_t, \bar{h}_t)=
\begin{cases}
h_t^T\bar{h}_s, \qquad \text{直接点乘}\\
h_t^T \mathbf{W}_a \bar{h}_s, \qquad\text{加入一个Weights, 结果看，这个方式最好！}\\
\mathbf{W}_a[h_t;\bar{h}_s], \qquad\text{先合并向量，然后做mapping}
\end{cases}
$$

### Local attention

<center><img src="https://images2015.cnblogs.com/blog/670089/201610/670089-20161012111507500-812049044.png"></img></center>

Global attention的思想是计算每个目标端词和每个源语言词端的对齐概率，这也许会称为一种缺点，尤其针对长句子，这种方法的代价很大。
因此，有了一种折中的方法，来自xu etal中soft+hard.

局部attentio对 $t$ 时刻的输出生成一个它在源语言端的对齐位置 $p_t$，接着在源语言端取窗口 $[p_t-D,p_t+D]$，上下文向量 $c_t$ 则通过计算窗口内的hidden state的加权平均得到。至于这个对齐位置 $p_t$ 如何确定，他们定义了2种：local-m和local-p。

1. Monotonic alignment (local-m)

就是简单设置 $p_t=t$

2. Predictive alignment (local-p)

针对每个目标端输出，预测它在源语言端的对齐位置，计算公式为：
$$
p_t=S\cdot \text{sigmoid} (v_p^T \tanh (\mathbf{W}_p h_t))
$$
其中，$\mathbf{W}_p$ 和 $v_p$ 都是模型参数。最后文章引入一个服从于 $\mathcal{N}(p_i, D/2)$ 的高斯分布来设置对齐权重，因为直觉上，离对齐位置 $p_t$ 距离越近，对后续决策的影响越大。那么目标端位置 $t$ 与源语言端位置 $s$（在窗口内）的对齐概率计算如下：
$$
\mathbf{a}_t(s)=\text{align}(h_t,\bar{h}_s)\exp(-\frac{(s-p_t)^2}{2\sigma^2})
$$
Align函数的定义与softmax定义类似 (见上文)。<font color="Red">作者的实验结果是局部的比全局的attention效果好</font>。


## 实验结果
![](https://ai2-s2-public.s3.amazonaws.com/figures/2017-08-08/1b1df9f75ee6f27433687dad302387f811dab64d/6-Table1-1.png)
