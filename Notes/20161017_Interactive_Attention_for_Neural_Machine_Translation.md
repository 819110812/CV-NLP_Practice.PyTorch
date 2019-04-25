# [Interactive Attention for Neural Machine Translation](https://arxiv.org/pdf/1610.05011v1.pdf)

在传统的基于attention机制的NMT模型（详细介绍可见《Modeling Coverage for Neural Machine Translation简读》）中，在翻译过程开始后，输入序列的隐藏层状态是保持不变的，在每个时间步发生改变的只是alignment model（也就是attention的权重）。本文提出了一种新的attention机制，在更新attention权重的时候加入了上一次的输出的信息，另外，还通过“读写”机制不断地更新输入序列的隐藏层状态，完成翻译过程与输入序列的表示之间信息的交互（Interactive），也收到了更好的效果。


在这篇简述中，按照原文的顺序，依次对传统的attention model、作者改进的attention model以及作者重点提出的Interactive Attention model进行介绍，最后，会给出各个模型实验结果的对比。

1. Conventional attention model

<center><img src="https://pic4.zhimg.com/v2-352b9d350a584d2c5ee5493609958867_b.png"></img></center>

$c_t$表示时间$t$时输入文本的表示向量，是encoder侧top states的加权累积：$c_t=\sum_{j=1}^Na_{t,j}h_j$

其中，attention权重向量的计算如下：$a_{t,j}=\frac{e^{e_{t,j}}}{\sum_{k=1}^Ne^{e_{t,k}}}$

$s_t$表示的是时间t时的解码单元的状态, $s_t=g(s_{t-1},y_{t-1},c_t)$

而$P(y_t|y_{<t},x)=softmax(g(y_{t-1},s_t,c_t))$,通过softmax层得到词表中各个单词的解码概率；

2. Improved Attention Model

<center><img src="https://pic3.zhimg.com/v2-1328ca7010df6a369c76f8b91fff712e_b.png"></img></center>

与传统attention机制的变化就是在更新attention权重的时候加入了上一次的输出$y_{t-1}$的信息，

$$
e_{t,j}=\mathbf{V}_a^T\tanh (\mathbf{W}_a\tilde{s}_{t-1}+U_ah_j)
$$

其中，中间状态表示为：$\tilde{s}_{t-1}=GRU(s_{t-1},e_{y_1-1})$. $e_{y_{t-1}}$表示上一步输出$y_{t-1}$的词向量；而状态$s_t$
的更新公式变为：$s_t=GRU(\tilde{s}_{t-1},c_t)$. 其他环节与传统attention模型保持一致；

3. Interactive Attention Model

<center><img src="https://pic1.zhimg.com/v2-490fad178e0f7fe3ba1a01cd397d7ff4_b.png"></img></center>

$H=\{h_1,h_2,...,h_N\}$表示输入序列$x=\{x_1,x_2,...,x_N\}$的隐层状态，由于H会随着翻译过程的进行发生变化，因此，我们使用$H^{(t)}$ 表示时刻t时的$H$,而具体的细胞状态表示为$h_{j}^{(t)}$ .新的attention机制下的翻译过程可以总结为以下过程：与上边的improved model中一样，在更新$c_t$时加入前一次输出的信息，

$$
\begin{aligned}
\tilde{s}_{t-1}&=GRU(s_{t-1},e_{y_{t-1}})\\
c_t&=Read(\tilde{s}_{t-1},H^{(t-1)})
\end{aligned}
$$

仔细看一下，不难发现$s_t$的更新所利用的信息与传统方法基本是一致的：$s_t=RGU(\tilde{s}_{t-1},c_t)$. 更新状态$s_t$之后要对$H^{(t)}$ 进行更新，$H^{(t)}=Write(s_t,H^{(t-1)})$


接下来讨论具体的Read和Write机制：
### Attention Read

在更新$H^{(t)}$ 的时候，加入了Forget和Update机制：
与LSTM的遗忘门一样，Forget决定丢弃哪些信息，

$$
\tilde{h}_i^{(t)}=h_i^{(t-1)}(1-w_t^{W}(i).F_t),i=1,2,\cdots, n
$$

其中，$F_t=\sigma (W_F,s_t),F_t\in R^{m}$ ,而$W_F\in R^{m\times m}$ ; $\sigma$ 表示sigmoid函数，$W_t^{W}\in R^{n}$ ,表示修改$H^{(t)}$时输入序列相应的归一化权重；而类似于LSTM的输入门，Update决定加入哪些新的信息，

$$
h_i^{(t)}=\tilde{h}_i^{(t)}+w_t^W(i).U_t,i=1,2,\cdots, n
$$

其中，$U_t=\sigma (W_U,s_t),U_t\in R^{m} ,W_U\in R^{m\times m}$ ;在作者进行实验的时候，$W_t^{W}$和$W_t^{R}$共享同一组参数；

## 简评：

跟《Modeling Coverage for Neural Machine Translation》这篇paper一样，作者抓住了关键问题，即在翻译过程中输入序列的表示没有充分利用翻译过程中已有的信息，是固定不变的，attention权重的更新也没有充分利用已翻译的信息；作者通过“读写”机制巧妙地解决了这个问题，效果上有很明显的提升，是一篇好文章，值得学习。
