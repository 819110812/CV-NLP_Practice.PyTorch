# [Attention is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

该 paper 可以算作是 Google 针对 Facebook 之前的 CNN seq2seq 的回应(我的工作其实比他们早几个月，可惜没人关注 😂)。整体而言，这个工程性较强，主要目的是在减少计算量和提高并行效率的同时不损害最终的实验结果。

<center><img src="https://pic4.zhimg.com/80/v2-ea808f587add15d09079448390f01f06_hd.jpg"></img></center>

## 创新点
1. (masked) self-attention。attention 可以不只是用于 decoder 里每步输入一个符号，而是可以用在网络中的任意一层，把一个序列转换成另一个序列。这个作用与 convolutional layer、recurrent layer 等是类似的，但有一个好处就是不再局限于局域性。attention 可以一般地描述为在一个 key-value mapping 中进行检索，只不过 query 跟 key 可以进行模糊匹配，检索结果是各个 key 对应的 value 的加权平均。
2. position embedding去保留全句的位置信息

## 模型结构

### Encoder 部分

6 个 block，每个 block 中有两层，他们分别是 Multi-head self attention 和 Position-wise feed forward。

1. Multi-head self attention

这里attention有3个输入：query，keys，values。选择三个输入考虑到模型的通用性。输出是value的加权求和，value的权重来自于query和keys的乘积，经过一个softmax得到。可以得到scaled dot-product attention的公式为：
$$
\text{Attention}(Q,K,V)=\text{softmax}(QK^T/\sqrt{d_k})V
$$
其中 $Q$ 与 $K$ 均为输入，$V$ 为learned value。这里$d_k$是用来调节的。至于multi-head， 则是对于$K,V,Q$输入，采用不同的权重，连续进行H词Scaled dot-product attention，类似于卷积网络里面采用不同的卷积核多次进行卷积操作。同时，对每个维度多了reduce，使multi-head attention总的计算量与single-head attention一致。

2. Add&Norm (AN)单元

Add&Norm的实现为： $\text{Add&Norm}=\text{LayerNorm}(x+\text{Sublayer}(x))$ 。这里的<font color="Red">$\text{Sublayer}(x)$</font>是前面部分（Multi-head attention或者FeedForward layer）的输出。这样连接有2个好处：（1）训练速度加快：LayerNorm是batch normalization的一个变体。它和BN的区别是：LN是本次输入模型的一组样本进行Normalization， BN是对一个batch数据进行normalization。因此，LN可以用于RNN规范化操作。 (2)映入了残差，尽可能保留原始输入$x$信息。

3. 逐项的feed-forward网络作用
Attention的sublayer之间嵌入了一个FFN层，两个线性变换组成：$\text{FFN}(x)=\max (0,x\cdot \mathbf{W}_1+b1)W_2+b_2$。 同层拥有相同的参数，不同层之间拥有不同的参数，目的是提高模型特征提取的能力。

4. Position encoding
模型的输入embeddings加入了Position embedding，使网络可以获取输入序列的位置之间的相对or绝对位置信息。Position embedding有很多方式，本文采用了简单的方式，基于sin和cos函数，根据pos和维度 $i$ 来计算：
$$
\begin{align}
PE_{\text{pos}, 2i}=\sin (\text{pos}/10000^{2i/d_{\text{model}}})\\
PE_{\text{pos}, 2i+1}=\cos (\text{pos}/10000^{2i/d_{\text{model}}})
\end{align}
$$
这样做的目的是因为正弦和余弦函数具有周期性，对于固定长度偏差$k$（类似于周期），$\text{psot+k}$ 位置的PE可以表示成关于pos位置PE的一个线性变化（存在线性关系），这样可以方便模型学习词与词之间的一个相对位置关系。

### Decoder 部分
1. Multi-head self attention (with mask) 与 encoder 部分相同，只是采用 0-1mask 消除右侧单词对当前单词 attention 的影响。
2. Multi-head attention(with encoder) 引入 encoder 部分的输出在此处作为 multi-head 的其中几个 head。
3. Position-wise feed forward network 与encoder部分相同。

some tricks during training process：(1) residual dropout; (2) attention dropout; (3) label smoothing
