# [Neural Machine Translation by Jointly Learning to Align and Translate](http://arxiv.org/abs/1409.0473)

TLDR; The authors propose a novel "attention" mechanism that they evaluate on a Machine Translation task, achieving new state of the art (and large improvements in dealing with long sentences). Standard seq2seq models typically try to encode the input sequence into a fixed length vector (the last hidden state) based on which the decoder generates the output sequence. However, it is unreasonable to assume the all necessary information can be encoded in this one vector. <b><font color="red">Thus, the authors let the decoder depend on a attention vector, which based on the weighted sum (expectation) of the input hidden states. The attention weights are learned jointly, as part of the network architecture.</font></b>

# 方法
## 基本的encoder-decoder结构NMT方法
<center><img src="https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2017/10/Depiction-of-Sutskever-Encoder-Decoder-Model-for-Text-Translation.png"></img></center>

在基本encoder-decoder方法中，encoder读一个输入句子，$\mathbf{x}=\{x_1, \cdots, x_{T_x}\}$， 其中每个$x_i=\mathbf{W}_e X_i$是经过word embedding的结果（$\mathbf{W}_e$是要学到的embedding weights, 对应Pytorch，则是nn.Embeddings(.)的参数）。

有了输入的word vectors $\mathbf{x}$，我们先考虑常用的one-directional RNN, 传播公式如下:
$$
h_t=\text{RNN}_{\text{Enc}}(x_t, h_{t-1}) \qquad \text{(1)}
$$
其中, $h_t$是在每个时刻$t$, RNN的hidden state输出。那么有了这些hidden state，我们怎么用呢？

这里我们先不考虑那么多，假设我们有个模块，输入的是之前产生的hidden states， 输出一个压缩后的向量$c$，这时候我们就有：
$$
c=q(\{h_0,h_1,\cdots, h_{T_x}\}) \qquad \text{(2)}
$$

然后，我们考虑解码（<font color="red">注意我们先把 $q(\cdot)$ 放在一边，把整个流程走下来，后面再考虑</font>）.

解码器怎么做呢，我有了$c$, 那么就需要根据$c$和历史与预测的词 $y_0,\cdots,y_{t-1}$ 来预测当前的词 $y_t$, 这个就用到链式法则了，具体的公式如下：
$$
p(\mathbf{y})=\prod_{t=0}^{T-1}p(y_t|y_0,\cdots, y_{t-1},c)=p(y_t|y_0,\cdots, y_{t-1},c)\cdots p(y_1|y_{0},y_{\text{<BOS>}},c) \cdot p(y_0|y_{\text{<BOS>}},c) \qquad \text{(3)}
$$
这里，我们有一个 $y_{\text{<BOS>}}$, 因为实际的做seq2seq模型时，一般先给一个起始标志告诉模型 ”我们开始了！”
而上面的公式则是可以用RNN来实现，因为RNN的递归结构正好可以模拟这种链式法则，当然我们也可以用CNN。所以本质上<font color="red">我们需要一个递归结构！</font>

至于训练，我们可以考虑（目前主流的）cross-entropy loss函数：
$$
\mathcal{L}_{\text{xe}}=-\sum_{t=0}^{T-1}\log p_{\theta}(Y_t|Y_{0:t-1};\theta) \qquad \text{(4)}
$$
所以训练的目标就是学 $\theta$ (这里 $\theta$是值得所有的参数)。另外需要注意的是：loss函数里面是 $Y_t$，而RNN公式里面我用的 $y_t$。一般来说，RNN的输出并不是Word，而是一个向量$o_t\in \mathbb{R}^{1\times D}$, $D$是hidden state的维度。

我们一般先做一个mapping， 将 $o_t$map到一个字典长度的向量，然后做logsoftmax (e.g., 在PyTorch中用F.log_softmax($o_t$))得到输出的probability：
$$
Y_t \sim \text{logsoftmax}(\mathbf{W}_o o_t + \mathbf{b}_o) \qquad \text{(5)}
$$
这样我们就可以完成一个基本的encoder-decoder模型训练了！


## 基本encoder-decoder优缺点
优点很明显，它是一个end-to-end的网络，所有的参数都可以学习。只要给定训练数据即可训练出效果还不错的模型，省去了很多特征抽取以及各种复杂中间步骤。

## 改进
![](https://pic1.zhimg.com/80/v2-1c551cc1accef8d9ab0e09035dd2f4b1_hd.jpg)


但是，这样的系统有哪些可改进的地方？
1. Eq. (1)我们采用的是单向RNN作为编码器，这样的编码器只考虑了“history”信息，如果可以考虑“future”信息岂不更好。于是我们有了第一个改进：采用Bidirectional的RNN来作为编码器，这样双向RNN比较好地解决了“过去”与“未来”相结合的问题；
2. 之前我们一直回避$c$，怎么求得$c$呢？ 一种做法是直接拿最后一个时刻的hidden 输出$h_{T-1}$ (这里我们认为最后一个时刻的输出一定程度上能够表示整个句子); 另外一种做法是，我不仅仅用最后一个时刻的hidden output，我还用其它时刻的输出。那么问题来了，怎么用？一种是简单的做个均值 $c=\frac{1}{T}\sum_{t=0}^{T-1}h_t$, 这样很简单，但是我们知道，在做翻译的时候，当前预测词其实是和有限的词相关的，我不需要那么多信息（信息太多了，反而会让模型迷糊）。所以有了第一个改进，对每个$h_t$给一个权重，就有了:
$$
c=\sum_{i=0}^{T-1}\alpha_i h_i, i \in [0,T-1] \qquad \text{(6)}
$$
问题又来了-- 每个时刻我其实想要的不一样，Eq. (6)并没有体现出来，那么我怎么办呢？ 就考虑引入时刻的概率, $h_t$是变不了了，因为是encoder的输出。我考虑 $\alpha_{i}\rightarrow \alpha_{it}$, 这样就引入 $t$了。怎么计算呢？这就是这篇文章的改进，他们用了一个attention机制，这种机制在计算机视觉和自然语言处理中都很常见，具体的做法是：
$$
\alpha_{it}=\frac{\exp (e_{i,t})}{\sum_{k=0}^{T-1}\exp (e_{i,k})}
$$
其中$e_{i,t}$ 是有前一个单词 $y_{t-1}$ 加上 $h_k^{\text{Enc}},k\in {T-1}$ 通过一个linear mapping得到的: $e_{i,,t}=a(y_{t-1},h_k^{\text{Enc}})$, 注意这里 $h_k^{\text{Enc}}$ 是编码器的hidden state.

# 实验分析
## 数据集和实验结果
1. Dataset: WMT '14 BLEU: 36.15
2. Performance: Bidirectional GRU, 1000 hidden units. Multilayer maxout to compute output probabilities in decoder.

## Key Takeaways
- Attention mechanism is a weighted sum of the hidden states computed by the encoder. The weights come from a softmax-normalized attention function (a perceptron in this paper), which are learned during training.
- Attention can be expensive, because it must be evaluated for each encoder-decoder output pair, resulting in a len(x) * len(y) matrix.
- The attention mechanism improves performance across the board, but has a particularly large affect on long sentences, confirming the hyptohesis that the fixed vector encoding is a bottleneck.
- The authors use a bidirectional-GRU, concatenating both hidden states into a final state at each time step.
- It is easy to visualize the attention matrix (for a single input-ouput sequence pair). The authors show that in the case of English to French translations the matrix has large values on the diagonal, showing the these two languages are well aligned in terms of word order.


## 缺点和思考
> <font color="red">The attention mechanism seems limited in that it computes a simple weighted average. What about more complex attention functions that allow input states to interact?</font>

### Global attention + local attention
[Luong, Minh-Thang, Hieu Pham, and Christopher D. Manning. "Effective approaches to attention-based neural machine translation." arXiv preprint arXiv:1508.04025 (2015).]()
