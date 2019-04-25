# [A Teacher-Student Framework for Zero-Resource Neural Machine Translation]()

Train source-to-target NMT (student) without parallel corpora available, guided by the existing pivot-to-target NMT (teacher) on a source-pivot parallel corpus. 这里 $x$ 是 source, $y$ 是target, $z$ 是pivot。


## 细节
我们先回顾下历史，传统的基于 Triangulated pivot-based method $x\rightarrow z, z \rightarrow y$ 会存在error propagation的问题。简单的说就是，$x\rightarrow z$ 产生的错误会传递到 $z\rightarrow y$ （见左图）。

![](https://user-images.githubusercontent.com/7529838/32471186-5e035214-c39f-11e7-98fe-9ad406c56e59.png)

所以这篇文章没有用pivot-based方法，而是提出了一个Teacher-Student方法。

## 模型

### Sentence-level learning
首先假设：<font color="Red">如果 $x$ 是翻译自 $z$ (pivot language), 那么从 $x$ 生成 $y$ 的概率应该接近与 $y$ 对应的 $z$ 的概率。</font>

那么优化这2个概率就可以得到更好的 $x$ 到 $y$，
$$
\mathcal{J}_{\text{SENT}}(\theta_{x\rightarrow y})=\sum_{<x,z>\in D_{x,z}} \text{KL}(P(y|z;\hat{\theta}_{z\rightarrow y})\|P(y|x;\theta_{x\rightarrow y})) \qquad \text{(1)}
$$

由于teacher model是pretrained的，所以我们可以去掉对应参数，得到：
$$
\mathcal{J}_{\text{SENT}}(\theta_{x\rightarrow y})=\sum_{<x,z>\in D_{x,z}} \text{KL}(P(y|x;\theta_{x\rightarrow y})) \qquad \text{(2)}
$$

由此，我们得到新的训练目标函数：
$$
\hat{\theta}_{x\rightarrow y}=\arg\min_{\theta_{x\rightarrow y}}\{\mathcal{J}_{\text{sent}}(\theta_{x\rightarrow y})\}\qquad \text{(3)}
$$

### Word-Level Teaching

$$
\mathcal{J}_{\text{WORD}}(\theta_{x\rightarrow y})=\sum_{<x,z>\in D_{x,z}} \mathbb{E}_{y|z;\hat{\theta}_{z\rightarrow y}}[S(x,y,z,\hat{\theta}_{z\rightarrow y},\theta_{x\rightarrow y})] \qquad \text{(4)}
$$
其中 $S(x,y,z,\hat{\theta}_{z\rightarrow y},\theta_{x\rightarrow y})=\sum_{j=1}^{|y|}\sum_{y\in \mathcal{V}_y}P(y|z,Y_{<j};\hat{\theta}_{z\rightarrow y})\times \log P(y|x,y;\theta_{x\rightarrow y})$
