# [Unsupervised Machine Translation Using Monolingual Corpora Only]()

The original mark can be found in this [blog](https://zhuanlan.zhihu.com/p/31404350)

这篇论文实现了在完全没有平行语料训练的情况下（<font color="red">语料库还是属于同一类型，虽然不同语言。另外，借助了word-level align, 这点也是算一个先验</font>），无监督学习机器翻译。这篇论文利用了auto-encoder, GAN这两种结构实现机器翻译的无监督学习。


## Motivation

基于encoder-decoder的NMT过程可以看作将src language 句子 encode 到一个 latent space 中，再用decoder 将其从 latent space 中decode 到 tgt language 句子。问题是：<b>NMT系统只有当它的输入是由受过训练的encoder产生的时候，它的decoder工作得很好(<font color="red">decoder的输入分布在infer的时候和train的时候是一致的</font>)</b>

基于以上考虑，作者有了思路：我们希望encoder可以将各种语言(src and tgt)都映射到同一个latent space. 这样decoder就能够忽略src language，将其从 latent space 中 decode 成 tgt language. 和之前的一个工作(通过引入双语词典信息实现 word level 的空间对齐已经取得了效果)不一样，作者认为这个latent space 还包含了句子的结构信息，对齐这个 latent space 可以实现 sentence level 的对齐。这样就可以基于word-level对齐，加上sentence-level对齐得到翻译系统。

# Model

NMT模型有一个encoder和一个decoder构成，encoder 和 decoder 的表示如下：

- encoder： $\textbf{z}=e_{\theta_{enc}}z(\textbf{x},l)$ 简化表示为 $e(x,l)$;
- decoder： $\textbf{y}=d_{\theta_{dec}}z(\textbf{z},l)$ 简化表示为 $d(x,l)$

其中 $\theta$ 代表参数， $z$ 代表embedding， $l\in\{src,tgt\}$ 代表语言， $\textbf{x}$、$\textbf{z}$ 代表输入。

将在src domain 的 sentence集合用 $D_{src}$ 表示，在tgt domain 的 sentence集合用 $D_{tat}$ 表示。目标是寻求一个src domain 和 tgt domain 共用的 latent space。作者的做法是，训练 encoder-decoder 能够 reconstruct sentence(<font color="red">这个sentence可能是原本 sentence 的 noisy version，也可能是其转化成 tgt domain 的 sentence</font>)，也就是接下来介绍的 denoting auto-encoding 和 cross domain training。同时为了增强对 latent space 对齐的约束，作者采用了对抗学习的方法。

<center><img src="https://pic1.zhimg.com/80/v2-2c533358398273eddcd2668b1b6b5b51_hd.jpg"></img></center>

## Denoising auto-encoding
过程是把某种语言L的句子加一些噪声（<font color="red">打乱顺序/丢掉一些词</font>），然后用 $e(x,l)$ 编码加噪声后的句子，再用 $d(x,l)$ 恢复它。通过最小化与原句子的差异来训练encoder 和 decoder，loss函数为：
$$
\mathcal{L}_{\text{auto}}(\theta_{\text{enc}}, \theta_{\text{dec}}, \mathcal{Z}, \ell)=\mathbb{E}_{x\sim \mathcal{D}_l, \hat{x}\sim d(e(C(x),\ell),\ell)}[\Delta (\hat{x}, x)] \qquad \text{(1)}
$$
这里 $C(x)$ 加噪声的目的是让模型不是仅仅学习到了复制句子，尤其模型加了attention机制的情况下。Denoising auto-encoding 的训练能过让模型学习去噪，而在 cross domain training 中模型要学习的是从存噪的特征中翻译出无噪的句子，如此促进其更好地学习。

## Cross domain training
这里采用的是back-translation的方法。语言 $\ell_1$ 的句子 $x$ 通过上次迭代得到的模型 $d(e(x, l_1), l_2)$ 翻译出语言 $\ell_2$ 的句子 $y$，构造出了伪平行句对 $(y, x)$。之后将 $y$ 加噪，经过模型  $d(e(C(y), l_2), l_1)$ ，得到 $\widehat{x}$ ，通过最小化与原句子的差异来训练encoder 和 decoder。可以一定程度上对齐两种语言的 latent space。loss函数为：
$$
\mathcal{L}_{\text{cd}}(\theta_{\text{enc}}, \theta_{\text{dec}}, \ell_1, \ell_2)=\mathbb{E}_{x\sim \mathcal{D}_{\ell_1}, \hat{x}\sim d(e(C(y),\ell_2),\ell_1)}[\Delta (\hat{x}, x)] \qquad \text{(2)}
$$

## Adversarial training
具体做法是，用对抗的方法训练一个生成器 G（这里就是encoder），以及一个判别器 D（分辨该 latent representation 是来自src domain(0) 还是 tgt domain(1)），学好以后两个空间就对齐了。

![](https://pic4.zhimg.com/50/v2-1730db8daf5faa6a7f0d18d470e881b6_hd.jpg)

判别器的loss函数为：
$$
\mathcal{L}_{\mathcal{D}}(\theta_D|\theta, \mathcal{Z})=-\mathbb{E}_{x_i,\ell_i}[\log p_D(\ell_i|e(x_i,\ell_i))] \qquad \text{(2)}
$$
encoder的loss函数为：
$$
\mathcal{L}_{adv}(\theta_{\text{enc}},\mathcal{Z}|\theta_D)=-\mathbb{E}_{x_i,\ell_i}[\log p_D(\ell_j|e(x_i,\ell_i))] \qquad \text{(3)}
$$
nmt最终的loss函数为：
$$
\begin{align}
\mathcal{L}(\theta_{\text{enc}},\theta_{\text{dec}},\mathcal{Z})=&\lambda_{auto}[\mathcal{L}_{auto}(\theta_{\text{enc}},\theta_{\text{dec}},\mathcal{Z},src)+\mathcal{L}_{auto}(\theta_{\text{enc}},\theta_{\text{dec}},\mathcal{Z},tgt]+\\
&\lambda_{cd}[\mathcal{L}_{auto}(\theta_{\text{enc}},\theta_{\text{dec}},\mathcal{Z},src,tgt)+\mathcal{L}_{cd}(\theta_{\text{enc}},\theta_{\text{dec}},\mathcal{Z},tgt,src]+\\
  &\lambda_{adv}[\mathcal{L}_{adv}(\theta_{\text{enc}},\mathcal{Z}|\theta_D) \qquad \text{(4)}
\end{align}
$$

## Training
![](https://pic1.zhimg.com/80/v2-67adfc6b59078bf0c16205350a0fb0c8_hd.jpg)

## 总结

该模型具有以下特点：
1. 首先使用无监督方法获得跨语言的词翻译 (word-level alignment初始化很重要)，然后使用<font style="color:Red">逐词翻译的方法</font>初始化翻译模型；
2. 模型的损失函数可以<font style="color:Red">度量从有噪输入序列中重构句子或翻译句子的能力</font>。（1）对于自编码任务，有噪输入通过丢弃或交换句子中的词获得；Denoising Auto-encoding. 自编码器可以使用seq2seq+attention来实现。然而，如果不加任何限制，自编码器会完全变成一个copy的网络，在这种情况下，模型实际上并没有从数据中学到任何有用的模式。为了解决这一问题，本文使用了类似于Denoising Auto-encoder（去噪自编码器，DAE）的思想：先为输入的句子添加噪声，然后再进行编码. （2）对于翻译任务，有噪输入则使用上次迭代获得的翻译结果来表示。为了提高源语言与目标语言句子分布的对齐程度，本文还使用了对抗训练的方法。

当然，也有不足：
1. 还是需要pre-trained word-level alignment作为种子；
2. 虽然src和tgt不是paired， 但还是从一个domain采集的，对于完全语言风格不一致的数据库，有待验证;
3. 英法德的文字表达其实差别不到，但是如果中文日文等，这差异较大；
