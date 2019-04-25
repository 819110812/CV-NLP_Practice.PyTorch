# [Joint training for pivot-based neural machine translation](https://www.ijcai.org/proceedings/2017/0555.pdf)

<center><img src="https://media.springernature.com/original/springer-static/image/chp%3A10.1007%2F978-981-10-3635-4_2/MediaObjects/439248_1_En_2_Fig1_HTML.gif"></img></center>

针对低资源语言的神经机器翻译提出了源语言-桥接语言和桥接语言-目标语言翻译模型的联合训练算法，增强两个模型在参数估计中的关联性

给定一个源文本 $\mathbf{x}$ 和目标文本 $\mathbf{y}$ ,这里我们用 $P(\mathbf{y}|\mathbf{x}; \mathbf{\theta}_{x \rightarrow y})$ 来代表一个标准的attention-based NMT, $\mathbf{\theta}_{x \rightarrow y}$ 是模型参数。
所以给定一个source-target并行语料库 $D_{x,y}=\{\langle \mathbf{x}^{(s)}, \mathbf{y}^{(s)} \rangle\}_{s=1}^{S}$， 可以通过下面的方程训练得到:
$$
\begin{align}
\hat{\mathbf{\theta}}_{x \rightarrow y} =& \arg\max_{\mathbf{\theta}_{x \rightarrow y}}\Big\{ \mathcal{L}(\mathbf{\theta}_{x \rightarrow y}) \Big\}\\
\mathcal{L}(\mathbf{\theta}_{x \rightarrow y}) =& \sum_{s=1}^{S} \log P(\mathbf{y}^{(s)}|\mathbf{x}^{(s)}; \mathbf{\theta}_{x \rightarrow y})
\end{align}
$$

我们假设 $\mathbf{z}$ 是pivot language sentence. 假设 $D_{x,z}=\{\langle \mathbf{x}^{(m)}, \mathbf{z}^{(m)} \rangle\}_{m=1}^{M}$ 是 source-pivot parallel语料库, $D_{z,y}=\{\langle \mathbf{z}^{(n)}, \mathbf{y}^{(n)} \rangle\}_{n=1}^{N}$ 是 pivot-target parallel 语料库. 这样我们就有了新的模型:
$$
P(\mathbf{y}|\mathbf{x}; \mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y}) \nonumber = \sum_{\mathbf{z}} P(\mathbf{z}|\mathbf{x}; \mathbf{\theta}_{x \rightarrow z}) P(\mathbf{y}|\mathbf{z}; \mathbf{\theta}_{z \rightarrow y})
$$
模型的参数可以获得为：
$$
\hat{\mathbf{\theta}}_{x \rightarrow z}， \hat{\mathbf{\theta}}_{z \rightarrow y} = \arg\max_{\mathbf{\theta}_{x \rightarrow z}}\Big\{ \mathcal{L}(\mathbf{\theta}_{x \rightarrow z}) \Big\} ， \arg\max_{\mathbf{\theta}_{z \rightarrow y}}\Big\{ \mathcal{L}(\mathbf{\theta}_{z \rightarrow y}) \Big\}
$$
$$
\begin{align}
\mathcal{L}(\mathbf{\theta}_{x \rightarrow z}) &= \sum_{m=1}^{M}\log P(\mathbf{z}^{(m)}|\mathbf{x}^{(m)}; \mathbf{\theta}_{x \rightarrow z}) \\
\mathcal{L}(\mathbf{\theta}_{z \rightarrow y}) &= \sum_{n=1}^{N}\log P(\mathbf{y}^{(n)}|\mathbf{z}^{(n)}; \mathbf{\theta}_{z \rightarrow y})
\end{align}
$$
所以，给定一个新的unseen source sentence $\mathbf{x}$, decision rule是:
$$
\begin{align}
\hat{\mathbf{y}} =& \arg\max_{\mathbf{y}}\Bigg\{ \sum_{\mathbf{z}}P(\mathbf{z}|\mathbf{x}; \hat{\mathbf{\theta}}_{x \rightarrow z}) P(\mathbf{y}|\mathbf{z}; \hat{\mathbf{\theta}}_{z \rightarrow y}) \Bigg\}\\
\hat{\mathbf{y}} =& \arg\max_{\mathbf{y}}\Big\{ P(\mathbf{y}|\hat{\mathbf{z}}; \hat{\mathbf{\theta}}_{z \rightarrow y}) \Big\}，
\hat{\mathbf{z}} = \arg\max_{\mathbf{z}}\Big\{ P(\mathbf{z}|\mathbf{x}; \hat{\mathbf{\theta}}_{x \rightarrow z}) \Big\}
\end{align}
$$


## 训练
目标函数是：
$$
\begin{align}
\hat{\mathbf{\theta}}_{x \rightarrow z}, \hat{\mathbf{\theta}}_{z \rightarrow y} =& \arg\max_{\mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y}}\Big\{ \mathcal{J}(\mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y}) \Big\}\\
\mathcal{J}(\mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y}) =& \mathcal{L}(\mathbf{\theta}_{x \rightarrow z}) + \mathcal{L}(\mathbf{\theta}_{z \rightarrow y})+ \lambda \mathcal{R}(\mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y})
\end{align}
$$
这里connection term作者试了3种
1. Hard connection:
$$
\mathcal{R}_{\mathrm{hard}}(\mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y}) = \prod_{w \in \mathcal{V}^{z}_{x \rightarrow z} \cap \mathcal{V}^{z}_{z \rightarrow y}} \delta(\mathbf{\theta}^{w}_{x \rightarrow z}, \mathbf{\theta}^{w}_{z \rightarrow y})
$$
2. Euclidean distance:
$$
\mathcal{R}_{\mathrm{soft}}(\mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y}) = - \sum_{w \in \mathcal{V}^{z}_{x \rightarrow z} \cap \mathcal{V}^{z}_{z \rightarrow y}} || \mathbf{\theta}^{w}_{x \rightarrow z} - \mathbf{\theta}^{w}_{z \rightarrow y} ||_2
$$
3. Log-likelihood:
$$
\mathcal{R}_{\mathrm{likelihood}}(\mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y}) = \sum_{s=1}^{S} \log P(\mathbf{y}^{(s)}|\mathbf{x}^{(s)}; \mathbf{\theta}_{x \rightarrow z}, \mathbf{\theta}_{z \rightarrow y}) = \sum_{s=1}^{S}\log \sum_{\mathbf{z}} P(\mathbf{z}|\mathbf{x}^{(s)}; \mathbf{\theta}_{x \rightarrow z})P(\mathbf{y}^{(s)}|\mathbf{z}; \mathbf{\theta}_{z \rightarrow y})
$$
