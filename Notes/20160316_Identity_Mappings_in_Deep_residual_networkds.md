
## Identity Mappings in Deep Residual Networks

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, ArXiv, 2016

### Summary

This is follow-up work to the ResNets paper. It studies the propagation formulations behind the connections of deep residual networks and performs ablation experiments. A residual block can be represented with the equations $y_l = h(x_l) + F(x_l, W_l); x_{l+1} = f(y_l)$. $x_l$ is the input to the $l$-th unit and $x_{l+1}$ is the output of the $l$-th unit. In the original ResNets paper, $h(x_l) = x_l$, $f$ is ReLu, and $F$ consists of 2-3 convolutional layers (bottleneck architecture) with BN and ReLU in between. In this paper, they propose a residual block with both $h(x)$ and $f(x)$ as identity mappings, which trains faster and performs better than their earlier baseline. Main contributions:

- Identity skip connections work much better than other multiplicative interactions that they experiment with:
    - Scaling ($h(x) = \lambda x$): Gradients can explode or vanish depending on whether modulating scalar $\lambda > 1$ or $< 1$.
    - Gating ($1-g(x)$ for skip connection and $g(x)$ for function $F$):
    For gradients to propagate freely, $g(x)$ should approach 1, but
    F gets suppressed, hence suboptimal. This is similar to highway
    networks. g(x) is a 1x1 convolutional layer.
    - Gating (shortcut-only): Setting high biases pushes initial g(x)
    towards identity mapping, and test error is much closer to baseline.
    - 1x1 convolutional shortcut: These work well for shallower networks
    (~34 layers), but training error becomes high for deeper networks,
    probably because they impede gradient propagation.

- Experiments on activations.
    - BN after addition messes up information flow, and performs considerably
    worse.
    - ReLU before addition forces the signal to be non-negative, so the signal is monotonically increasing, while ideally a residual function should be free to take values in (-inf, inf).
    - BN + ReLU pre-activation works best. This also prevents overfitting, due
    to BN's regularizing effect. Input signals to all weight layers are normalized.

### Strengths

- Thorough set of experiments to show that identity shortcut connections
are easiest for the network to learn. Activation of any deeper unit can
be written as the sum of the activation of a shallower unit and a residual
function. This also implies that gradients can be directly propagated to
shallower units. This is in contrast to usual feedforward networks, where
gradients are essentially a series of matrix-vector products, that may vanish, as networks grow deeper.

- Improved accuracies than their previous ResNets paper.

### Weaknesses / Notes

- Residual units are useful and share the same core idea that worked in
LSTM units. Even though stacked non-linear layers are capable of asymptotically
approximating any arbitrary function, it is clear from recent work that
residual functions are much easier to approximate than the complete function.
The [latest Inception paper](http://arxiv.org/abs/1602.07261) also reports
that training is accelerated and performance is improved by using identity
skip connections across Inception modules.

- It seems like the degradation problem, which serves as motivation for
residual units, exists in the first place for non-idempotent activation
functions such as sigmoid, hyperbolic tan. This merits further
investigation, especially with recent work on function-preserving transformations such as [Network Morphism](http://arxiv.org/abs/1603.01670), which expands the Net2Net idea to sigmoid, tanh, by using parameterized activations, initialized to identity mappings.

## Identity Mappings in Deep Residual Networks

[Code is avaiable at here](https://github.com/KaimingHe/resnet-1k-layers)

They analyze the <span style="color:red">***propagation formulations***</span> behind the residual building blocks, which suggest that the <span style="color:red">***forward and backward signals can be directly propagated from one block to any other block, when using identity mappings as the skip connections and after-addition activation***</span>.


### ResNets
<p align="center"><img src="http://img.blog.csdn.net/20151216160852064" width="300" ></p>

ResNets consist of many stacked “Residual Units”. Each unit can be expressed in a general form:
$$\begin{cases}
\mathbf{y}_l=h(\mathbf{x}_l)+F(\mathbf{x}_l,\mathbf{W}_l)\\
\mathbf{x}_{l+1}=f(\mathbf{y}_l)\end{cases}
$$


where $\mathbf{x}_l$ and $\mathbf{x}_{l+1}$ are input and output of the $l$-th unit, and $F$ is a <span style="color:red">***residual function***</span>.

In "<span style="color:red">***Deep Residual Learning for Image Recognition***</span>", $h(\mathbf{x}_l)=\mathbf{x}_l$ is an identity mapping and $f$ is a ReLU function.

<p align="center"><img src="https://dl.dropboxusercontent.com/s/7z4pcawz8lpxkjl/Screenshot_2016-05-09_20-50-43.png" width="400" ></p>

### Analysis of Deep Residual Networks
$$\mathbf{x}_{l+1}=\mathbf{x}_l+F(\mathbf{x}_l,W_l)\Rightarrow \mathbf{x}_{l+2}=\mathbf{x}_{l+1}+F(\mathbf{x}_{l+1},W_{l+1})
= \mathbf{x}_l+F(\mathbf{x}_l+W_l)+F(\mathbf{x}_{l+1},W_{l+1})$$
$$\mathbf{x}_L=\mathbf{x}_l+\sum_{i=l}^{L-1}F(\mathbf{x}_{i},W_{i})$$
for any deeper unit $L$ and any shallower unit $l$.

The equation above also leads to nice backward propagation properties. Denoting the loss function as $\varepsilon$, from the chain rule of backpropagation we have:
$$
\frac{\partial \varepsilon}{\partial x_l}=\frac{\partial \varepsilon}{\partial x_L}\frac{\partial x_L}{\partial x_l}=\frac{\partial \varepsilon}{\partial x_l}(1+\frac{\partial}{\partial x_l}\sum_{i=1}^{L-1}F(x_i,W_i))
$$


### On the Importance of Identity Skip Connections
$$\mathbf{x}_{l+1}=\lambda_l \mathbf{x}_l+F(\mathbf{x}_l,W_l)$$
$$\mathbf{x}_L=\prod_{i=1}^{L-1}\lambda_i \mathbf{x}_l+\sum_{i=l}^{L-1}\hat{F}(\mathbf{x}_{i},W_{i})$$
$$
\frac{\partial \varepsilon}{\partial x_l}=\frac{\partial \varepsilon}{\partial x_L}\frac{\partial x_L}{\partial x_l}=\frac{\partial \varepsilon}{\partial x_l}(\prod_{i=1}^{L-1}\lambda_i+\frac{\partial}{\partial x_l}\sum_{i=1}^{L-1}\hat{F}(x_i,W_i))
$$

|Various types of shortcut connections|case|on shortcut|on $F$|error %|remark|
|---|---|---|---|---|---|
|<p align="center"><img src="https://dl.dropboxusercontent.com/s/oxy3p3u09zppeba/Screenshot_2016-05-09_21-07-47.png" width="200" ></p>|original |1|1|6.61||
|<p align="center"><img src="https://dl.dropboxusercontent.com/s/q9qev76sijorsvz/Screenshot_2016-05-09_21-08-41.png" width="200" ></p>|constant scaling|$\begin{cases}0\\0.5\\0.5\end{cases}$|$\begin{cases}1\\1\\0.5\end{cases}$|$\begin{cases}fail\\fail\\12.35\end{cases}$|This is a plain net|
|<p align="center"><img src="https://dl.dropboxusercontent.com/s/ddx5plurxcjvibq/Screenshot_2016-05-09_21-12-23.png" width="200" ></p>|exclusive gating|$\begin{cases}1-g(x)\\1-g(x)\\1-g(x)\end{cases}$|$\begin{cases}g(x)\\g(x)\\g(x)\end{cases}$|$\begin{cases}fail\\8.70\\9.81\end{cases}$|$\begin{cases}init \ b_g \ to \ -5\\init \ b_g =-6\\init \ b_g =-7\end{cases}$|
|<p align="center"><img src="https://dl.dropboxusercontent.com/s/ql347063jkcww47/Screenshot_2016-05-09_21-12-34.png" width="200" ></p>|shortcut only gating|$\begin{cases}1-g(x)\\1-g(x)\end{cases}$|$\begin{cases}1\\1\end{cases}$|$\begin{cases}12.86\\6.91\end{cases}$|$\begin{cases}init \ b_g = 0\\init \ b_g =-6\end{cases}$|
|<p align="center"><img src="https://dl.dropboxusercontent.com/s/g3bjkt28l7lo9vr/Screenshot_2016-05-09_21-12-44.png" width="200" ></p>|$1\times 1$ conv shortcut|1x1 conv|1|12.22||
|<p align="center"><img src="https://dl.dropboxusercontent.com/s/nt6upy5l6qqx32m/Screenshot_2016-05-09_21-12-55.png" width="200" ></p>|dropout shortcut|dropout 0.5|1|fail||


<p align="center"><img src="https://dl.dropboxusercontent.com/s/9xh9enr2prgqwtx/Screenshot_2016-05-09_21-29-11.png" width="800" >


### Experiments on Activation

<p align="center"><img src="https://dl.dropboxusercontent.com/s/jrp8cvxm0dm70cd/Screenshot_2016-05-09_21-31-11.png" width="800" >

### Analysis
We find that the impact of pre-activation is twofold.
1.  <span style="color:red">***the optimization is further eased***</span> (comparing with the baseline ResNet) because $f$ is an identity mapping.
2. using BN as  <span style="color:red">***pre-activation improves regularization of the models.***</span>
