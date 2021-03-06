## [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385)


Deep Residual Learning 是解决<span style="color:red">***超深度CNN网络训练问题，152层及尝试了1000层***</span>。

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

TLDR; The authors present Residual Nets, which achieve 3.57% error on the ImageNet test set and won the 1st place on the ILSVRC 2015 challenge. ResNets work by introducing "shortcut" connections across stacks of layers, allowing the optimizer to learn an easier residual function instead of the original mapping. This allows for efficient training of very deep nets without the introduction of additional parameters or training complexity. The authors present results on ImageNet and CIFAR-100 with nets as deep as 152 layers (and one ~1000 layer deep net).


#### Key Points

- <span style="color:red">***Problem***</span>: Deeper networks experience a *degradation* problem. They don't overfit but nonetheless perform worse than shallower networks on both training and test data due to being more difficult to optimize.
- <span style="color:red">***Because Deep Nets can in theory learn an identity mapping for their additional layers they should strict outperform shallower nets***</span>. In practice however, optimizers have problems learning identity (or near-identity) mappings. Learning residual mappings is easier, mitigating this problem.
- <span style="color:red">***Residual Mapping***</span>: If the desired mapping is H(x), let the layers learn F(x) = H(x) - x and add x back through a shortcut connection H(x) = F(x) + x. An identity mapping can then be learned easily by driving the learned mapping F(x) to 0.
- <span style="color:red">***No additional parameters or computational complexity are introduced by residuals nets***</span>.
- <span style="color:red">***Similar to Highway Networks, but gates are not data-dependent (no extra parameters) and are always open***</span>.
- Due the the nature of the residual formula, input and output must be of same size (just like Highway Networks). We can do size transformation by zero-padding or projections. Projections introduce additional parameters. Authors found that projections perform slightly better, but are "not worth" the large number of extra parameters.
- 18 and 34-layer VGG-like plain net gets 27.94 and 28.54 error respectively, not that higher error for deeper net. ResNet gets 27.88 and 25.03 respectively. Error greatly reduces for deeper net.
- Use Bottleneck architecture with 1x1 convolutions to change dimensions.
- Single ResNet outperforms previous start of the art ensembles. ResNet ensemble even better.


## Notes/Questions

- Love the simplicity of this.
- I wonder how performance depends on the number of layers skipped by the shortcut connections. The authors only present results with 2 or 3 layers.
- "Stacked" or recursive residuals?
- In principle Highway Networks should be able to learn the same mappings quite easily. Is this an optimization problem? Do we just not have enough data. What if we made the gates less fine-grained and substituted sigmoid with something else?
- Can we apply this to RNNs, similar to LSTM/GRU? Seems good for learning long-range dependencies.

### Basic Idea

随着CNN网络的发展，尤其的VGG网络的提出，大家发现网络的层数是一个关键因素，貌似越深的网络效果越好。但是随着网络层数的增加，问题也随之而来。

首先一个问题是 <span style="color:red">***vanishing/exploding gradients***</span>，即梯度的消失或发散。这就导致训练难以收敛。


- [Learning long-term dependencies with gradient descent is difficult](http://www.dsi.unifi.it/~paolo/ps/tnn-94-gradient.pdf).
- [Understanding the difficulty of training deep feedforward neural networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf).

但是随着:
1.  <span style="color:red">***Normalized initialization***</span>
    - [Efficient backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    - [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120)
    - [Object Detection Networks on Convolutional Feature Maps](http://arxiv.org/abs/1504.06066)
2.  <span style="color:red">***Intermediate normalization layers***</span>
    - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://arxiv.org/abs/1502.03167)

的提出，解决了这个问题。


当收敛问题解决后，又一个问题暴露出来： <span style="color:red">***随着网络深度的增加，系统精度得到饱和之后，迅速的下滑***</span>。让人意外的是这个性能下降<span style="color:red">***不是过拟合导致的***</span>。

按理说我们有一个shallow net，在不过拟合的情况下再往深加几层怎么说也不会比shallow的结果差，所以degradation说明不是所有网络都那么容易优化，这篇文章的motivation就是通过“deep residual network“解决degradation问题。

<p align="center"><img src="http://img.blog.csdn.net/20160114000613328" width="500" ></p>

文章：
1. [Convolutional Neural Networks at Constrained Time Cost](http://arxiv.org/abs/1412.1710)
2. [Highway Networks](https://arxiv.org/abs/1505.00387)

指出，对一个合适深度的模型加入额外的层数导致训练误差变大。


如果我们加入额外的 层只是一个<span style="color:red">*** identity mapping***</span>，那么随着深度的增加，训练误差并没有随之增加。所以我们认为可能存在另一种构建方法，随着深度的增加，训练误差不会增加，只是我们没有找到该方法而已。


这里我们提出一个 deep residual learning 框架来解决这种因为深度增加而导致性能下降问题。 假设<span style="color:red">***我们期望的网络层关系映射为$H(x)$***</span>, 我们让 the stacked nonlinear layers 拟合另一个映射，
$$F(x):=H(x)-x$$

那么原先的映射就是 $F(x)+x$。 这里我们假设优化残差映射$F(x)$ 比优化原来的映射 $H(x)$容易。

$F(x)+x$ 可以通过<span style="color:red">***shortcut connections***</span> 来实现，如下图所示：

<p align="center"><img src="http://img.blog.csdn.net/20151216160852064" width="300" ></p>

### Related Work
1. Residual Representations

以前关于残差表示的文献表明，问题的重新表示或预处理会简化问题的优化。

 2. Shortcut Connections

CNN网络以前对shortcut connections 也有所应用。

其实本文想法和Highway networks（Jurgen Schmidhuber的文章）非常相似， 就连要解决的问题（degradation）都一样。<span style="color:red">***Highway networks一文借用LSTM中gate的概念***</span>，除了正常的非线性映射$H(\mathbf{x}, \mathbf{W}h)$外，还设置了一条从$x$直接到$y$的通路，<span style="color:red">***以$T(\mathbf{x}, \mathbf{W}t)$作为$gate$来把握两者之间的权重***</span>，如下公式所示：

$$
y=H(x,WH).T(x,WT)+x.(1-T(x,WT))
$$

shortcut原意指<span style="color:red">***捷径，在这里就表示越层连接，就比如上面Highway networks里从x直接到y的连接***</span>。其实早在googleNet的inception层中就有这种表示：

<p align="center"><img src="http://img.blog.csdn.net/20160114003438140" width="300" ></p>

Residual Networks一文中，作者将Highway network中的含参加权连接变为固定加权连接，即 :
$$
y=H(x,WH).WT+x
$$

## Deep Residual Learning

1. Residual Learning

至此，我们一直没有提及residual networks中residual的含义。那这个“残差“指什么呢？我们想：
如果能用几层网络去逼近一个复杂的非线性映射$H(x)$，那么同样可以用这几层网络去逼近它的residual function：$F(x)=H(x)-x$，但我们“猜想“优化residual mapping要比直接优化$H(x)$简单。

正如前言所说，如果<span style="color:red">***增加的层数可以构建为一个 identity mappings***</span>，那么增加层数后的网络训练误差应该不会增加，与没增加之前相比较。性能退化问题暗示多个非线性网络层用于近似identity mappings 可能有困难。<span style="color:red">***使用残差学习改写问题之后，如果identity mappings 是最优的，那么优化问题变得很简单，直接将多层非线性网络参数趋0***</span>。

实际中，identity mappings 不太可能是最优的，但是上述改写问题可能能帮助预处理问题。<span style="color:red">***如果最优函数接近identity mappings，那么优化将会变得容易些***</span>。 实验证明该思路是对的。

推荐读者们还是看一下本文最后列出的这篇reference paper，本文中作者说与Highway network相比的优势在于：
|x|Highway Network|Residual Network|评论|
|---|---|---|---|
|gate参数|有参数变量$WT$|没参数，定死的, 方便和没有residual的网络比较|不上优势，参数少又data-independent，结果肯定不会是最优的，文章实验部分也对比了效果，确实是带参数的error更小，<span style="color:red">***但是$WT$这个变量与解决degradation问题无关***</span>|
|关门？|有可能关门$(T(x,WT)=0)$|不会关门|$T(x,WT)\in[0,1$], 但一般不会为0|

所以说这个比较还是比较牵强。。anyway，人家讲个故事也是不容易了。

2. Identity Mapping by Shortcuts

$$\mathbf{y}=F(\mathbf{x},\{W_i\})+\mathbf{x}$$
这里假定输入输出维数一致，如果不一样，可以通过 linear projection 转成一样的。

3. Network Architectures

<p align="center"><img src="http://img.blog.csdn.net/20151216164510071" width="300" ></p>

Plain Network 主要是受 VGG 网络启发，主要采用3*3滤波器，遵循两个设计原则：
1) <span style="color:red">***对于相同输出特征图尺寸，卷积层有相同个数的滤波器，***</span>
2) <span style="color:red">***如果特征图尺寸缩小一半，滤波器个数加倍以保持每个层的计算复杂度。通过步长为2的卷积来进行降采样。一共34个权重层。***</span>

需要指出，我们这个网络与VGG相比，<span style="color:red">***滤波器要少，复杂度要小。***</span>
<span style="color:red">***Residual Network 主要是在 上述的 plain network上加入 shortcut connections***</span>

3. 34层 residual network

网络构建思路：基本<span style="color:red">***保持各层complexity不变，也就是哪层down－sampling了，就把filter数＊2***</span>， 网络太大，此处不贴了，大家看paper去吧， <span style="color:red">***paper中画了一个34层全卷积网络， 没有了后面的几层fc，难怪说152层的网络比16-19层VGG的计算量还低***</span>。

<span style="color:red">***这里再讲下文章中讲实现部分的 tricks***</span>：
- 图片resize：短边长random.randint(256,480)
- 裁剪：224＊224随机采样，含水平翻转
- 减均值
- 标准颜色扩充[2]
- conv和activation间加batch normalization[3]
- 帮助解决vanishing/exploding问题
- minibatch-size:256
- learning-rate: 初始0.1, error平了lr就除以10
- weight decay：0.0001
- momentum：0.9
- 没用dropout[3]

4.  实验结果

<span style="color:red">***34层与18层网络比较***</span>：训练过程中，
34层plain net（不带residual function）比18层plain net的error大
34层residual net（不带residual function）比18层residual net的error小，更比34层plain net小了3.5%(top1)
18层residual net比18层plain net收敛快

Residual function的设置：
A）在H(x)与x维度不同时， 用0充填补足
B） 在H(x)与x维度不同时， 带WT
C）任何shortcut都带WT
loss效果： A>B>C


## Comparing Highway Networks & GradNets'

### tl;dr

Highway Networks and GradNets both allow interpolation of network architecture. GradNets rely on a heuristic for global interpolation, while Highway Networks employ learnable weights for neuron-specific gating. The latter turned out to be easier to train due to the flexibility and self-optimization.

### Intro

As part of a deep learning study group, I implemented both Highway Networks and GradNets to compare results under similar conditions. The Highway Network implementation is based on Jim Fleming's [blog post](https://medium.com/jim-fleming/highway-networks-with-tensorflow-1e6dfa667daa#.7z1d6allb) with small  modifications and the GradNet implementation is derivative of that. Both demos are in Jupyter Notebooks, run Tensorflow, and do not require GPUs to finish quickly. To keep formulations simple, I only compare fully-connected networks.

### Highway Networks

Highway Networks are an architectural feature that allows the network to adaptively "flatten" itself by passing certain neurons through without any transformation. The network typically starts with initial biases towards passing the data through, behaving like a shallow neural network. After some training, the weights that control the "gates" of the network start to adjust and close down the highway in the early layers. Certain "lanes" of the highway will selectively activate.

These networks in fact learn not just the weights for the underlying affine transformations that are then run through the nonlinearn activation kernels (sigmoid, ReLU, tanh, etc), but also a companion set of weights for the gate that determines how much of that activation to use. This gate is controlled by a sigmoid activation applied to an affine transform, parameterized by the companion weights.

The paper describes these more formally as the Hypothesis $H$, the Transform Gate $T$, and the Carry Gate $C$. The value of the Carry Gate is simply 1 minus the value of the Transform Gate. The Hypothesis is the underlying transformation being performed at the layer.

A standard fully connected layer looks like

$$
y = H(x, W_H) = activation(W_H^Tx + b_H)
$$

A Highway Layer looks like

$$
y = H(x, W_H) \cdot T(x, W_T) + x \cdot C(x, W_C)
$$

$\cdot$ denotes elementwise multiplication. Note that all W matrices match in dimension.

Since $T(x, W_T) = 1 - C(x, W_C)$, the last term in the equation above becomes $1 - T(x, W_T)$, so we don't need $W_C$ anymore.

$T(x, W_T) = sigmoid(W_H^Tx + b_T)$ produces element-wise "gates" between 0 and 1.

A critical point of initialization is $b_T$. It should be set to a fairly negative value so that the network initially passes the $x$ through.

In code:

```python
def highway_layer(x, size, activation, carry_bias=-1.0):
    W = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[size]), name='bias')

    W_T = tf.Variable(tf.truncated_normal([size, size], stddev=0.1), name='weight_transform')
    b_T = tf.Variable(tf.constant(carry_bias, shape=[size]), name='bias_transform')

    H = activation(tf.matmul(x, W) + b, name='activation')
    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name='transform_gate')
    C = tf.sub(1.0, T, name="carry_gate")

    y = tf.add(tf.mul(H, T), tf.mul(x, C), 'y')
    return y
```

More about highway networks on this [page](http://people.idsia.ch/~rupesh/very_deep_learning/) and the papers listed there.


### GradNets

[GradNets](http://arxiv.org/abs/1511.06827) offer a simplified alternative to gradual interpolation between model architectures. The inspiration is similar to that of Highway Networks; early in training, prefer simpler architecture, whereas later in training, transition to complex.

The variable $g$ anneals over a $\tau$ epochs (full passes through shuffled data), controlling the amount of interpolation between the simple activation and the nonlinear one. Using similar notation as before:

$$
g = \min(t / \tau, 1)
$$

$$
H(x, W) = ReLU(W^Tx + b)
$$

$$
J(x, W) = I(W^Tx + b)
$$

$$
y = g \cdot H(x, W) + (1 - g) \cdot J(x, W)
$$

$t$ is the continuous or stepwise epoch number

In code:

```python
def grelu_layer(x, input_size, output_size, g):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='bias')
    u = tf.matmul(x, W) + b
    y = g * tf.nn.relu(u) + (1 - g) * u
    return y
...
for i in range(num_iter):
  batch_xs, batch_ys = mnist.train.next_batch(mini_batch_size)

  epoch = i / iter_per_epoch
  gs = min(epoch / tau, 1.0)
...
```

I used `__future__.division` to default to floating point division with a single `/`, whereas integer floor division would be `//`. The former corresponds to a continuous $t$, and the latter a stepwise $t$. The paper was not explicit about which one to use, but it made sense to be as gradual as possible in GradNets.


### Experiment

### Highway Network

I was able to reproduce results on Highway Networks quite easily. Using the following parameters:

* 50 hidden layers of size 50 each
* Minibatch size of 50
* SGD optimizer
* $10^{-2}$ starting learning rate
* No weight decay
* Initial carry bias of -1

I got to ~92% test accuracy within 2 epochs and hit the best test accuracy of ~96% around epoch 13. The network started to overfit after that, which is expected because I did not apply learning rate decay or any form of regularization.

I tried a few other configuration and referred to Jim's post (linked above) in order to confirm that the model converged under a variety of conditions.

### GradNets - Linear GReLU

I tried to reproduce the first example from GradNets, interpolating between a simple linear (Identity) activation and a ReLU activation. The underlying weights are still the same under each path, so the output is weighted mix of 2 different activations on the same affine transformation.

Using the same optimizer (SGD), the same learning rate, and otherwise the same architecture as Highway Networks, I was not able to get the network to converge. The norm of the gradients moved toward 0 as `g` annealed to 1, and when g hit 1, the gradients all hit 0.

### GradNets v2 - Identity GReLU

As an alternative approach, I tried to obtain an interpolation closer to what Highway Networks achieved. Following the same constraint as Highway Networks, I modified the GReLU layer to interpolate between the full transformation and an identity function directly on $x$ rather than on the affine transform $W^Tx + b$. Here is the revised output:

$$
g \cdot ReLU(W^Tx + b) + (1 - g) \cdot I(x)
$$

The GReLU layer now looks like:

```python
def grelu_layer(x, input_size, output_size, g):
    W = tf.Variable(tf.truncated_normal([input_size, output_size], stddev=0.1), name='weight')
    b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='bias')
    y = g * tf.nn.relu(tf.matmul(x, W) + b) + (1 - g) * x
    return y
```

Because of the multiplicative nature of backpropogation, I ended up with situations where I had exploding gradients and weights. I added a bit of monitoring to keep track of the L1 norm of both.

To understand how this happens, here's an example. Suppose `y_` is non-zero and `y` is very close to zero. `log(y)` is a very big negative number, and gets multiplied by a non-zero `- y_`. When gradients flow more freely as `g` increases, this large number is multipled and summed across many nodes in many layers.

The value of gradient $dW$ at a given layer is $dx \times dy$. $dy$ is that big number that came from the next layer, and $dx$ is the activation from previous layer. Repeating the process down all layers through backprop can lead to exponential growth under the wrong conditions.

A good proxy for big activations is big weights, so I collected those as well. I noticed some weight explosion, so I applied the `relu6` activation, which clips the output of the unit. It prevented gradient explosion, but at saturation in hidden layers, the network quickly diverges without means to recover.

### Kitchen Sink Fix

Without making the network any shallower, I tried:

* Regularization: dropout, L1, L2
* Optimizer: Adam, AdaGrad, RMSProp
* Learning Rate: start rate, exponential and constant decay
* Gradient clipping and normalization

Nothing seemed to help the 50-layer Linear GReLU train. As for the Identity GReLU (the one that more resembles Highway Networks), it took a combination of:

* L1 regularization - very carefully chosen to stabilize weights
* Lower starting learning rate of $10^{-3}$
* Aggressive exponential decay of the learning rate
* Using the Adam optimizer
* Mild dropout (5%)

...in order to keep the Identity GReLU from diverging, but it would still hit a random `NaN` spike that didn't seem to follow  a climb in the L1 norm of weights & gradients. I could have considered checkpointing + early-stopping, but I would prefer that the network demonstrate stability without outside help. It certainly would've saved a lot of time.

Eventually, I figured out from reading forum posts that the Identity GReLU spikes when $y \to 0$ and $log(y) \to -\infty$. Doh! Lesson learned: read the forums!

The simple fix is to add a small number to `y`:

```python
cross_entropy = -tf.reduce_sum(y_ * tf.log(y + 1e-9))
```

I ended up hitting 94% accuracy around epoch 15 and staying there until the end of training. I was pretty happy with the graphs for weights and gradients - they remained in a pretty small range throughout. It should be emphasized that the convergence property was sensitive to ALL of the hyperparameters above. Significant changes in any of them led to divergence or no learning.

Here are some graphs of a run:

[![Training Accuracy][1]][1]
[1]: https://lh3.googleusercontent.com/yx6TJ-IduYF0OCScLU9pT0zbQOmKtwn7wqCJiBFOHL1p9i2SLhOtc1CqH2TUpmPZYJkgwWnRhgvNqw=w1515-h422-no

[![Weight Norm][2]][2]
[2]: https://lh3.googleusercontent.com/8D90Lv9eIKuuzr_OMIOLAR7jsQYki16TOpBm03oyGkI3KwiydCkZczhe48QS5wKtBps0r_XH0giGOg=w1510-h414-no

[![Gradient Norm][3]][3]
[3]: https://lh3.googleusercontent.com/baOtgb3xTS4fSp4r7l0ABw5JC3ePC91wdotvfy2OdlL73k8otEVPajE8RbGiVk8vS9HnX7RWUUMeMA=w1509-h415-no

### Sanity Check

For my own sanity check, I left the hyperparameters in their tuned state and removed the GradNet portion to see how well the network would do. As expected, the network failed to bounce out of its initial state.

## Final Thoughts

While Highway Networks effectively double the number of parameters per layer, they converge quickly and stably with naive hyperparameters.

In contrast, GradNets work best when interpolating between "Highway Mode" (Identity) and ReLU, with the caveat that one must tune hyperparameters very carefully (perhaps through systematic Grid/Randomized search). They certainly make training the net easier in comparison to no interpolation.

The winning characteristic of Highway Networks is ability to learn good gating parameters for every neuron as part of end-to-end training, whereas GradNets must apply a single multiplier $g$ for the entire network.

However, gradual interpolation of other aspects of the architecture, such as dropout, batch normalization, convolutions, are available and have been tested in GradNets, but have no counterpart in Highway Networks. Further investigation into using Highway Networks for these components could be interesting.

Demo runs with inline implementation can be found [here](https://github.com/ZhangBanger/highway-vs-gradnet)
