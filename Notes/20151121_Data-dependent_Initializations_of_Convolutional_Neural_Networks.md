## [Data-dependent Initializations of Convolutional Neural Networks](https://arxiv.org/abs/1511.06856)


> This work aims to explore how to better fine-tune CNNs

A small miscalibration of the initial weights leads to vanishing or explod- ing gradients, as well as poor convergence properties.

In this work we present <span style="color:red">***a fast and simple data-dependent initialization procedure, that sets the weights of a network such that all units in the network train at roughly the same rate, avoiding vanishing or exploding gradients***</span>.


<span style="color:red">***Our initialization matches the current state-of-the-art unsupervised or self-supervised pre-training methods on standard computer vision tasks, such as image classification and object detection, while reducing the pre-training time by three orders of magnitude***</span> .


### Background

This “pre-trained” representation is then “fine-tuned” on a smaller dataset where the target labels may be more expensive to obtain. These fine-tuning datasets generally do not fully constrain the CNN learning: different initializations can be trained until they achieve equally high training-set performance, but they will often perform very differently at test time.

<span style="color:red">***However, little else is known about which other factors affect a CNN’s generalization performance when trained on small datasets***</span>.

 There is a pressing need to understand these factors：
 1. first because we can potentially exploit them to improve performance on tasks where few labels are available.
 2. Second they may already be confounding our attempts to evaluate pre-training methods.

> We show that simple statistical properties of the network, which can be easily measured using training data, can have a significant impact on test time performance.

> Surprisingly, we show that controlling for these statistical properties leads to a fast and general way to improve performance when training on relatively little data.
Empirical


### Related researches

1. Empirical evaluations have found that when transferring deep features across tasks, <span style="color:red">***freezing weights of some layers during fine-tuning generally harms performance***</span> ([Yosinski et al., 2014](https://arxiv.org/abs/1411.1792)).
2. These results suggest that, given a small dataset, it is better to adjust all of the layers a little rather than to adjust just a few layers a large amount, and so perhaps <span style="color:red">***the ideal setting will adjust all of the layers the same amount***</span>.


<span style="color:red">***While these studies did indeed set the learning rate to be the same for all layers, somewhat counterintuitively this does not actually enforce that all layers learn at the same rate***</span>.

> Example

Say we have a network where there are two convolution layers separated by a ReLU. Multiplying the weights $\mathbf{W_1}$ and bias $\mathbf{B_1}$ term of the first layer by a scalar $\alpha \mathbf{W_1}, \alpha>0$. And then dividing the weights $\mathbf{W_2}$ (but not bias) of the next (higher) layer by the same constant $\mathbf{W_2}/\alpha$ will result in a network which computes exactly the same function.

However, note that the gradients of the two layers are not the same: they will be <span style="color:red">***divided***</span> by $\alpha$ for the first layer, and <span style="color:red">***multiplied***</span> by $\alpha$ for the second.

$$
\alpha\mathbf{W_1} \Leftarrow \alpha\mathbf{W_1}+\Delta \alpha\mathbf{W_1} , \Delta \alpha\mathbf{W_1} = -\eta \nabla J(\alpha\mathbf{W_1}), \nabla J(\alpha\mathbf{W_1}) = \frac{\partial J(\alpha\mathbf{W_1})}{\partial \alpha \mathbf{W_1}}=\frac{\partial E\{(t-\alpha\mathbf{W_1}^T\mathbf{x})^2\}}{\alpha\partial \mathbf{W_1}}
$$

<span style="color:red">***Worse, an update of a given magnitude will have a smaller effect on the lower layer than the higher layer, simply because the lower layer’s norm is nowlarger***</span>.


> A number of works have already suggested that statisti- cal properties of network activations can impact network performance.

1. Many focus on initializations which control the variance of network activations.
    1.  Krizhevsky et al. (2012) carefully designed their architecture to ensure gradients neither vanish nor explode. However, this is no longer possible for deeper architectures such as VGG (Simonyan & Zisserman, 2015) or GoogLeNet (Szegedy et al., 2015).
    2. Glorot & Bengio (2010); Saxe et al. (2013); Sussillo & Abbot (2015); He et al. (2015); Bradley (2010) show that properly scaled random initialization can deal with the vanishing gradi- ent problem, if the architectures are limited to linear transformations, followed by a very specific non-linearities.
    3. Saxe et al. (2013) focus on linear networks, Glorot&Bengio (2010) derive an initial- ization for networks with tanh non-linearities, while He et al. (2015) focus on the more commonly used ReLUs.
    4. However, none of the above papers consider more general network including pooling, dropout, LRN layers (Krizhevsky et al., 2012), or DAG-structured networks (Szegedy et al., 2015).

### Data-dependent initialization

- 算法1：with-in layer 初始化
1. for each affine layer $k$ do
    1. Initialize weights from a zero-mean Gaussian $W_k\sim N(0,1)$ and biases $b_k=0$
    2. Draw samples $z_0\in \tilde{D}\subset D$ and pass them through the first $k$ layers of the network
    3. compute the per-channel sample mean $\hat{\mu}_k(i)$ and variance $\hat{\delta}_k(i)^2$ of $z_k(i)$
    4. rescale the weights by $W_k(i,:)\leftarrow W_k(i,:)/\bar{\sigma_k}(i)$
    5. set the bias $b_k(i)\leftarrow \beta - \hat{\mu}_k(i)/\hat{\sigma}_k(i)$
2. end for

### with-in layer weight normalization

They aim to ensure that each channel that a layer k + 1 receives a similarly distributed input.

### between-layer scale adjustment

Because the initialization given in "with-in layer weight normalization" results in activations $z_k(i)$ with unit variance, the expected change rate $C_{k,i}^2$ with unit variance.


- Algorithm 2 Between-layer normalization.

1. Draw samples from $z_0\in \tilde{D}\subset D$
2. Repeat
    1. Compute the ratio $\tilde{C}_k=\mathbb{E}_j [\tilde{C}_{k,j}]$
    2. Compute the average ratio $\tilde{C}=(\prod_{k}C_k)^{1/N}$
    3. Compute a scale correction $r_k=(\tilde{C}/\tilde{C}_k)^{\alpha/2}$
    4. Correct the weights and biases of layer $k: b_k \leftarrow r_kb_k$,$W+k \leftarrow r_kW_k$ Undo the scaling $r_k$ in the layer above.
3. until Convergence (roughly 10 iterations)


### weight initialization

Until now, we used a random Gaussian initialization of the weights, but our procedure does not require this.

They explored two data-driven initializations:
1. a PCA-based initialization
2. a k-means based initialization.


### conclusion

Our method is a conceptually simple data-dependent initialization strategy for CNNs which enforces empirically identically distributed activations locally (within a layer), and roughly uniform global scaling of weight gradients across all layers of arbitrarily deep networks.
