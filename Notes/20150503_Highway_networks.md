## Highway Networks

*"Our Highway Networks take inspiration from Long Short Term Memory (LSTM) and allow training of deep, efficient networks (even with hundreds of layers) with conventional gradient-based methods. Even when large depths are not required, highway layers can be used instead of traditional neural layers to allow the network to adaptively copy or transform representations"*


### Highway Networks

一般一个 plain feedforward neural network 有L层网络组成，每层网络对输入进行一个非线性映射变换，可以表达如下:

$$\mathbf{y}=H(\mathbf{x}, \mathbf{W_H})$$

一般后续还有其他处理，例如非线性激活函数， convolutional or recurrent

对于高速CNN网络，我们定义一层网络如下

$$
\mathbf{y}=H(\mathbf{x}, \mathbf{W_H}).T(\mathbf{x},\mathbf{W_T})+\mathbf{x}.C(\mathbf{x},\mathbf{W_C})
$$

We refer to T as the transform gate and C as the carry gate

在这篇文献中我们设置 C=1-T，则得到下式

$$
\mathbf{y}=H(\mathbf{x}, \mathbf{W_H}).T(\mathbf{x},\mathbf{W_T})+\mathbf{x}.(1-T(\mathbf{x},\mathbf{W_T}))
$$

上公式中<span style="color:red">***参数的维数须一致***</span>。


for the Jacobian of the layer transform： $$
\frac{d \mathbf{y}}{d \mathbf{x}}=
\begin{cases}
\mathbf{I}, if \ T(\mathbf{x},\mathbf{W_T})=0\\
H'(\mathbf{x}, \mathbf{W_H}), if \ T(\mathbf{x},\mathbf{W_T})=1
\end{cases}
$$

Thus, depending on the output of the transform gates, a highway layer can smoothly vary its behavior between that of H and that of a layer which simply passes its inputs through

### Training Deep Highway Networks

我们定义 transform gate : $T(\mathbf{x})=\sigma(\mathbf{W_T}^T\mathbf{x}+\mathbf{b_T})$

W是权重矩阵， b是 bias 向量

This suggests a simple initialization scheme which is independent of the nature of H: $b_T$ can be initialized with a negative value (e.g. -1, -3 etc.) such that the network is initially biased towards <span style="color:red">***carry***</span> behavior. This scheme is strongly inspired by the proposal to initially bias the gates in an LSTM network, to help bridge long-term temporal dependencies early in learning

 初始化时可以给b初始化一个负值，相当于网络在开始的时候侧重于搬运行为（carry behavior），就是什么处理都不做。这个主要是受[文献](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=818041&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D818041)启发。我们的实验也证明了这个推测是正确的。
