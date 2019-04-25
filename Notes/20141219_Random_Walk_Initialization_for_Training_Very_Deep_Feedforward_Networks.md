## [Random Walk Initialization for Training Very Deep Feedforward Networks](http://arxiv.org/abs/1412.6558)

### ***background***

Training very deep networks is an important open problem in machine learning. One of many difficulties is that the norm of the back-propagated error gradient can grow or decay exponentially.

### ***What they do***

Here we show that training very deep ***feed-forward networks (FFNs)*** is not as difficult as previously thought.

They show that the ***successive application of correctly scaled random matrices*** to an ***initial vector*** results in a random walk of the log of the norm of the resulting vectors, and they compute the scaling that makes this walk unbiased.


## ***Introduction***

Since the early 90s, it has been appreciated that deep neural networks suffer from a vanishing gra- dient problem:

1. (Hochreiter, 1991),
2. ([Bengio et al., 1994 : Learning Long-Term Dependencies with Gradient Descent is Difficult](http://deeplearning.cs.cmu.edu/pdfs/Bengio_94.pdf)).
3. ([Bengio et al., 1994 : he problem of learning long-term dependencies in recurrent networks](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=298725&url=http%3A%2F%2Fieeexplore.ieee.org%2Fxpls%2Fabs_all.jsp%3Farnumber%3D298725)),
4. ([Hochreiter et al., 2001 : Gradient flow in recurrent nets: the difficulty of learning long-term dependencies](ftp://ftp.idsia.ch/pub/juergen/gradientflow.pdf)).


The term ***[vanishing gradient/梯度消失]()*** refers to the fact that in a feedforward network (FFN) the ***back- propagated error signal typically decreases*** (or increases) exponentially as a function of the ***distance*** from the final layer.

***[vanishing gradient/梯度消失]()*** problem is also observed in [recurrent networks (RNNs)](), where the errors are back-propagated in time and the error signal decreases (or increases) exponentially as a function of the distance back in time from the current error.

Because of the ***vanishing gradient***, adding many extra layers in FFNs or time points in RNNs does not usually improve performance.


Although it can be applied to both feedforward and recurrent networks, [the analysis of the vanishing gradient problem is based on a recurrent architecture]().

<font size="2">In a recurrent network, ***back-propagation through time involves applying*** [similar matrices](https://zh.wikipedia.org/zh-sg/%E7%9B%B8%E4%BC%BC%E7%9F%A9%E9%99%A3) repeatedly to compute the error gradient. The outcome of this process depends on whether [the magnitudes of the leading eigenvalues of these matrices tend to be greater than or less than one]().</font>

### ***The magnitude of the error gradient FFNs***
$$
\begin{cases}\mathbf{a_d}=g\mathbf{W_d}\mathbf{h}_{d-1}\mathbf{b}_d\\ \mathbf{h}_d=f(\mathbf{a_d})\end{cases}
\Rightarrow
\begin{cases}
\mathbf{\delta}_d=g\mathbf{\tilde{W}_{d+1}}\mathbf{\delta}_{d+1}\\
\mathbf{\tilde{W}_d}(i,j)=f'(a_d(i))W_d(j,i)\\
|\mathbf{\delta}_d|^2=g^2z_{d+1}|\mathbf{\delta}_{d+1}|^2\\
z_d=|\mathbf{\tilde{W}_d}\mathbf{\delta}_d/|\mathbf{\delta}_d||^2
\end{cases}
\Rightarrow
\begin{cases}
Z=\frac{|\mathbf{\delta}_0|^2}{|\mathbf{\delta}_D|^2}=g^{2D}\prod_{d=1}^{D}z_d \\
\ln(Z)=D\ln(g^2)+\sum_{d=1}^D\ln(z_d)
\end{cases}
$$

### ***Calcuation of the optimal $g$ value's***


$$
\begin{cases}
\langle \ln(z) \rangle=D(\ln(g^2))+\langle\ln(z) \rangle=0\\
g=\exp(-\frac{1}{2}\langle \ln(z)\rangle)\\
z=|\mathbf{\tilde{W}\mathbf{\delta}}/|\mathbf{\delta}||^2
\end{cases}
\Rightarrow
\begin{cases}
\langle \ln(z) \rangle  \approx  \langle (z-1)\ rangle -\frac{1}{2}\langle (z-1)^2\rangle =-\frac{1}{N}\\
g_{linear}=\exp(\frac{1}{2N})\\
\langle (\ln(z))^2\rangle -\langle \ln(z) \rangle ^2=\frac{1}{2N} \\
\langle \ln(z)\rangle \approx -\ln(2)-\frac{2.4}{\max(N,6)-2.4} \\
\langle (\ln(z))^2\rangle - \langle \ln(z) \rangle^2 \approx \frac{5}{\max(N,6)-4}\\
g_{ReLU}=\sqrt{2}\exp(\frac{1.2}{\max(N,6)-2.4})
\end{cases}
$$


### ***Random walk initialization***

The general methodology used in the ***RandomWalk Initialization*** is to set g according to ***the values given in equations*** $\begin{cases}g_{\text{Linear}}=\exp(\frac{1}{2N})\\g_{\text{ReLU}}=\sqrt{2}\exp(\frac{1.2}{\max(N,6)-2.4})\end{cases}$ for the linear and ReLU cases, respectively. For tanh, the values between 1.1 and 1.3 are shown to be good in practice.

The scaling of the input distribution itself should also be ***adjusted to zero mean*** and ***unit variance in each dimension***.

Poor input scaling will effect the back-propagation through the derivative terms in equation $\mathbf{\delta}_d=g\mathbf{\tilde{W}_{d+1}}\mathbf{\delta}_{d+1}$ for some number of early layers before the randomness of the initial matrices “washes out” the poor scaling.


***A slight adjustment to $g$*** may be helpful, based on the actual data distribution, as most real-world data is far from a ***normal distribution***. By similar reasoning, the initial scaling of the final output layer may need to be adjusted separately, as the back-propagating errors will be affected by the initialization of the final output layer.


In summary, ***RandomWalk Initialization requires tuning of three parameters***: input scaling (or $g_1$), $g_D$, and $g$, the first two to handle transient effects of the inputs and errors, and the last to generally tune the entire network. By far the most important of the three is $g$.
