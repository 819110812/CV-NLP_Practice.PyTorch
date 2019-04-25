## [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120)

[Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](http://arxiv.org/abs/1312.6120) Andrew M. Saxe, James L. McClelland, Surya Ganguli. arxiv (2013).

#### ***Key Points***

This work aims to start ***analyzing a gnawing question in machine learning***:

<span style="color:red">***How do deep neural networks actually work***</span>

In particular, for a given set of input-output pairs $\{x^{\mu},y^{\mu}\}$, ***how do the network weights evolve[进化]***, and how ***fast can the network converge***?

To this end, the authors take a close look at the simplest form of deep networks, <span style="color:red">***deep linear networks***</span>, and ***describe the dynamics on learning the network weights over time***. While a simple case, the intuition[直觉] they build for the linear case gives the authors <span style="color:red">***a way to discuss intelligent ways of initializing deep networks***</span>. The ensuing discussion about network initialization focuses on a kind of <span style="color:red">***variance-invariance over network layers***</span>, a concept which the authors also discuss in the concept of nonlinear networks.


> ***The paper is roughly divided into three sections:***

- The first third of the paper discusses an ***exact analysis*** of the ***learning process for a linear neural network with one hidden layer***.
- The second third of this paper ***extends the ideas and intuition*** for a single hidden layer to networks with multiple hidden layers, and further develop the idea of ***greedy network initialization based on the network dynamics***.
- In the final section of the paper the authors look at the properties ***each of greedy and random network initialization*** and ***discuss the implications for learning rates in both linear and nonlinear networks***.

> ***For the first third of the paper***

the network being analyzed consists of:

- ***input layer $x\in\mathbb{R}^{N_1}$***
- ***output layer $y\in\mathbb{R}^{N_3}$***
- ***hidden layer $h\in\mathbb{R}^{N_2} $***.

- The weights connecting layer ***one to layer two*** are $W^{21} \in\mathbb{R}^{N_2,N_1}$
- The weights connecting layer ***two to layer three*** are $W^{32} \in\mathbb{R}^{N_3,N_2}$.

In a linear network the ***back-propagation learning rule*** simply projects the error of each training sample down the network. Taking the limit of small step sizes on the update rule and the expectation of the step with respect to the training set, ***the authors are able to write the learning rule as the set of differential equations***:

$$\tau\frac{dW^{21}}{dt}={W^{32}}^T\left(\Sigma^{31}-W^{32}W^{21}\Sigma^{11}\right),\qquad\tau\frac{dW^{32}}{dt}=\left(\Sigma^{31}-W^{32}W^{21}\Sigma^{11}\right){W^{21}}^T ,$$

where $\tau$  is a constant, $\Sigma^{11} = E[xx^T]$  is the input covariance matrix and $\Sigma^{31} = E[yx^T]$  is the input-output cross-covariance matrix. ***To simplify the analysis***, the authors ***assume appropriate whitening*** to at the input to set $\Sigma^{11} = I$ and replace $\Sigma^{31}$ with its SVD $\Sigma^{31} = USV^T$.

This replacement allows the authors to analyze a simpler set of equations for an equivalent network that operates in the principal directions of $\Sigma^{31}$.

This equivalent set of network weights can be written in terms of $\overline{W}^{21} = W^{21}V = [a^1,a^2... a^\alpha, ... a^{N_2}]$ and $\overline{W}^{32} = UW^{32}= [b^1,b^2... b^\alpha, ... b^{N_2}]^T$.

***By writing an equivalent set of differential equations in terms of the a‘s and b‘s, we see an interesting phenomenon: the resulting equations essentially solve the cost function***.

$$\frac{1}{2\tau}\left[ \sum_{\alpha} (s_\alpha - {a^\alpha}^T b^{\alpha})^2 + \sum_{\alpha\neq\beta}{a^\alpha}^T b^{\beta} \right]$$

This cost function is essentially saying that the $\alpha^{th}$  row-column couple $\{a^\alpha,b^\alpha\}$ is trying to represent the $\alpha^{th}$  mode of the cross covariance matrix $s_\alpha$, while simultaneously trying to be as orthogonal as possible from all other sets of row-column pairs. ***For a typical matrix, this is essentially trying to find the best rank $N_2$ approximation to the cross covariance matrix (i.e. a dynamical system whose solution should be the SVD of $\Sigma^{31}$)***. The authors continue on to find closed form solutions for very simple cases of $N_2 < \min(N_1,N_2)$ and $a^{\alpha} = b^{\alpha} \propto r^{\alpha}$ . While exact solutions for the values of proportionality are derived over time (i.e. the learning rate), the entire argument only holds for when $r^{\alpha}$ are pre-defined. Since these vectors cannot change naturally once initialized, I feel that a lot of the analysis would be difficult to move to a general case. Additionally, the analysis depends heavily on initializing the network to have orthogonal pairs of $\{a,b\}$‘s: something that is impossible when the hidden layer is overcomplete: a common property of the oft-used convolutional neural networks. As such, the intuition built via this analysis allows the authors to continue onto deeper neural networks.

For deep neural networks, the mathematical details for reducing and analyzing the network learning dynamics becomes more complicated. To reduce the network analysis into an analysis of the rows and columns of a rotated set of weight matrices requires another set of rather stringent assumptions: that the right singular vectors of each weight matrix and match the left singular vectors of the weight matrix from the previous layer. Mathematically, if $W^l = U^lS^l{V^l}^T$, then $V^l = U^{l-1}$. The first and last set of weights are again modified via the left and right singular vectors of the input-output cross-covariance matrix. This additional assumption then allows them to again reduce the learning rules to an update on a set of proportionality variables (i.e. every layer has modes proportional to $r^\alpha$). The authors then derive similar learning curves as in the single hidden layer case.

The authors also use ***the intuition from these analyses to discuss the virtues of greedy and random initialization of the network weights***. For greedy methods they discuss a greedy method from the literature where the network is pre-trained in a two-step process. First, an auto-encoder is used to train the network to predict its own input. Second, the network is fine tuned to adapt to the desired output. The authors note that this procedure produces an input weight matrix proportional to the right singular vectors of the $\Sigma^{31}$, meaning that the subspace chosen for the problem at hand is correct prior to learning. This implies that the subsequent learning will be very fast since the network does not need to adapt the principal directions, but rather only the strength of each layer.

As an alternative, the authors also discuss ***smarter random network initialization: using random orthogonal matrices over random Gaussian matrices***. The intuition here comes from the fact that preservation of statistics across layers can imply faster learning. Alternatively, if a network layer does not preserve the total variance of the inputs from the previous layer, vital information could be lost either in terms of the lack of representation power at the output or by the lack of ability for gradients to effectively modify lower levels of the network. Gaussian matrices are particularly terrible in this regard since they are almost guaranteed to have many small singular values. This implies that many vectors, either coming up or down the network, will be severely attenuated, hindering learning. Random orthogonal matrices are guaranteed to preserve the norms of vectors, and consequently are invaluable in this regard. While interesting in its own right, this observation allows the authors to begin to talk about nonlinear networks. This is because the authors state that the greedy methods for initialization might not carry over well to the nonlinear case. In particular the greedy initialization observations depended on the relationship between the SVD of $\Sigma^{31}$, which in turned was intuition gleaned from the update equations for linear networks. Nonlinear networks have a much more complex set of dynamics.

***Using the ideas of energy preservation and random orthogonal matrices***, the authors discuss what it means for a nonlinear network to preserve information across network layers. To accomplish this, the authors present the concept of isometry called the dynamical isometry. Isometries in general discuss the preservation of distances across a function, and the dynamical isometry is particular to discussing the preservation of distances of the back-propagation functions in a deep network. Specifically, the authors describe DI as an isometry condition on the “product of Jacobians associated with error signal back-propagation”.

For linear networks, random orthogonal matrices automatically preserve the norms of vectors, so dynamical isometry follows directly. For nonlinear cases, the authors analyze networks with random orthogonal weights, tanh nonlinearities and a scaling factor g at each layer. The authors plot empirical distributions for the singular values of the Jacobian matrix as a function of the gain at each layer g and the strength of the input layer $q = \frac{1}{N_1}\sum_i^{N_1} {x^1_i}^2$. For $g<1$, the singular values for the Jacobian are very small (less than 1% of the input variance) and therefore the network does not act as an isometry and learning will be slow. Similarly, when $g\geq 1.1$, there are some large singular values, which look troubling to me. The authors mention that this condition is still better than random Gaussian weights. Right around $g = 1$ (what the authors term the “edge of chaos”) the singular values mostly cluster around one, indicating good learning properties.

Overall, I thought ***that this paper begins to ask the right questions about what properties of neural networks helps them learn***. I’m a little skeptical that the analysis of the linear case will yield much for more general classes of networks actually in use, but I think the concepts that the authors are discussing are going to be very important. In particular, I’d be excited to see what comes of this dynamical isometry.
