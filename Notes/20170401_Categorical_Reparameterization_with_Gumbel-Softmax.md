# [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/pdf/1611.01144.pdf)

## Reference

- http://blog.evjang.com/2016/11/tutorial-categorical-variational.html
- [The Gumbel-Softmax Trick for Inference of Discrete Variables](https://casmls.github.io/general/2017/02/01/GumbelSoftmax.html)


```python
import torch.nn.functional as F
from torch.autograd import Variable

def sampler(input, tau, temperature):
    noise = torch.rand(input.size())
    noise.add_(1e-9).log_().neg_()
    noise.add_(1e-9).log_().neg_()
    noise = Variable(noise)
    x = (input + noise) / tau + temperature
    x = F.softmax(x.view(input.size(0), -1))
    return x.view_as(input)
```

## Notes

The striking similarities between the main idea of [1] and [2]; namely, the “Gumbel-Softmax trick for re-parameterizing [categorical distributions, 类别分布]” serves as an example of such [simultaneous, 同时发生] discovery in machine learning. Here, the underlying explanation for the [coincidence, 同时发生] is rather obvious: one of the most popular new techniques in [variational inference, 变分推断] and generative modeling; the so-called “re-parameterization trick,” could not be applied to discrete-valued random variables, imposing a significant [hurdle, 障碍]  for the development of new results. Now, with the Gumbel-Softmax trick as an add-on, we can do re-parameterization for inference involving discrete latent variables. This creates a new promise for new findings in areas where the primary objects are of discrete nature; e.g. text modeling.

Before stating the results we start by reviewing the <b>re-parameterization trick</b> and its uses.

> <font style="color:red">The re-parameterization trick is a hot idea, but it fails on discrete data</font>

Let’s first recall the [Law of the Unconscious Statistician (LOTUS),期望值], a simple rule of [calculus, 微积分学] stating that one can compute the expectation of a measurable function $g$ of a random variable $\epsilon$ by integrating $g(\epsilon)$ with respect to the distribution function of $\epsilon$, that is: $\mathbb{E}(g(\epsilon))=\int g(\epsilon) d F_\epsilon$. In other words, <font style="color:red">to compute the expectation of $z =g(\epsilon)$ we only need to know $g$ and the distribution of $\epsilon$. We do not need to explicitly know the distribution of $z$</font>. We can also express the above with the convenient alternative notation: $\mathbb{E}_{\epsilon \sim p(\epsilon)}(g(\epsilon))= \mathbb{E}_{ z \sim p(z)}(z)$.

Now, suppose <font style="color:red">a certain variable $z$ has a distribution that depends on a parameter $\phi$, i.e $z\sim p_\phi(z)$</font>. Moreover, assume one can express <font style="color:red">$z=g(\epsilon,\phi)$ for a certain known function $g$ of the parameters and a certain noise distribution</font> (e.g,  $\epsilon \sim \mathcal{N}(0,1)$) From the LOTUS we know that for any measurable function $f$:

$$
\mathbb{E}_{z\sim p_\phi(z)}(f(z))= \mathbb{E}_{\epsilon \sim p(\epsilon)}(f(g(\epsilon,\phi))).
$$

However, by itself the above formula is not enough. Indeed, in our ML applications we will be faced with the need to compute

$$
\nabla_\phi \mathbb{E}_{Z\sim p_\phi(Z)}(f(Z)) = \nabla_\phi \mathbb{E}_{\epsilon\sim p(\epsilon)}(f(g(\epsilon,\phi))) =  \mathbb{E}_{\epsilon\sim p(\epsilon)}(\nabla f(g(\epsilon,\phi))).
$$

The second equality of the above expression constitutes another feature of the re-parameterization trick: <font style="color:red">we have conveniently expressed $z$ so that expectations of functions of $z$ can be expressed as integrals w.r.t a density that does not depend on the parameter</font>; therefore, we exchange the expectation and gradient or “differentiate under the integral sign”.

The final feature of the re-parametrization trick has to do with how to use the above gradient formula to construct good unbiased estimates of the gradient, a task that we will often involved with (see below). The above formula gives an <font style="color:red">immediate way to obtain an unbiased estimate of the above gradient via Monte Carlo</font>:

$$
\nabla_\phi \mathbb{E}_{z\sim p_\phi(z)}(f(z)) \approx \frac{1}{M}\sum_{i=1}^M \nabla f(g(\epsilon^i,\phi)).
$$

Although not completely understood, in real applications it is seen this re-parameterization based estimate of the gradient exhibits much less variance than of competing estimators.

> Why do we need this trick: <font style="color:red">variational inference, and adversarial generative modeling</font>

Let’s now briefly state two [scenarios, 方案] where we will need to apply the <font style="color:red">re-parameterization trick</font>. They are not mutually exclusive, but here we state them separately for [pedagogical, 教育学] reasons.

1. Variational inference

First, consider variational inference in a latent variable model: we want to have access (i.e evaluate, maximize) to the posterior $p_\theta(z \mid x)$, which will be usually intractable because of the evidence $p_\theta(x)$ in the denominator. In a standard variational setup we find a <font style="color:red">variational approximation $q_\phi(z \mid x)$</font> such that it minimizes its “distance” with the true posterior; or equivalently, it maximizes a lower bound for the log evidence,

$$
\mathcal{L}_{\theta,\phi}(x) = \mathbb{E}_{z\sim q_\phi(z \mid x)}\left(\log(p_\theta(x,z)) - \log(q_\phi(z \mid x))\right).
$$

Learning will be performed, then, by a double maximization of this surrogate function (the ELBO): with respect to $\phi$ in order to bridge the gap with the true posterior, and with respect to $\theta$, to maximize the evidence.

The most direct approach for learning is to do <font style="color:red">gradient descent</font>. However, unfortunately, gradients won’t be available in closed form because they are applied to the often intractable expectation with respect to <font style="color:red">$z\sim q_\phi(z \mid x)$</font>. The standard, then, is to use gradient-based stochastic optimization, a family of methods that solve the above problem by <font style="color:red">assuming that noisy but unbiased estimators of the gradient are available</font>: in that case, we can <font style="color:red">replace the true gradients by the noisy ones as long as we adapt the learning rate of our algorithms accordingly</font>.

Whenever we can <font style="color:red">re-parameterize $q_\phi(z \mid x)$ with respect to a noise distribution, $s(\epsilon)$</font>, we will be able to construct a good unbiased estimator of the ELBO (essential for fast convergence of stochastic optimization methods) based on Monte Carlo samples, as we’ll have:

$$
\mathcal{L}_{\theta,\phi}(x)= \mathbb{E}_{\epsilon \sim s(\epsilon)} (\log (p_{\theta}(x,g(\phi,\epsilon)))-\log (q_\phi(g(\phi, \epsilon)))).
$$

> Adversarial learning of generative models

Another case where we can profit from re-parameterization comes from <font style="color:red">adversarial learning</font>. Suppose we are concerned with learning a (perhaps, very high-dimensional) distribution $p_\theta(x)$ based on observed samples $x_{data}$. Here, we assume the generative parameters $\theta_g$ allow us to express the complex dependencies between the variables in $x$ (e.g. through a deep generative process) so that $p_{\theta_g}(x)$ will be a rich, parsimonious parametric approximation of the empirical distribution $p_{data}(x)$.

As nicely stated in [5], if we assume that for some function $G$ and noise distribution $z$ we have <font style="color:red">$x=G(\theta_g,z)=G(z)$</font>, then, we can cast the (intractable) problem of learning $p_{\theta_g}(x)$ as a minimax game where we simultaneously try to minimize with respect to generative model parameters $\theta_g$ and maximize with respect to the parameters $\theta_d$ of an ad-hoc discriminative network $D$ (e.g. a multi-layer perceptron). In other words, we do:

$$\min_G \max_D\;\; \mathbb{E}_{x\sim p_{data}(x)}(\log(D(x))) - \mathbb{E}_{z\sim p(z)}(\log(1-D(G(z))).$$

Adversarial learning algorithms iteratively sample batches from the data and noise distributions and use the noisy gradient information to simultaneously ascend in the parameters of $D$  (i.e,  $\theta_d$) while descending in the parameters of $G$  (i.e,  $\theta_g$).

<font style="color:red">Notice that if we were not able re-parameterize, we could only express the above as</font>

$$\min_{\theta_g} \max_D \;\;\mathbb{E}_{x\sim p_{data}(x)}(\log(D(x))) - \mathbb{E}_{x\sim p_{\theta_g}(x)}(\log(1-D(x)).$$

Therefore, alternative unbiased estimates of the gradients of the second term (w.r.t  $\theta_g$) would be required if stochastic optimization was attempted.

One alternative (which also applies to the variational setup) is to use the <font style="color:red">score-function estimator (also called the REINFORCE estimator by historical reasons)</font>, which is based on the log derivative trick. However, the variance of this new estimator can be so high that it could not qualify as a realistic alternative to the re-parameterization based estimator.

> Why things can go wrong in discrete cases

The reason why we cannot apply the re-parameterization trick to discrete variables is simple: by [elementary, 基本的] real analysis facts, it is mathematically impossible for a non-degenerate function that maps a continuous set onto a discrete set to be differentiable (not even continuous!). That is, for the functional relation $z=g(\phi,\epsilon),$ it does not make any sense to conceive $\frac{\partial z}{\partial \phi}$ in the discrete case, regardless of the value of $\epsilon$. Alternatively, in deep learning jargon: <font style="color:red">“we cannot backpropagate the gradients through discrete nodes in the computational graph”</font>.

As stated above, in both cases (variational inference and adversarial generative modeling) we still will be able to construct <font style="color:red">alternative estimates of the gradients</font>. However, they may not (and do not!) enjoy the low-variance property of the re-parameterization-based ones.

> The Gumbel distribution and softmax function to the rescue

The Gumbel-softmax trick is an attempt to <font style="color:red">overcome the inability to apply the re-parameterization trick to discrete data</font>. It is the result of two insights: 1) <font style="color:red">a nice parameterization for a discrete (or [categorical, 分类的]) distribution is given in terms of the Gumbel distribution (the Gumbel trick)</font>; and 2) although the corresponding function is non-continuous, it can be made continous by applying using a <font style="color:red">continuous approximation that depends on a temperature parameter</font>, which in the zero-temperature case degenerates to the discontinuous, original expression. Now we describe both components

1. The Gumbel distribution trick

Let’s first recall what a <font style="color:red">Gumbel distribution</font> is. The random variable $G$ is said to have a standard Gumbel distribution if $G=-\log(-\log( U))$ with $U\sim \text{Unif}[0,1]$.

For us, its importance is a consequence that we can <font style="color:red">parameterize any discrete distribution in terms of Gumbel random variables by using the following fact</font>:

<font style="color:purple">Let $X$ be a discrete random variable with $P(X=k)\propto \alpha_k$ random variable and let $\{G_k\}_{k\leq K}$ be an i.i.d sequence of standard Gumbel random variables. Then:</font>

$$X=\arg\max_k \left(\log \alpha_k +G_k\right).$$

In other words, a recipe for sampling from a categorial distribution is: 1) <font style="color:red">draw Gumbel noise by just transforming uniform samples</font>; 2) <font style="color:red">add it to $\log \alpha_k$, which only has to be known up to a normalizing constant</font>; and 3) <font style="color:red">take the value $k$ that produces the maximum</font>.

> Relaxing the discreteness

Unfortunately, <font style="color:red">the $\arg\max$ operation that relates the Gumbel samples, the $\alpha_k$’s and the realizations of the discrete distribution is not continuous</font>. One way of circumvent this, as suggested in [1] and [2] is to relax the discrete set by considering random variables taking values in a larger set. To construct this relaxation we start by recognizing that 1) <font style="color:red">any discrete random variable can always be expressed as a one-hot vector</font> (i.e, a vector filled zeros except for an index where the coordinate is one), by mapping the realization of the variable to the index of the non-zero entry of the vector, and 2) <font style="color:red">that the convex hull of the set of one-hot vector is the probability simplex</font>:

$$\Delta^{K-1}=\{x\in R_{+}^K\quad, \sum_{k=1}^K x_k \leq 1\}.$$

Therefore, a natural way to extend (or ‘relax’) a discrete random variables is by allowing it to take values in the probability simplex. Both [1] and [2] propose to consider the <font style="color:red">softmax map (indexed by a temperature parameter)</font>:

$$f_\tau(x)_k =\frac{\exp(x_k/\tau)}{\sum_{k=1}^K \exp(x_k/\tau)}.$$

with this definition we can define (instead of the discrete valued random variable $X$) the sequence of simplex-valued random variables:

$$X^\tau = (X_k^\tau)_k=f_\tau(\log \alpha+G) =\left(\frac{\exp((\log \alpha_k +G_k)/\tau)}{\sum_{i=1}^K \exp((\log \alpha_{i} +G_{i})/\tau)}\right)_k.$$

The random variable $X^\tau$ defined as above is said to have the <em>concrete</em> distribution (<em>concrete</em> is a <em>portmanteau</em> between <em>con</em>tinuous and <em>dis</em>crete, in the case you haven’t got the pun already), denoted $x^\tau \sim \text{Concrete}(\alpha,\tau)$. Its density (follows simply from the change of variable theorem and some integration) is given by:

$$p_{\alpha,\tau}(x)= (n-1)! \tau^{n-1}\prod_{k=1}^K \left(\frac{\alpha_k x_k^{-\tau-1}}{\sum_{i=1}^K\alpha_i x_i^{-\tau}} \right), x \in \Delta^{K-1}$$

In practice, what really matters is not the specific expression for the density, but the fact that this expression is in <font style="color:red">closed-form and can be evaluated exactly for different values of $x,\alpha$ and $\tau$</font>. Indeed, in the standard methodology in variational inference, by looking at the definition of the ELBO we see we need to evaluate the entropy term  $\mathbb{E}_{z\sim q_\theta(z \mid x)}\left(- \log(q_\theta(z \mid x))\right)$ which explicitly depends on the density.

> Properties

Now we will briefly comment what the above definitions entail, to better understand the nature of this relaxation. <font style="color:red">The following four properties (first three from [1], fourth from [2]) are specially informative</font>

1. Rounding:   $P(X^\tau_k>X^\tau_i,\forall i\neq k) =\frac{\alpha_k}{\sum_{i=1}^K \alpha_i}$.
2. Zero temperature:   $P(\lim_{\tau\rightarrow 0} X^\tau_k =1) = \frac{\alpha_k}{\sum_{i=1}^K \alpha_i}$.
3. Convex eventually:  if $\tau\leq (n-1)^{-1}$ then  $p_{\alpha,\tau}(x)$ is a log-convex function of x.
4. For learning, there is a tradeoff between small and large temperatures.

The <font style="color:red">rounding property</font> is important to conceive actual discrete samples in this relaxed framework. <font style="color:red">The rounding property is a simple consequence of the fact that $\exp(\cdot)$ is an increasing function, and implies that even in this relaxed regime with non-zero temperature, we can still easily sample from our original discrete distribution by mapping points $x$ in the simplex to the one-hot vector (an extreme-point of the simplex) with the non-zero coordinate $k$ so that $x_k$ is the closest to one</font>. This property is exploited in [2] to construct the ‘Straight-Through’ Gumbel Estimator, needed in cases where one does not want to destroy the discrete structure in the hidden model.

The zero temperature property says things are well behaved in that in the zero-temperature limit we recover the original discrete distribution. One way to see this is to think in the logistic function modulated by a slope parameter. For high slope (low temperature) the logistic function will become the Heaviside function, taking only two values.

The log-convex function gives us a good guarantee for optimization: recall we will usually faced with optimization that involves the log density. The convexity property tell us we will be better off as long as the temperature we choose is low enough.

Finally, we (almost literally) cite the tradeoff commented in [2], which is stated as an empirical fact but a proof was unavailable:

> For small temperatures the samples are close to one-hot but the variance of the gradients is large. For large temperatures, samples are smooth but the variance of the gradients is small.

As stated at the beginning, these result have provided a good solution to a common hurdle in machine learning, opening new directions for future research: first (and more immediately), related to the application of this method for the inference/generation of discrete objects. Indeed, [3] is already an example of a direct application of this techniques for the generation of text using GANS. A second sensible direction of research relates to the creation of new discrete reparameterizations based on relaxations, beyond the Gumbel distribution.

## References

1. Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. “The Concrete Distribution: a Continuous Relaxation of Discrete Random Variables.” ICLR Submission, 2017.
2. Eric Jang, Shixiang Gu and Ben Poole. “Categorical Reparameterization by Gumbel-Softmax.” ICLR Submission, 2017.
3. Matt Kusner and José Miguel Hernández-Lobato. “GANS for Sequences of Discrete Elements with the Gumbel-Softmax Distribution.” NIPS workshop on adversarial training, 2016.
4. Diederik P Kingma, Max Welling. “Auto-Encoding Variational Bayes.” ICLR, 2014.
5. Ian Goodfellow. “Generative Adversarial Networks.”, NIPS Tutorial, 2016.
6. Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville and Yoshua Bengio. “Generative Adversarial Nets.” NIPS, 2014
