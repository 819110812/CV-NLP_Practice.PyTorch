# [Generative Adversarial Networks](https://arxiv.org/pdf/1406.2661v1.pdf),

<!-- <iframe width="560" height="315" src="https://www.youtube.com/embed/KeJINHjyzOU" frameborder="0" allowfullscreen></iframe> -->

The offical code was released, it can be found [here](http://www.github.com/goodfeli/adversarial),[Homepage](http://cs.stanford.edu/people/karpathy/gan/). <u>The motivation of GAN is to generate good samples</u>, it can be used for: <u>start of the art image generation with Laplacian pyramids of GANs</u>. The background of GAN is the "min-max" game, see the demo [here](http://cs.stanford.edu/people/karpathy/gan/).

<center><image src="http://cs.stanford.edu/people/karpathy/gan/gan.png" width="800"></image></center>

In GAN, the <u>Generator ($G$)</u> tries to generate good samples, and <u>Discriminator ($D$)</u> tries to learn which samples come from the <u>true distribution of the input</u> and </u>which samples come from $G$</u>. At the end of training, **we hope that $D$ cannot distinguish between real samples and samples from $G$**.

The loss is $\mathcal{L}=\min_{G}\max_{D}V(D,G)=\mathbb{E}_{x\sim p_{data}(x)}[\log D(x)]+\mathbb{E}_{z\sim p_z(z)} [ \log(1-D(G(z)))]$. During the training process, we update $D$ $k$ times for each update of $G$. Be careful, $\min \log(1-D(G(z)))$ does poorly at the begging of training because $D$ has an advantage (easy to distinguish between samples from $G$ and samples from the real distribution), so instead $\max \log(D(G(z)))$, which has a stronger gradient.


![](https://dl.dropbox.com/s/qzl4x1ce1c3jvbv/GAN.png)

## Theoretical

The Generator $G$ implicitly defines a probability distribution $p_g$. Then if given enough capacity and training time, $G$ can be a good estimator of $p_{data}$. This min-max game has a global optimum for $p_g=p_{data}$.  Consider we have a optimal Discriminator $D$ for any given Generator $G$.

<u><font color="red">*Proposition/命题 1*</font>: For $G$ fixed, the optimal Discriminator $D$ is $D_G^{\star}(\mathbf{x})=\frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x})+p_g(\mathbf{x})}$.</u>
*Proof:* the training criterion for the Discriminator $D$ (given any Generator $G$) is to maximize the quantity $V(G,D)$. Then the problem become maximizing the log-likelihood for estimating the conditional probability $P(Y=y|\mathbf{x})$. For the minmax game, we can reformulated as :
$$
\begin{cases}
C(G)=\max_D V(G,D)\\
C(G)=\mathbb{E}_{x\sim p_{data}}[\log D_G^{\star}(\mathbf{x})]+\mathbb{E}_{z\sim p_{z}}[\log (1-D_G^{\star}(G(\mathbf{z})))]\\
C(G)=\mathbb{E}_{x\sim p_{data}}[\log D_G^{\star}(\mathbf{x})]+\mathbb{E}_{\mathbf{x}\sim p_{g}}[\log (1-D_G^{\star}(\mathbf{x}))]\\
C(G)=\mathbb{E}_{x\sim p_{data}}[\log (\frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x})+p_g(\mathbf{x})})]+\mathbb{E}_{x\sim p_{g}}[\log (\frac{p_{g}(\mathbf{x})}{p_{data}(\mathbf{x})+p_g(\mathbf{x})})]
\end{cases}
$$


<u><font color="red">*Theorem 1*</font>. The global minimum of the virtual training criterion $C(G)$ is achieved if and only if $p_g =p_{data}$. At that point, $C(G)$ achieves the value $-\log 4$.<u> By subtracting this expression from $V(G)=V(D_G^{\star},G)$, we obtain:
$C(G)=-\log(4)+KL(p_{data}||\frac{p_{data}+p_g}{2}+KL(p_g||\frac{p_{data}+p_g}{2}))$.

<u><font color="red">*Proposition 2. If $G$ and $D$ have enough capacity, and at each step of Algorithm 1, the discriminator
is allowed to reach its optimum given $G$, and $p_g$ is updated so as to improve the criterion C(G), then $p_g$ converges to $p_{data}$*</font></u>.
