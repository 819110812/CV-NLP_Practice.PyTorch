# [A Neural Representation of Sketch Drawings](https://arxiv.org/abs/1704.03477)

Recently,  there have been major advancements in generative modelling of images using neural networks as a generative tool. GANs, Variational Inference(VI), and Autoregressive (AR) models have become popular tools in this fast growing area.Most of the work, however, has been targeted towards modelling rasterized images represented as a <font style="color:red">two dimensional grid of pixel values</font>. While these models are currently able to generate realistic,low resolution pixel images, a key challenge for many of these models is to generate images with coherent structure.

## Dataset
They constructed a dataset from sketch data obtained from The Quickdraw A.I. Experiment, an online demo where the users are asked to draw objects belonging to a particular object class in less than 20 seconds. We have selected 75 classes from the raw dataset to construct the quick draw-75dataset. Each class consists of a training set of 70K samples, in addition to 2.5K samples each for validation and test sets.

A sketch is a list of points, and each point is a vector consisting of 5 elements: $(\Delta x, \Delta y, p_1, p_2, p_3)$. The first two elements are the offset distance in the x and y directions of the pen from the previous point.he last 3 elements represents a binary one-hot vector of 3possible states.  The first pen state, $p_1$, indicates that the pen is currently touching the paper, and that a line will be drawn connecting the next point with the current point. The second pen state, $p_2$ ,indicates that the pen will be lifted from the paper after the current point, and that no line will bed rawn next.  The final pen state, $p_3$, indicates that the drawing has ended, and subsequent points,including the current point, will not be rendered.


## Model

Sequence-to-Sequence Variational Autoencoder (VAE), their encoder is a bidirectional $RNN$ [26] that takes in a sketch as an input, and outputs a latent vector of size $N_z$. Specifically, they feed the sketch sequence, $S$, and also the same sketch sequence in reverse order, $S_{\text{reverse}}$, into two encoding RNNs that make up the bidirectional RNN, to obtain two final hidden states:
$$
\begin{cases}
h_{\rightarrow}=\text{encode}_{\rightarrow}(S)\\
h_{\leftarrow}=\text{decode}_{\leftarrow}(S_{\text{reverse}})\\
h=[h_{\rightarrow}, h_{\leftarrow}]
\end{cases}
$$

Then take this final concatenated hidden state, $h$, and project it into two vectors $\mu$ and $\hat{\sigma}$, each of size $N_z$, using a fully connected layer. They convert $\hat{\sigma}$ into a non-negative standard deviation parameterσusing an exponential operation. They use $\mu$ and $\hat{\sigma}$, along with $\mathcal{N}(0,1)$, a vector of IID Gaussian variables of size $N_z$,  to construct a random vector, $z\in \mathbb{R}^{N_z}$, as in the approach for a Variational Autoencoder:

$$\begin{cases}\mu=W_{\mu}h+b_{\mu}\\
\hat{\sigma}=W_{\sigma}h+b_{\sigma}\\
\sigma=\exp (\frac{\hat{\sigma}}{2})\\
z=\mu+\sigma\odot \mathcal{N}(0,1)
\end{cases}$$
Under this encoding scheme, the latent vector $z$ is not a deterministic output for a given input sketch,but a random vector conditioned on the input sketch.

Their decoder is an autoregressive RNN that samples output sketches conditional on a given latent vector $z$. The initial hidden statesh0, and optional cell statesc0(if applicable) of the decoder RNN is the output of a single layer network:
$$[h_0;c_0]=\tanh (W_z z+b_z)$$

Their generated sequence is conditioned from a latent code $z$ sampled from our encoder, which is trained end-to-end alongside the decoder.
$$
p(\Delta x, \Delta y)=\sum_{j=1}^M \prod_j \mathcal{N}(\Delta_x, \Delta_y|\mu_{x,j}, \mu_{y,j},\sigma_{x,j},\sigma_{y,j},\rho_{xy,j})
$$

The next hidden state of the RNN, generated with its forward operation, projects into the output vector $y_i$ using a fully-connected layer:
$$x_i=[S_{i-1}:z]$$
$$[h_i;c_i]=\text{forward}(x_i, [h_{i-1};c_{i-1}])$$
$$y_i=W_yh_i+b_y$$

The vector $y_i$ is broken down into the parameters of the probability distribution of the next data point.
$$[(\hat{\prod_{1}}\mu_x,\mu_y,\hat{\sigma}_x,\hat{\sigma_y},\hat{\rho_{xy}})_1,\cdots,\hat{\prod_{1}}\mu_x,\mu_y,\hat{\sigma}_x,\hat{\sigma_y},\hat{\rho_{xy}})_M (\hat{q}_1,\hat{q}_2,\hat{q}_3) ]=y_i$$

They also apply $\exp$ and $\tanh$ to ensure the standard deviation values are non-negative.
$$\begin{cases}\sigma_x=\exp (\hat{\sigma_x})\\\sigma_y=\exp (\hat{\sigma_y})\\rho_{xy}=\tanh (\hat{\rho_{xy}})\end{cases}$$

The probabilities for the categorical distributions are calculated using the outputs as logit values.
$$\begin{cases}
q_k=\frac{\exp (\hat{q}_k)}{\sum_{j=1}^3\exp (\hat{q}_j)}\\
\prod_k=\frac{\exp (\hat{\prod}_k)}{\sum_{=1}^M \exp (\hat{\prod}_j)}
\end{cases}$$
