# [Pixel Recurrent Neural Networks](https://arxiv.org/pdf/1601.06759v3.pdf)


Generative image modeling is a central problem in unsupervised learning. Probabilistic density models can be used for a wide variety of tasks that range from image compression and forms of reconstruction such as image in painting and deblurring,  to generation of new images.

When the model is conditioned on <font style="color:red">external information</font>, possible applications also include creating images based on text descriptions or simulating future frames in a planning task.  <font style="color:red">One of the great advantages in generative modeling is that there are practically endless amounts of image data available to learn from.</font>


<center><img src="https://pbs.twimg.com/media/CZprVAZW0AAWaB_.png"></img></center>


## Generating an Image Pixel by Pixel

We can write an image $\mathbf{x}$ as 1-D sequence $x_1,\cdots, x_{n^2}$. To estimate the joint distribution $p(\mathbf{x})$ we write it as the product of the Conditional distribution over the pixels:

$$
p(\mathbf{x})=\prod_{i=1}^{n^2}p(x_i|x_1,\cdots, x_{i-1})
$$
where $p(x_i|x_1,\cdots, x_{i-1})$ is the probability of the $i$-th pixel $x_i$ given all the previous pixels $x_1,\cdots,x_{i-1}$.
