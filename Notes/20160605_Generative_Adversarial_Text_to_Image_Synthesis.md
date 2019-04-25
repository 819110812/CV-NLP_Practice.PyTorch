# [Generative Adversarial Text to Image Synthesis](http://arxiv.org/pdf/1605.05396.pdf)

TLDR: They develop a novel deeparchitecture and GAN formulation to effectively bridge these advances in text and image modeling,  translating visual concepts from characters to pixels.

## code

https://github.com/reedscot/icml2016
https://indico.io/blog/iclr-2016-takeaways/
http://www.slideshare.net/mmisono/generative-adversarial-text-to-image-synthesis

![](http://cdn-ak.f.st-hatena.com/images/fotolife/P/PDFangeltop1/20160530/20160530180703.png)


<font style="font-family:Cursive;color:green">In machine learning, one is always trying to optimize some kind of loss function. Traditionally, this is something like cross entropy or mean squared error. Sadly though, for some use cases, this is not the loss you want your final model to be good at. Take anything in the space of image generation, for example. To demonstrate a common error case, consider a model trying to generate a sharp edge. If a model is uncertain as to where exactly this edge should be, it will try to minimize the loss function to the best of its ability, instead of making a guess like a human would do. If this loss is mean squared error (or really any loss in pixel space) the model will output a blurry line — effectively averaging among all possible predictions. This blurry prediction has a lower loss value than a sharp random guess.</font>

<font style="font-family:Cursive;color:green">This is not what we want in a generative model. Ideally, we want to fool a human looking at the image. Sadly, human decisions can’t simply be plunked into a neural network, so we need some kind of proxy — preferably one that we can take the gradient of. The idea behind adversarial training is to turn this proxy into another neural network so that you essentially have an entire neural network as a loss function. The question then becomes how to train and work with these “adversarial” neural networks.</font>


Their method is to train a deep CNN generative adversarial network(DC-GAN) conditioned on text features encoded by hybrid character-level convolutional recurrent neural network. The generator network $G:\mathbb{R}^Z\times \mathbb{R}^T \rightarrow \mathbb{R}^D$, and discriminator as $D:\mathbb{D}^D\times \mathbb{R}^T \rightarrow \{0,1\}$, where $T$ is the dimension of the txt feature, and $D$ is the dimension of the image, and $Z$ is the dimension of the noise input to $G$.

![](https://www.dropbox.com/s/t792f4yph8p4o07/Generative%20Adversarial%20Text%20to%20Image%20Synthesis.png?dl=1)

![](https://www.dropbox.com/s/qirul1npb3xrhmg/Generative%20Adversarial%20Text%20to%20Image%20Synthesis%20ConvDeconv.png?dl=1)
