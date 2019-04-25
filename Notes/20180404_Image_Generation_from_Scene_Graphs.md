# [Image Generation from Scene Graphs]()

在ICCV2017, [StackGAN](https://github.com/hanzhanggit/StackGAN-v2)已经能做到很好的文本到图片的生成了，但是对于语义简单的句子，StackGAN还能hold住，可是句子中有多个物体且位置关系复杂的话，这个生成的图像完全不能用了。为了解决这个问题，该文想出了一个办法：<font color="red">先把文本处理一下，把句子中的物体及他们的相对位置用一个物体关系图(Scene Graph)表示出来，然后再交给模型处理</font>。

![](https://arxiv-sanity-sanity-production.s3.amazonaws.com/render-output/80976/x2.png)

为了生成更符合物理世界规律的图像，生成过程中所用到素材必须取自真实世界的图像。因此，第一个挑战就是要构建一个能处理真实图像的输入处理器。除此之外，生成的每一个物体都必须看起来真实，而且能正确反映出多个物体的空间透视关系。最后一个，就是整个图中所有物体整合到一起，得是看起来是自然和谐不别扭的。

## Method

选Visual Genome和COCO两个数据集里的图片作为素材源。只挑那些含有3~8个物体的图片。然后把这些图片人工地给出物体关系图。像这样：

![](https://pic2.zhimg.com/80/v2-e052f5bdec7144a3b76cc606f0099712_hd.jpg)

然后用模型预测物体之间的位置，大概给出一个图片元素的布局。

![](https://pic2.zhimg.com/80/v2-3c434c2ae1ab6d595a63dc71aef33e2e_hd.jpg)

### Graph Convolution Network
作者使用了graph convolution network（由一系列graph convolution layers组成）。

传统2D convolution输入是spatial的特征向量，这里的graph convolution也是类似，只不过输入是graph vectors ($D_{\text{in}}$), 输出维度是$D_{\text{out}}$。具体的来说，对于所有的物体 $o_i\in O$ 和边 $o_i,r,o_j$ 给定输入向量$v_i,v_r \in \mathbb{R}^{D_{in}}$。我们可以通过3个方程$g_s,g_p,g_o$来计算向量 $v_i^{'},v_r^{'}$ .


最后根据多个判别模型保证输出的图像是符合真实感知的。

![](https://pic4.zhimg.com/80/v2-d640651f86ef53b01ead1ddc4c31529c_hd.jpg)

这里GAN loss是：
$$
\mathcal{L}_{GAN}=\mathbb{E}_{x\sim p_{\text{real}}} \log D(x)+\mathbb{E}_{x\sim p_{\text{data}}}\log(1-D(x))
$$
其中 $x\sim p_{\text{fake}}$ 是生成模型的输出。

最后的评价，他们是在Amazon Mechanical Turk平台上找了人帮忙做评估。和StackGAN相比，合成效果好了一倍。
