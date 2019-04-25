## [Deformable Part Models with CNN Features](http://cvn.ecp.fr/personnel/iasonas/pubs/Savalle_cnndpm_PnA14.pdf)

## ***概述***

发表在ECCV14 workshop的文章， 做的工作有主要有两步

- 得到了AlexNet的conv5的特征金字塔
- 对特征金字塔进行了***降维操作***

## ***得到特征金字塔***

如果使用AlexNet，对于224x224x3的输入图像，conv5后的输出为13x13x256.

也就是在conv5上的一个feature对应了原图上的16x16的一个patch(明明224／13=17.23..笔者不知道为何可以当成16)

考虑到这样太粗糙了(coarse)了，于是作者对原图进行了2倍的oversample，得到了更加精细(fine)的feature map，也就是***一个feature对应原图8x8的patch***。

## ***降维***

原因是由于HoG特征是***32维的***，而***CNN特征是256维***。考虑到运算速度，需要对CNN特征进行降维。
具体降维方法没看，下面摘抄原文：

> To achieve efficiency during training we exploit the LDA-based acceleration to DPM training of [5], using a whitened feature space constructed for CNN features; this reduces the computation time typically by a factor of four. To achieve efficiency during convolutions with the part templates (used both during training and testing), we perform convolutions using the Fast Fourier Transform, along the lines of [2]. This reduces the convolution cost from typically 12 seconds per object (using an optimized SSE implementation) to less than 2 seconds.

## ***效果***

如下图所示:

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/Screenshot%20from%202014-09-13%2011%3A40%3A24.png" width="600" ></p>

***作者的总结是***：

1. 第4行表示，用CNN特征代替到HoG特征后，mAP飙升了15个点
2. 第8行表示，CNN+DPM比RCNN要差4个点

***作者认为按道理说，CNN+DPM应该比RCNN要好的，因为***：

1. 前者用的是sliding windows，后者用的是selective search
2. 前者用的DPM model，后者用的linear-SVM

***作者认为前者反而表现得比后者差的理由有***：

1. 训练数据量变少了（DPMs split the training set into roughly 3 subsets (for the different aspect ratios/mixtures), effectively reducing by 3 the amount of training data ）
2. DPM处理multiscale的方式不够好（DPMs are somewhat rigid when it comes to the kind of aspect ratio that they can deal with, (3 fixed ratios) which may be problematic in the presence of large aspect ratio variatios; by contrast RCNN warps all region proposals images onto a single canonical scale.）

不过笔者觉得作者忽略了一点，就是其实CNN+DPM和RCNN用的特征是不一样的。前者用的是conv5，后者用的是pool5，少了一个下采样，还是有差别的。
