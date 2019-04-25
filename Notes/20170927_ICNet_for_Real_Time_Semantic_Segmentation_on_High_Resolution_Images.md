# [ICNet for Real-Time Semantic Segmentation on High-Resolution Images](https://arxiv.org/abs/1704.08545)

本文提出了一个实时性的语义分割网络，Image Cascade Network（ICNet），在兼顾实时性的同时比原来的Fast Semantic Segmentation,比如SQ, SegNet, ENet等大大地提高了准确率，足以与Deeplab v2媲美，给语义分割的落地提供了可能。具体各个网络速度与性能的对比如下图所示：

![](https://pic3.zhimg.com/80/v2-e38c7373449745022532c02714e760f5_hd.jpg)

文章首先对语义网络进行了速度分析。因为整个网络是基于PSPNet修改的，所以整个比较是在此基础上进行的。经过分析，本文认为影响速度最重要的因素是图像分辨率，进而总结了提高速度的方法，分别是：对输入降采样，对特征降采样，或者进行模型压缩。各个方法的效果如下：

## 对输入降采样：
![](https://pic2.zhimg.com/80/v2-953f1825ac8bcb09255b5fa99fb3d09f_hd.jpg)

## 对特征降采样：
![](https://pic2.zhimg.com/80/v2-0eb8fd615e54805b0265b39e59c02a35_hd.jpg)

## 模型压缩：
![](https://pic3.zhimg.com/80/v2-6d44c2b3071221ea3a397c209170eeee_hd.jpg)

所以，在这基础上，本文提出的模型利用了低分辨率图片的高效处理和高分辨率图片的高推断质量两种优点。主要思想是：让低分辨率图像经过整个语义网络输出一个粗糙的预测，然后利用文中提出的cascade fusion unit来引入中分辨率和高分辨率图像的特征，从而逐渐提高精度。整个网络结构如下：
![](https://pic3.zhimg.com/80/v2-66e004d3ee8ce76f1adaa0881929a122_hd.jpg)

其中的CFF(cascade feature fusion unit) 如下：
![](https://pic4.zhimg.com/80/v2-6d90d137cc89cbbc5f47eaa7fc1287cb_hd.jpg)

这样只有低分辨率的图像经过了最深的网络结构，而其他两个分支经过的层数都逐渐减少，从而提高了网络的速度。而这也恰恰是ICNet和其他cascade structures的不同，虽然也有其他的网络从单一尺度或者多尺度的输入融合不同层的特征，但是它们都是所有的输入数据都经过了整个网络，所以它们的计算效率就大大降低了。其中CFF使用dilated convolution可以整合相邻像素的特征信息，而直接上采样就使每个像素只依赖于一个位置。

最后，文章还对网络进行了模型压缩，再次提速。最终，该模型能达到实时进行语义分割处理，同时有较高准确率。

