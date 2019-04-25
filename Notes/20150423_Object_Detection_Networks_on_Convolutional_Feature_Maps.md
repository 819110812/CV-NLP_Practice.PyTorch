## [Object Detection Networks on Convolutional Feature Maps](http://arxiv.org/abs/1504.06066)

这是CVPR 2015的一篇 关于深度学习和目标检测的文章，作者是MSRA的Shaoqing Ren， Kaiming He等。

[PPT](http://kaiminghe.com/iccv15tutorial/iccv2015_tutorial_convolutional_feature_maps_kaiminghe.pdf)

阅读总结如下（思想重要，实验暂略）：

## 1. 当前的目标检测方法

总体框架是 feature extractor + object classifier。
传统方法：HOG/SIFT/LBP + SVM/boosted/DPM 。
当前方法又细分为三个流派。
第一种 fine-tuned ConvNet + MLP，以R-CNN为代表。
第二种是想把deep ConvNet 和 traditional detector做一个结合。把传统的 feature extractor 升级为 pre-trained deep ConvNet，但是用传统的分类器，即 pre-trained ConvNet +DPM/boosted, 称为Hybrid派。
第三种方法介于R-CNN和Hybrid之间，pre-trained ConvNet + fine-tuned MLP，以SPPnet为代表。 从性能上来说，R-CNN还是占统治地位，SPPnet和R-CNN接近，Hybrid方法一般要比前两者差点。

## 2. 问题的提出

当前方法主流是 deep ConvNet (pre-trained 或者 fine-tuned ) + fine-tuned MLP。
问题：(1) 有没有比 MLP 更好的 region classifier ？ (2) fine tuning 有多重要？

## 3. 方法：

先用一个fixed, pre-trained ConvNet对输入的图像得到卷积的 feature maps。
在得到的feature map上再训练网络做分类，该网络作者称为NoCs ，即Network on Convolutional feature maps)。这里就不局限于MLP了，可以是各种网络。
作者尝试了三种网络：各种深度的MLP， 各种深度的 ConvNet， ConvNet with maxout.这是回答问题(1)。
在训练网络时，每种网络又分随机初始化，和来自 pre-trained 两种初始化策略，回答问题(2)。

## 4. 结论

在相同的网络下，pre-training + fine-tuning work的最好。
精心设计的网络获得的性能提升要比 pre-training + fine-tuning 获得的性能提升大，作者的三种网络中，ConvNet with maxout 最好。
HOG + NoC > HOG + DPM/boosted，说明NoC的发现具有可推广性。
R-CNN 比 Hybrid 效果好，主要是由于其classfier占优势。
