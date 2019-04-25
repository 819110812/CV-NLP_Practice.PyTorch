# [Unsupervised Learning of Stereo Matching](http://openaccess.thecvf.com/content_iccv_2017/html/Zhou_Unsupervised_Learning_of_ICCV_2017_paper.html)

## Cost-volume Computation
用对应的分支来计算cost-volume，输入即左右图像，生成cost-volume。这部分是由八个卷基层构成的双塔结构。每个层后面有normalization和Relu层。这些层对两个图像的每个块都会产生特征向量。这些特征向量再进入correlation layer，计算得到cost-volume，这个correlation层就是用的Dispnet-C中的那个correlation层。

## Cost-volume Aggregation
之前的方法大多使用包边滤波器去聚合cost-volume。我们却使用图像特征网络去学习这个过程中图像的结构。这个网络从两个输入图像中提取特征。这里说的也就是correlation层后面再接一些卷积层去提取特征。

当得到图像特征之后，用联合滤波器整合cost-volume以及输入图像的颜色信息。特征以cost-volume中的每个通道数与输入的颜色信息相融合，然后再连接三个卷积层来产生最终的cost-volume。这是模仿了传统立体匹配方法中的成本聚合过程。这种学习策略更好因为它可以自适应的去找到最合适的参数，细节稍后讨论。

## Disparity Prediction
经过处理过后的cos-volume，用winner-take-all的策略来产生视差映射。然而，argmax操作反向无法求，所以用一个soft argmax的操作。在每个像素求得coat-volume中的最大值的系数。

经过上述三个操作。可以直接端到端的来处理立体匹配问题了

![](https://img-blog.csdn.net/20171023104041881)