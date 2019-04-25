# [Deep Automatic Portrait Matting](http://xiaoyongshen.me/papers/deepmatting.pdf)

论文使用传统的matting方法，构造了一个portrait image数据库，并在此基础上，提出了一种基于CNN的matting方法。CNN已经在很多计算机视觉任务上有good performance。high-level的有检测、分类、识别、分割，但是他们都无法处理matting的细节问题；low-level的有超分辨、去噪、复原、增强，但是没有分割信息。

论文提出两种函数，第一种是利用CNN将像素分为三类，前景、背景和不确定标签；第二种是一个新的matting layer，通过前向后向传播得到抠图信息。

## Motivation
传统方法主要分为两类，一类是color sampling的方法，以Bayesian matting为代表，通过对前景和背景的颜色采样构建高斯混合模型，但是这种方法需要高质量的trimap，不易获取；另一类是Propagation的方法，根据像素亲和度将用户绘制的信息传播到不确定像素，以Poisson Matting和KNN matting为代表，但是也不是自动抠图。

因此，作者希望通过CNN实现人像自动抠图，其困难点主要有三个：
1. 丰富的抠图细节，例如头发等细节信息
2. 模糊的语义预测
3. 不均匀的抠图值，真正不确定的像素只占整张图的5%左右

## 网络结构
![](https://img-blog.csdn.net/20171116133236882?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzYxNjU0NTk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
网络框架包括trimap labeling 和 image matting，整个过程可以实现整体的前向传导和后向传播。

Image matting takes a color image $I$ as input and decomposes it into background $B$ and foreground $F$ assuming that $I$ is blended linearly by $F$ and $B$.
$$
I = (1-\alpha)B+\alpha F
$$

## Trimap Labeling
输入是原图，输出是每个像素分别是前景F，背景B和不确定区域U的概率，还有一个shape mask，一共四个通道。__Shape mask实际上是通过线性变换将原图和一般的人像结构进行对齐（align）__，个人理解为是向其中加入人像的结构化信息。

## Image Matting Layer
根据trimap labeling的输出，可以使用softmax公式得到前景F和背景B的概率图。 

$$
F=\frac{\exp (F^s)}{\exp (F^s) + \exp (B^s) + \exp (U^s)}
$$

之后使用以下公式得到alpha matte，其中A是alpha matte向量，1是全1向量，B和F分别是前景和背景概率图的对角矩阵。L是输入图像对应的抠图拉普拉斯矩阵，λ是平衡参数。 
$$
\min \lambda \mathcal{A}^TBA+\lambda (\mathcal{A}-1)^TF(\mathcal{A}-1)+\mathcal{A}^T\mathcal{L}\mathcal{A}
$$

## Loss function
论文没有采用一般的L1，L2范式作为loss function，因为label大多数是0和1，这两个范式不利于学习，考虑对每一个alpha值进行加权得到最终的loss函数。 

$$
\mathcal{L}(\mathcal{A}, \mathcal{A}^{gt})=\sum_i (w(\mathcal{A}_i^{gt}\|\mathcal{A}_i-\mathcal{A}_i^{gt}\|)), \qquad w(\mathcal{A}_i^{gt})=-\log (p(A=\mathcal{A}_i^{gt}))
$$

## Data set and training
数据集的制作是获取Flickr上的图片，根据图片质量进行筛选，裁剪到图片中心基本都是人像。利用之前的closed-form matting和KNN matting方法对图像进行标注，同时也是用Photoshop进行细节修复。训练使用数据增强（旋转，缩放，灰度变换），caffe实现，SGD优化，使用FCN-8s模型作为初始化权重。

## 结果
比主流的分割算法的准确率要高，特备强调shape mask的优化作用。 

![](https://img-blog.csdn.net/20171116133728415?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzYxNjU0NTk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)