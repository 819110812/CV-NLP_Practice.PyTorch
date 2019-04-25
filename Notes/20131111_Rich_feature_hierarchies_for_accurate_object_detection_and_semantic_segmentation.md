## [Rich feature hierarchies for accurate object detection and semantic segmentation](http://www.cs.berkeley.edu/~rbg/#girshick2014rcnn)

项目地址：https://github.com/rbgirshick/rcnn
论文地址：http://www.cs.berkeley.edu/~rbg/#girshick2014rcnn

## ***系统简述***

<p align="center"><img src="http://i.imgur.com/owzgVal.png" width="600" ></p>

***很简单的框架***:

1. 用***selective search***代替传统的滑动窗口，提取出***2k个候选region proposal***
2. 对于每个region，用***摘掉了最后一层softmax层的AlexNet***来提取***特征***
训练出来***K个L-SVM作为分类器***，使用AlexNet提取出来的***特征***作为输出，得到每个region属于***某一类的得分***
3. 最后对每个类别用***NMS***来舍弃掉一部分region，得到detection的结果

## ***2 Object detection with R-CNN***

这部分对上面3个部分进行了更详细的介绍

### ***2.1 Model design***

1. ***Region proposals***:这部分是用来替代传统的sliding windows的，文中提到的方法有objectness，selective search，CPMC等。作者说他最后选择了selective search是为了后面方便做对比试验。
3. ***Feature extraction***：也就是使用CNN，具体来说是AlexNet来提取特征，***摘掉了最后一层softmax***，利用前面5个卷积层和2个全连接层来提取特征，得到一个4096维的特征。一个值得注意的细节是***如何将region缩放到CNN需要的227x227***,作者是***直接忽略aspect ratio之间缩放到227x227（含一个16宽度的边框）***，这样的好处是稍微扩大region，将背景也包括进来来提供先验信息。

### ***2.2 Test-time detection***

略

### ***2.3 Training***

1. ***Supervised pre-training***： 先用imagenet120w的cls数据训练一个模型（出来的效果比alex差2%）
2. ***Domain-specific fine-tuning***: 将上面训练出来的模型用到new task(dection)和new domain(warped region proposals)上，作者将最后一个softmax从1000路输出替换成了***N+1路输出（N个类别+1背景）***。然后将***IoU大于50%的region当成正样本，否则是负样本***。将fine-tuning学习率设置成pre-train模型中的1/10（目的是为了既能学到新东西但是不会完全否定旧的东西）。batch为128，其中正负样本比例是1:3。
3. ***Object category classifiters***:选择SVM对每一类都做一个二分类，在选择样本的时候，区分正负样本的IoU取多少很重要，取IoU=0.5时候，mAP下降5%，取IoU=0，mAP下降4%，作者最后取了0.3，用的是grid search（应该算是穷举逼近的一种）。

## ***3.Visualzation, ablation, and modes of error***

## ***3.1 Visualzing learned feature***

核心思想是在pool5中一个神经元***对应回去原图的227x227中的195x195个像素***(术语是pool5神经元的感受野是195*195)。

可视化的方法是将10M的region在训练好的网络中FP，然后看某个pool5中特定的神经元的激活程度并且给一个rank。出来的高分的图片如下图：

<p align="center"><img src="http://i.imgur.com/9WYwSB3.png" width="600" ></p>

## ***Ablation Studies***

<p align="center"><img src="http://i.imgur.com/KlFMvjT.png" width="600" ></p>

- ***Performance layer-by-layer, without fine tuning***，这里想说明的是，用pool5，fc6，fc7的特征做SVM分类，出来的效果都差不多。作者得到的结论是：CNN的特征表达能力大部分是在卷积层。
- ***Performance layer-by-layer, with fine tuning***，这里想说明的是，pool5经过finetuning之后，mAP的提高不明显，所以卷积层提取出来的特征是具有普遍性的，而fc7经过finetuning后得到很大的提升，说明finetuning的效果主要是在全连接层上。
- ***Comparision to recent feature learning methods***,这里主要说明CNN的特征学习能力比其他方法要好。

## ***3.3 Detection error analysis***

用了一个工具来分析错误

## ***3.4 Boundary-box regression***

作者最后还是用了***regression的方法来进一步定位物体的***，这样子使得mAP提高了4个点。

## ***Appendix***

### ***A. Object proposal transformations***

<p align="center"><img src="http://i.imgur.com/UH8E2ic.png" width="600" ></p>

用了不同的缩放方法，最后作者用的是（d）的第二列的方法，准确率的差别大概是3-5个mAP

### ***B. Positive vs. Negative examples and softmax***

这里作者想讨论的是，为什么训练CNN时候和训练SVM时候，使用了不同的标准来定义正负样本。感觉还是调参调出来的。

另外作者讨论了***为什么最后用SVM替代softmax，因为效果会提升4个点***，作者认为原因在于***softmax中的背景样本是共享的，而SVM的背景样本是独立的***，更加hard，所以能够带来更好的分类效果。

### ***C. Bounding-box regression***

最后作者在***region proposal中再次做了一个regression***。（***这是有道理的，因为当时生成样本时候加了一个16像素宽的padding，所以出来的detection bbox是偏大的，做regression刚好可以应付这个情况***）

### ***F. Analysis of cross-dataset redundancy***

这里说明了VOC和imagenet数据的***重叠是很小的***，1%以下。
