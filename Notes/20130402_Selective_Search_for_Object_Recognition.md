## [Selective Search for Object Recognition](http://koen.me/research/pub/uijlings-ijcv2013-draft.pdf)

项目网址：http://koen.me/research/selectivesearch/

## **What is an object**?

<p align="center"><img src="http://i.imgur.com/cBm4V5f.png" width="600" ></p>

如何判别哪些**region属于同一个物体**？这个问题找不到一个统计的答案：

1. 对于图b，我们可以根据颜色来分开两只猫，但是不能根据纹理来分开。
2. 对于图c，我们可以根据纹理来找到变色龙，但是不能根据颜色来找到。
3. 对于图d，我们将车轮归类成车的一部分，既不是因为颜色相近，也不是因为纹理相近，而是因为车轮附加在车的上面（个人理解是因为车“包裹”这车轮）
4. 所以，我们需要用多种策略结合，才有可能找到图片中的所有物体。
5. 另外，图a说明了物体之间可能具有的层级关系，或者说一种嵌套的关系——勺子在锅里面，锅在桌子上。

## Multiscale

<p align="center"><img src="http://i.imgur.com/rVuYX8g.png" width="600" ></p>

由于物体之间存在**层级关系**，所以Selective Search用到了**Multiscale的思想**。从上图看出，Select Search在不同尺度下能够找到不同的物体。

注意，这里说的不同尺度，**不是指通过对原图片进行缩放，或者改变窗口大小的意思**，而是，通过**分割的方法将图片分成很多个region**，并且用**合并（grouping）的方法将region聚合成大的region**，重复该过程**直到整张图片变成一个最大的region**。这个过程就能够生成**multiscale的region了**，而且，也符合了上面“***物体之间可能具有层级关系***”的假设。

## ***Selective Search方法简介***
使用[Efficient GraphBased Image Segmentation](http://cs.brown.edu/~pff/segment/)中的方法来得到region

1. 得到所有region之间***两两的相似度***
2. ***合并最像的两个region***
3. 重新计算新合并region***与其他region的相似度***
4. ***重复***上述过程直到整张图片都聚合成***一个大的region***
5. 使用一种***随机的计分方式给每个region打分，按照分数进行ranking，取出top k的子集***，就是selective search的结果

细节看下面两节。

## ***策略多样化（Diversification Strategies）***

有两种多样化方法，一个是***针对样本的颜色空间***，另一个针对***合并时候计算相似性的策略***。
采用了***8种颜色空间***，包括***RGB，灰度图，Lab***，等等
采用了***4种相似性***：颜色相似性（对应Figure1a的情况），纹理相似性（对应Figure1b的情况），小物体先合并原则，物体之间的相容性（对应Figure1d的情况）

## ***如何对region打分***？

这里不是太确定，但是按照作者描述以及个人理解，觉得确实就是***随机地打分***。

对于***某种合并策略$j$***，定义$r_i^j$为位置在$i$的region，其中$i$代表它在合并时候的所位于的层数（i=1表示在整个图片为一个region的那一层，往下则递增），那么定义***其分数为$v)i^j=RND\times i$，其中RND为[0, 1]之间的一个随机值***。

## ***使用Selective Search进行Object Recogntion***

<p align="center"><img src="http://i.imgur.com/hu5G1TI.png" width="600" ></p>

大致流程如上图。用的是传统的“特征+SVM”方法：

1. 特征用了HoG和BoW
2. SVM用的是SVM with a histogram intersection kernel
3. 训练时候：正样本：groundtruth，负样本，seletive search出来的region中overlap在20%-50%的。
4. 迭代训练：一次训练结束后，选择分类时的false positive放入了负样本中，再次训练

## 评估（evalutation）

用数据和表格说明了文中的方法有效。
这部分写了很长，具体不表。
