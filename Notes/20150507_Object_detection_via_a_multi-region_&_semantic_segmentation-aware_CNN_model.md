
今天连看了Fast RCNN和这一篇，一开始以为这篇会是Fast RCNN的加强版。看了之后发现不是，这篇提出的框架更像是SPP-Net的加强版，因为这篇并没有实现joint training，不同的步骤还是分开来跑的。不禁让人想，如果能够结合这篇和Fast RCNN的所有技巧，VOC07的mAP会不会上80%了啊。。Detection进步确实太快了。

## ***motivation***

对于某个region proposal来说，***如何抽取比较好的特征？是否需要context辅助？是否需要考虑遮挡问题？***

上述就是作者的motivation，如Figure 1，子图1的羊需要context，子图2的船不要context，子图3的车需要考虑遮挡问题。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20150517151916.png" width="400" ></p>


所以该paper的核心研究内容是，如何***更好地localize一个object***，并抽取***好的特征***。
作者做了三件事：

1. 提出一个***multi-region CNN ***来增强特征
2. 提出一个***semantic segmentation-aware CNN***再进一步增强特征
3. 提出一个***CNN-based regression***方法，另外还提出***2个tricks来refine最后的定位***。

下面分开一点一点说。

### ***Multi-region CNN***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20150517151924.png" width="600" ></p>

Figure 2所示便是Multi-region CNN（简称为MR-CNN）在single scale下的给某个object proposal提取特征的过程，***用AlexNet举例，提取一个proposal的步骤是***

1. 用***前5个卷积层***提取到全图的在[conv5时候的feature map]()
2. 对于某个object proposal，将观察范围做[一定的形变和修改]()得到[不同的region]()，比如图中出来4个不同的region。
3. 将region投影到[conv5 feature map]()上，[crop出来对应的区域]()，然后用一个单层的SPP layer下采样到同样的大小
4. 然后各自经过[两个全连接层进一步提取特征]()
5. 最后所有特征[连在一起得到一个长特征]()。

可以看出，跟SPP提取proposal特征的过程很像，多出来是[第2步]()，也就是这里的主要贡献点。

***作者一共提出了4种共10个region：***

1. 原始的region，就是原来的那个[object proposal]()的位置，对应Figure的中[$a$]()
2. [截半]()，对应Figure3的[$b-e$]()
3. [中心区域]()，对应[$g$]()和$h$
4. [边界区域]()，对应[$i$]()和$j$

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20150517160709.png" width="400" ></p>

作者认为这样***multi region的[好处有两个]():***

1. [不同的region是focus在不同]()的物体区域的，所以他们应该是互补的，能够增强特征的多样性
2. 认为这个方法能够有效应对[object proposal时候定位不准确的问题]()，并在6.2和6.3通过实验验证

### ***Sematic segmentation-aware CNN***

这里的motivation是[通过segmentation的特征来辅助detection]()。然后这里训练segmention用的是很出名的[FCN]()的流程了，不过这里[不需要用segmentation的标注]()，而是用[bbox]()就好了，简单粗暴地把bbox里面认为是前景，外面认为是背景即可（也就是如Figure 5的中间一列）。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20150517151940.png" width="400" ></p>

***虽然表面看似这样的标注很粗暴，很多像素都会错标，但是[CNN的纠错能力是很强的]()***，就是将那些标错的pixel都看成是噪声，CNN依然能够根据更多的标对的像素来学习出来一个还不错的模型（如Figure 5的右列）。
用上述的方法训练出来一个[还不错的segmentation CNN]()后，[摘到最后一层，也加到上面的MR-CNN上，进一步增强特征]()。如Figure 4所示。


<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/%E5%B0%8FQ%E6%88%AA%E5%9B%BE-20150517151947.png" width="400" ></p>


### ***Object localization***

这一步，对应的是RCNN或者SPP-Net的最后一步，也就是得到结果之后，[对位置重新进行一次regression]()，***不过这里做了几点的改进***：

1. [使用CNN来训练regressor]()（在RCNN中是使用简单的函数来训练regressor的），具体来说跟Fast RCNN比较像啦，输出是4xC个值，其中C是类别个数，不过这里直接用L2 loss拟合完事。
2. [迭代优化]()，跟DeepFace比较像，也就是，利用[分类器打一个分]()，然后[筛掉低分]()的，对于剩下的高分的proposal[重新回归]()位置，之后根据这个重新回归的位置再利用分类器打个分，然后再回归一次位置。
3. [投票机制]()，上述两步会在[每个object附近都产生不少bbox]()，这里利用上附近的bbox进行投票打分，具体来说，取一个最高分的bbox，然后还有它附近跟他[overlap超过0.5的bbox]()，然后最后的bbox位置是他们的[加权平均]()（权值为overlap）。
