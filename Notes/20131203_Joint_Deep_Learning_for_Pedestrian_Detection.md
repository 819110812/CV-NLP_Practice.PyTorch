## [Joint Deep Learning for Pedestrian Detection](http://www.ee.cuhk.edu.hk/~xgwang/papers/ouyangWiccv13.pdf)

## ***Motivation***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141114154713.png" width="500" ></p>

见上图，一句话概括，将***deformable part***和***occlusion***融合到***CNN***里面做行人检测。

## ***概述***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141114155325.png" width="500" ></p>

整体框架见上图。***流程大致是***：

- 以修改过的YUV特征和edge map作为输出
- 一个卷积层（***应该是看成root***）
- 又一个卷积层（***引入不同大小的卷积核，带有part信息***）
- 处理deformable的一个层
- 处理occlusion的层并得到最后结果


## ***Input data preparation***

这里输出的***不是原始的RGB特征***，而是：

- YUV中的Y
- YUV缩放成1/4，多余填零
- sober边缘检测构成的边缘图

目的，***为了输出多尺度，并且引入边缘信息***。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141114155334.png" width="500" ></p>

作者设置了***不同大小的卷积核，对应不同的part，并且是分层结构***。图中的***黑色部分代表遮挡情况***。
总共有***20个代表part的卷积核***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141114155339.png" width="500" ></p>

对于***某个核形成的一张feature map***，认为是part的激活状况，通过***学习（或者人为设置）deformable layer，并进行融合***，能够得到一张***带deformable part信息的激活图***（上图中的Summed map），之后全图求max得到该part的得分。
[注意到，这里的deformable layer可以引入DPM所用的距离函数，这里假设了标准位置是人为设置的]()。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141114155349.png" width="500" ></p>

这部分是处理遮挡的，目前没有细看。
