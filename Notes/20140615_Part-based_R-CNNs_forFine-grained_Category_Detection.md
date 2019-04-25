## [Part-based R-CNNs forFine-grained Category Detection](http://arxiv.org/abs/1407.3867)

ECCV14的oral，论文出处：http://www.cs.berkeley.edu/~rbg/papers/part-rcnn.pdf

## ***概述***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141110155055.png" width="800" ></p>

fine-grained的核心挑战是***定位part***，于是这个工作在RCNN上加了part，主体流程见上图：

- 提取***region proposal***，这里用的selective search
- 找到***物体的bbox***和***part的bbox***，这里需要用到空间信息进行重打分
- 之后***将bbox的特征提取出来***，然后分类

## ***Part-based RCNNs***

训练一个有part的RCNN，满足了两个条件：

1. proposal里面***覆盖了95%的part***;
2. part***有标记信息***，可进行***有监督学习***

但是其实具体来说，跟RCNN框架还是有点差别的：

1. 都用ImageNet的模型做pretrain，但这里***finetuning一个200的分类器（这里对应了200种鸟）。而RCNN是tuning200类+1背景的***。
2. 之后得到***特征***后，用***SVM训练root和part的分类器***。RCNN中没有part分类器（而且为啥part的分类器不在CNN中tuning之后再重新训练SVM呢？）


## Geometric constraints

这里需要对上述root和part进行重新打分，利用到的是part相对于root的空间信息，用式子(1)表示：
$$
\mathbf{X}^*=\text{argmax}_X\Delta (X)\prod_{i=0}^n d_i(x_i) \qquad(1)
$$

上面的△(X)表示某种约束。

首先给出第一个约束是part的bbox应该要几乎都在root的bbox里面（最多只有10个像素能在外面），对应式子(2)和(3)：

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141110213318.png" width="500" ></p>

然后给出一个更强的约束是，***part相对于root应该是有一个“默认”的位置的（比如鸟头应该在上方等***，于是有基于第一个约束有了第二个约束，对应式子(4)：
$$
\Delta _{geometric}(X)=\Delta_{box}(X)(\prod_{i=1}^n\delta_i(x_i))^{\alpha}
$$

其中$\delta_i$代表对part的位置的某种建模方式，文中提到了两种，分别是基于***多高斯和最近邻的***，对应下面：

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141110214248.png" width="500" ></p>

基于***最近邻的定位效果***见这个：

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141110214356.png" width="500" ></p>

## ***Fine-grained categorization***

用上述方法得到了root和part的位置后，用CNN提取特征，然后拼成长向量，用SVM求解。

## ***Evaluation***

用的数据集是Caltech-UCSD bird dataset(CUB200-2011)，有1w+的图片，200类鸟（即平均每类鸟50+张左右），标记有bbox和15个part的位置。
细节：

- 作者只用到了***头部和身体这两个part***。
- CNN特征取的是***fc6***

对比结果如下：

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141111100054.png" width="500" ></p>

大致意思是，提升10%以上了。

在给出bbox时候，是state-of-the-art（其中Oracle82%是因为测试阶段也用了bbox还有part标注）。在不给出bbox的时候，因为太难基本没有其他人做，这个方法依然是state-of-the-art。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141111102825.png" width="500" ></p>

上图给出了***去掉part特征时候的结果，也就是只利用空间信息来定位object，依然有提高，说明了空间约束有助于提高结果***。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141111103034.png" width="500" ></p>

上图对应selective search给出的proposal的召回率。在ol>0.5时候，基本都能够召回。在ol>0.7时候，***召回率会大幅度下降***。所以作者认为***目前方法的bottleneck在于proposal方法***。

## ***总结***

不愧是oral，感觉挺有启发。
提高10%以上主要功劳应该是在CNN特征。
其他贡献是part的定位和几何约束，大概能提高1~2%个点。
