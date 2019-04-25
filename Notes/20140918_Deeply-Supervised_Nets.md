## [Deeply-Supervised Nets](http://arxiv.org/abs/1409.5185)

项目地址：http://vcl.ucsd.edu/~sxie/2014/09/12/dsn-project/

## ***动机和概述***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141102205625.png" width="800" ></p>

如上图，一般CNN***只在顶层接一个分类器***。DSN认为如果隐层特征更具判别性，对于整体效果会更好。于是在***隐层也接了SVM***.

## ***核心公式实现***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141102205852.png" width="800" ></p>

核心优化的***loss函数***是上图中的(3)，又可以分成两部分来看:

- ***左半部分***是顶层（输出层）的loss，标准的SVM的***square hinge loss***
- ***右半部分***是隐层的loss求和，也是标准的SVM的***square hinge loss***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141102210138.png" width="800" ></p>

BP时候的公式如上，也很直观，就是对square hinge loss进行求导。

对于第二个式子，$\alpha_m$是逐渐减少的，是为了***让隐层的梯度逐渐消失***。这$\gamma$是一个阈值，当隐层的loss小到一定程度时候，就会设置为0。（是为了加速吗？）

## ***具体代码实现***

代码github在这里：https://github.com/s9xie/DSN

目前作者给出了CIFAR10上重现实验的数据和脚本。跑了一下，结果可以重现。

看了代码实现，跟论文提供的公式相比有两个实现上的差别，已经有眼尖的同学发现并在issue中
回帖了，附上链接：

1. https://github.com/s9xie/DSN/issues/1
  这个链接讨论的是$\alpha_m$实现相关，目前代码用的是通过调节lr实现，比较合理的方式是通过调节loss_weight实现。作者当时的代码版本中不包含loss_weight于是没有实现。
2. https://github.com/s9xie/DSN/issues/2
  这个链接讨论的是$\gamma$的实现。在代码中没有$\gamma$. 的相关代码，作者使用的方式是early stopping，也就是通过验证集得到一个较为合适的迭代次数，之后的迭代会将隐层的SVM摘掉。
