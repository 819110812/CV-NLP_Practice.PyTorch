## [DenseNet:Implementing Efficient ConvNet Descriptor Pyramids](https://arxiv.org/abs/1404.1869)

## ***过程概述***

如下图，从左往右是：

1. 原始图片
2. ***缩放到25个分辨率***
3. 将25个不同分辨率的图片***拼成一个大图***（为啥？下面说）
4. 将***大图输入到CNN中***（具体实现是caffe）
5. 得到了大图的feature map
6. 将***大图裁剪成小图，得到特征金字塔***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20140912165233.png" width="600" ></p>

## ***动机***

自然是用来***替代HoG***了，可以参考DPM中HoG的使用方法
<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20140912165726.png" width="600" ></p>

作者最后也是给出了方便的接口:

```matlab
DPM HOG: pyra = featpyramid(image)
DenseNet: pyra = convnet featpyramid(image filename)
```

## ***实现***

- 为何要***拼成大图***
  - 因为Caffe的输出是***固定大小的batch***，为了适应这个限制，那么就将很多张小图拼成统一尺度的大图（1200x1200，或者2000x2000）
  - 这样带来的一个问题是***感受野污染***，也就是因为filter的感受野过大，那么位于***分界线附近的点***会受到其他图片的影响，于是作者的解决办法是加了***32px的padding***
- mean subtraction的问题
  - 通过统计mean pixel来解决，也就是将原来mean image的值再次求平均，得到一个pixel的均值（估计很接近128了。。）

## ***效果***

对比下面两图:

- 上图是用常规方法（类似selective search的region proposal）得到的feature map
- 下图是用本文的方法（直接在原图上做）得到的feature map

作者想claim的是，最后得到的feature map看起来长得差不多。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20140912170612.png" width="600" ></p>

## ***拓展***

这篇文章作者是没有做实验验证效果的。另外一篇ECCV14的[workshop](https://filebox.ece.vt.edu/~parikh/PnA2014/posters/Posters/SavallePnA2014.pdf)用了把***本文的CNN的特征金字塔用在了DPM上***.
