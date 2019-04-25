## [CNN:Single-label to Multi-label](http://arxiv.org/pdf/1406.5726v3.pdf)

## ***模型概述***

<p align="center"><img src="http://i.imgur.com/r0S6RRV.png


***三步走***：

1. BING生成***hypotheses***
2. 对于每个hypotheses***缩放到227*227后***，放到CNN中，得到结果
3. 做***max pooling得到结果***

跟R-CNN的思路略相似。下面展开叙述

## ***生成hypotheses***

<p align="center"><img src="http://i.imgur.com/quHHXjP.png" width="600" ></p>

用了***BING+Normalized Cut***，大致流程对照上图：

1. 输入一张***图片***
2. 使用BING提取***1k个hypotheses***，然后用IoU作为距离使用Normalized Cut，聚成了M个cluster。
3. 舍弃掉其中***太小***的或者***长宽比太大***的图片
4. 对于***每个cluster***，提取***top k***作为最后的hypotheses

## ***用CNN pre-train***

<p align="center"><img src="http://i.imgur.com/acriauN.png" width="600" ></p>

也是很常规的方法了：

- 先用AlexNet对ImageNet的cls120w问题训练一个model
- 替换掉model的***最后一层，修改loss，使用上面训练得到model来fine-tuning***。

具体细节参看本文一开始附上的slide。

## ***HSP***

本质上是用来抑制噪声的。用的思路也很简单，直接对所有hypotheses取***max***。

<p align="center"><img src="http://i.imgur.com/5xS6nDz.png" width="500" ></p>

## ***总结***

基本思路类似***R-CNN***，用一种object proposal方法提出一些窗口，然后放到CNN中得到最后的结果。
