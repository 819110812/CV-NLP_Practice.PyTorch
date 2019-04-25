## [What makes for effective detection proposals?](http://arxiv.org/abs/1502.05082)


[论文的项目地址](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/object-recognition-and-scene-understanding/how-good-are-detection-proposals-really/).

[ref](http://zhangliliang.com/2015/05/19/paper-note-object-proposal-review-pami15/)

### ***大纲***

根据文章的描述顺序，***以下内容大概会是***：

1. 回顾***object proposal（以下简称为OP）的各种方法***，将其分类。
2. 讨论不同***OP***在图片被扰动之后的在***复现上的鲁棒性***
3. 讨论不同OP在***PASCAL和ImageNet上的Recall***，这里作者提出了Average Recall（简称AR）的一种新的标准
4. 讨论***不同OP对于实际分类的性能比较***（用了DPM和RCNN这两个著名detector进行比较），以及说明了AR是一个跟性能相当相关的标准。

先上一个***效果的一览表格***:

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150519105213.png" width="600" ></p>

注意到这里只列出了可以找到源码的方法，那么，下面一点点开始整理。

### ***各种OP方法的回顾***

作者大致将OP方法分成了两类，一类叫***grouping method***，一类叫***window scoring method***。

前者是指先将图片打碎，然后再聚合的一种方法，比如selective search。

后者是生成大量window并打分，然后过滤掉低分的一种方法，比如objectness。另外还有一些介乎两者之间的方法，比如multibox。

1. ***Grouping proposal methods***

作者将grouping的方法继续细分为三个小类。SP，对superpixel进行聚合；GC，使用种子点然后groupcut进行分割；EC，从边缘图提取proposal。下面分别调一个进行介绍

- ***Selective Search (SP)***: 无需学习，首先将图片打散为superpixel，然后根据人为定义的距离进行聚合
- ***CPMC (GC)***: 随机初始化种子点，然后做graphcut进行分割，反复多次，然后定义了某个很长的特征进行排序。（所以速度超级慢）
- ***MCG (EC)***: 首先用现成方法快速得到一个层次分割的结果，然后利用边缘信息进行聚合。

2. ***Window scoring proposal methods***

不同于前者需要通过聚合小块来生成候选框，这里的方法是先生成候选框，然后直接打分排序来过滤掉低分的候选框。介绍***两种比较出名的方法***

- ***Bing***: 训练了一个简单的线性分类器来通过类似滑窗的方式来过滤候选框，速度惊人地快，在CPU上能够达到ms级别。但是被文献[40]攻击说分类性能不是来自于学习而是几何学。
- ***EdgeBoxes***: 跟selective search一样是一个不需要学习的方法，结合滑窗，通过计算窗口内边缘个数进行打分，最后排序。


3. ***Aliternate proposal methods***

- ***multibox***, 目前笔者所知唯一基于CNN提取proposal的方法，通过CNN回归N个候选框的位置并进行打分，目前在ImageNet的dectection track上应该是第一的。


4. ***Baseline proposal methods***

这里用了***Uniform，Gaussian，Sliding Window***和***Superpixels***作为***baseline***，不是重点就不展开说了。


### ***各种OP方法对于复现的鲁棒性的讨论***

这里作者提出这样的假设：

***一个好的OP方法应该具有比较好的复现能力***，也就是相似的图片中检索出来的object应该是具有一致性的。

验证的方法是对PASCAL的图片做了各种扰动（如Figure 2），然后***看是否还能检测出来相同的object的recall是多少***，根据IoU的严格与否能够得到一条曲线，最后计算曲线下面积得到repeatability。

作者的结论，Bing和Edgeboxes在repeatability上表现最好

### ***各种OP方法的recall***

这里提出了***好的OP方法应该有着较高的recall，不然就要漏掉检测的物体了***。这里讨论了三种衡量recall的方式：

1. ***Recall versus IoU threshold***: 固定proposal数量，根据不同的IoU标准来计算recall
2. ***Recall versus number of proposal windows***: 跟1互补，这里先固定IoU，根据不同的proposal数目来计算recall
3. ***Average recall(AR)***: 作者提出的，这里只是根据不同的proposal数目，计算IoU在0.5到1之间Recall。

数据集方面，作者在PASCAL VOC07和ImagNet Detection dataset上面做了测试。
这里又有不少图，这里只贴一张AP的，其他请参考原论文咯。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150519112811.png
" width="600" ></p>


***还是直接上结论***

- ***MCG， EdgeBox，SelectiveSearch, Rigor和Geodesic***在不同proposal数目下表现都不错
- 如果只***限制小于1000的proposal***，MCG,endres和CPMC***效果最好***
- 如果一开始***没有较好地定位好候选框的位置***，随着IoU标准严格，recall会下降比较快的包括了Bing, Rahtu, Objectness和Edgeboxes。其中Bing下降尤为明显。
- 在***AR这个标准下，MCG表现稳定***；Endres和Edgeboxes在较少proposal时候表现比较好，当允许有较多的proposal时候，Rigor和SelectiveSearch的表现会比其他要好。
- PASCAL和ImageNet上，各个OP方法都是比较相似的，这说明了这些OP方法的泛化性能都不错。

### ***各种OP方法在实际做detection任务时候的效果***

这里作者在OP之后接上了两种在detection上很出名的detector来进行测试，一个是***文献[54]的LM-LLDA（一个DPM变种）***，另外一个自然是***R-CNN***了，值得注意的是，这两个detector的作者都是rbg。。。真大神也。。。

这里用了各种OP方法提取了1k个proposal，之后作比较。
也是直接给作者结论：

- 如果***OP方法定位越准确，那么对分类器帮助会越大***，因为定位越准确，分类器返回的分数会越高：
  <p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150519140548.png" width="600" ></p>

- 在LM-LLDA和R-CNN下，使得***mAP最高的前5个OP方法***都是MCG,SeletiveSearch,EdgeBoxes,Rigor和Geodesic。分数一览如下图。
  <p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150519134819.png" width="600" ></p>

- 通过分析，作者发现***AR和mAP有着很强的相关性***：
  http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150519134830.png
- 作者用***AR作为指导去tuning EdgeBoxes的参数，然后取得了更好的mAP***（提高1.7个点）
  http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150519134857.png

### ***全文的总结和讨论***

***总结：***

- 对于repeatability这个标准，目前的OP方法效果都一般。可能通过对噪声和扰动更加鲁棒的特征能够提高OP方法的repeatablilty。但是repeatability低不代表最后mAP就低，比如SelectiveSearch，所以最后还是看要应用场景。
- 如果OP方法定位越准确，那么对分类器帮助会越大。所以对于OP方法来说，IoU为0.5的recall不是一个好的标准。高recall但是定位不准确，会伤害到最后的mAP
- MCG,SeletiveSearch,EdgeBoxes,Rigor和Geodesic是目前表现最好的5个方法，其中速度以EdgeBoxes和Geodesic为优。
- 目前的OP方法在***VOC07和ImageNet的表现都差不多***，说明它们都有着***不错的泛化性能***。

***讨论：***

- 如果计算能力上去了，OP还有用吗？作者认为***如果运算性能允许的话，滑动窗口加上CNN等强分类器会有着更好的效果***。
***作者观察到在目前OP中使用的特征（比如object boundary和superpixel），不会在分类器中使用；然后OP方法中除了- MultiBox之外就没有其他OP有使用CNN特征。作者期待会有工作能够结合下这两者的优势***。
- 最后，作者对做了三点猜测：之后***top down可能会在OP中起到更加重要的作用***；以后OP和detector的联系会更加紧密；***OP生成的segmentation mask会起到更加重要的作用***。
