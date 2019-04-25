## [DeepID-Net:multi-stage and deformable deep convolutional neural networks for object detection](https://arxiv.org/abs/1409.3505)

论文，ppt，项目出处：http://www.ee.cuhk.edu.hk/~wlouyang/projects/imagenetDeepId/index.html


## ***概述***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030162740.png" width="600" ></p>

从上图可以看出DeepID Net是***在RCNN的流程上增加了更多的步骤进行完善***。下面就一个一个来说咯。

## ***Bounding box rejection***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030163010.png" width="600" ></p>

用来（调参）时候的提速，核心思想是***认为negative里面有很多都是easy的，而我们要的是区分postive和hard neg就好了***，所以可以过滤掉easy neg。

这里的easy neg的定义是在200类中，SVM得分都低于***-1.1***

## ***Deep model training***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030163253.png" width="600" ></p>

模型预训练***考虑了两点***

- Clarifai模型比AlexNet要好（略废话）
- 从ImageNet的cls中预训练，比如在Imagenet的loc中做预训练（作者认为是cls的输入是image level，而loc和det都是object level）

## ***Def Pooling Layer***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030163606.png" width="600" ></p>

引入***deformable part***的思想，在conv5后接入一个类似***part filter***的结构，并且在pooling时候引入了***deformation的思想***——相当于在max pooling上做了改进。

## ***Sub-box features***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030163815.png" width="600" ></p>

相当于后处理的一步了，将***bbox四等分后***，寻找在object proposal中跟***四分有较大overlap的部分***，通过求***ave和max***来增强bbox的特征。

## ***SVM-net***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030164012.png" width="600" ></p>

直接上***hinge loss***来替换原来RCNN中softmax后+SVM的训练结构。用来提速的。

## ***Context modeling***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030164134.png" width="600" ></p>

用***1k cls***的***score***和***200 det***的***score***拉成***1200***的长向量，利用其中的context信息来refine结果。

## ***Multi-stage training***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030164544.png" width="600" ></p>

感觉类似boosting，用于***训练多个fc来提升结果***，大致流程是：

1. 先训练第一个fc
2. fix住conv和第一个fc，然后训练第二个fc
3. 同时训练conv，第一个fc和第二个fc
4. fixconv和第一个和第二个fc，训练第三个fc
5. 同时训练conv，第一二三个fc
6. …..

## ***总结***

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20141030164854.png" width="600" ></p>
