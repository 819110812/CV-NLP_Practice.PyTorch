# [Holistically-Nested Edge Detection](https://arxiv.org/pdf/1504.06375.pdf)/[code](https://github.com/s9xie/hed)

这篇文章属于deeplab-CRF的升级版本：
论文主要提出了使用CNN计算轮廓线(HED)，从而把CRF的部分换成计算轮廓线和一个叫做DT的将FCN结果与HED结合的算法从而达到更精确的语义分割。
HED 全称 Holistically-Nested Edge Detection。其意义是能够产生不同scale的轮廓线。该网络基于VGG，有五层不同level的轮廓线输出。

![](https://leanote.com/api/file/getImage?fileId=57ae94d6ab644135ea04a894)

如上图可以发现，越浅层越容易输出边界信息，越后面的层越能判定一些和语义有关的分割。相比起之前的传统方法，可以发现，在后面的层输出的结果中，那些没有任何语义但却有比较粗的轮廓线没有被输出（这是作者对HED算法在high recall部分有较低的precision的原因的分析）

<center><img src="https://leanote.com/api/file/getImage?fileId=57ae950fab644135ea04a897" width=500></img></center>
网络类型类似于FCN-2S。
作者列出了几个可能的网络架构，然后各自批判了一番。具体原因可以参见原文，这里不多做赘述。

## Formulation

The goal is to have a network that learns features from which it is possible to produce edge maps approaching the ground truth. They consider the objective funciton:

$$
\mathcal{L}_{side}(\mathbf{W},\mathbf{w})=\sum_{m=1}^M\alpha_m\ell_{side}^{(m)}(\mathbf{W},\mathbf{w}^{(m)})
$$

And they use a simpler strategy to automatically balance the loss between positive/negative classes. They also introduce a class-balancing weight $\beta$ on a per-pixel term basis. Index $j$ is over the image spatial dimensions of image $X$.


论文中作者特别对Loss有两点强调。
1.对轮廓的输出其实可以类比成FCN像素级分割对于轮廓线与非轮廓线两种分类。所以非轮廓线的像素点会远远大于轮廓线的像素点。如果使用正常的loss进行训练很容易会训练不出来，因为网络在计算的过程中会认为全都不是轮廓线会产生更小更稳定的loss。所以我们放大轮廓线部分的Loss：


$$
\ell_{side}^{(m)}(\mathbf{W},\mathbf{w}^{(m)})=-\beta \sum_{j\in Y} \log P_r(y_i=1|X;\mathbf{W},\mathbf{w}^{(m)})-(1-\beta)\sum_{j\in Y_{-}}\log P_r(y_i=0|X;\mathbf{W},\mathbf{w}^{(m)})
$$
