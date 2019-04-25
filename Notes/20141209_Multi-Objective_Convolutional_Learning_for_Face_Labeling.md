## [Multi-Objective Convolutional Learning for Face Labeling](https://eng.ucmerced.edu/people/jyang44/papers/cvpr15_faceparsing_final.pdf)


<iframe width="560" height="315" src="https://www.youtube.com/embed/Vgo3Y4iyxGk" frameborder="0" allowfullscreen></iframe>

## ***动机***

该文章解决的是***face labeling***的问题，如Figure 1，输入是原图(a)，想要得到的是结果图(d)，可以看成是semantic segmentation的一个子问题。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150512152507.png" width="800" ></p>

### ***CNN+CRF***

我觉得本文最厉害的一点是***将CRF的公式转化为了可以跟CNN联合求解的形式***。下面会描述这个过程。

公式1中用***CRF对问题建模***，X是输入的原始图像，Y是输出的label map，$E_{u}$表示CRF的unary项，这里可以理解为***X中的一个patch决定了Y中的某个点的输出***；$E_b$是binary项，表示$Y$中两个点$y_i$和$y_j$的关系由他们在X中对应的overlapping patch来决定。

$$
E(\mathbf{Y},\mathbf{X})=\sum_{i\in V}E_u(y_i,x_i)+\lambda \sum_{(i,j)\in V}E_b(y_i,y_j,x_{ij}) \qquad (1)
$$

关键是对于上面的公式，怎么转化为***一个CNN能够求解的形式***。

对于unary项，很自然就能对应上可以***将能量函数设为为softmax的形式***。

关键是对于binary项，可以通过引入一个额外的label $z_{ij}$来转化为一个二值问题，那么就可以用sigmoid来拟合了！

明白这一点之后，***CNN+CRF就比较顺理成章了***。

Figure 2给了示意图，作者将这两个loss都接到最后的FC上，联合训练。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150512152628.png" width="800" ></p>

最后在inferece的时候，需要将unary和binary得到的几张map做一个fusion，这一步用graphcut就好了:

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150512154402.png" width="800" ></p>


笔者觉得就上面而言，已经是一个挺漂亮的工作了。但是这篇paper还没有完，下面还有增加prior，full image inference，upsampling等工作。

### Nonparametric prior

face labeling虽然是semantic segmentation的子问题，但是也有自己的一些prior，比如人脸是一个强结构性的object，所以可以先通过估计一个label的概率图来提供prior。
<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150512152637.png" width="800" ></p>


### Full image inference
跟FCN一样，也是将全连接层换成1x1的卷积核，考虑到这是FCN的同期工作，看来大家都想一块去了。

### Upsampling
这篇文章也考虑到upsampling的过程，不过用的是不同于FCN的方法。这里是通过pixel shift，最后拼接到一起来实现的。

<p align="center"><img src="http://hexo-pic-zhangliliang.qiniudn.com/小Q截图-20150512152647.png" width="800" ></p>

## Appendix: CRF

<iframe width="420" height="315" src="https://www.youtube.com/embed/GF3iSJkgPbA" frameborder="0" allowfullscreen></iframe>
