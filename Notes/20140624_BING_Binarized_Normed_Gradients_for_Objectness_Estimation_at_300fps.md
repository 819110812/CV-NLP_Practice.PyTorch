## [BING: Binarized Normed Gradients for Objectness Estimation at 300fps]()

CVPR2014的ORAL
对应项目主页：http://mmcheng.net/bing/

## ***核心贡献***

一句话概括：***BING是一种objectness特征，可以用它来做object proposal，来加速直接用滑动窗口来做detection的速度***。

***解释***：传统DPM方法是训练***detector***，然后做***滑动窗口***进行搜索，缺点自然是***非常慢***。一个自然的提速思想就是***减少图片的分辨率***，但这样子明显会带来识别效果的下降。

但我们可以这样考虑：***在低像素的条件下，导致我们分不清这是什么物体，但是我们依然可以分辨[这是不是一个物体]()***。也就是，认为***objectness（objects are stand-alone things with well-defined closed boundaries ans centers）***在低像素下依然能够保持。于是作者提出了用一个8*8的梯度级数窗口就能够很好地进行objectness判断。

所以，其实这依然是一种***滑动窗口技术***，不过它不是在原图上做滑动窗口，而是先将原图缩放到
一个小的尺寸（10~320共36个scale），然后在上面做滑动窗口，得到每个窗口的objectness分数，然后排序得到***1k左右的proposal***。（如果直接在原图上做滑动窗口，大概需要500k个窗口）。

## ***模型***

公式：
$$
\begin{cases}
{s}_{l}=\langle{w},{g}_{l}\rangle \\
l = (i,x,y)
\end{cases}
$$

其中$i$是scale的编号，共36个scale

- $l$是第$i$个scale下位置在$(x,y)$下的窗口
- $g_l$就是对应位置的特征矩阵了（这里取梯度的级数为特征）
- $s_l$为该窗口的得分，就是等于$w$和$g_l$的内积（这里$(a,b)$表示内积

另外还认为***不同scale下objectness的标准应该是不一样的***，于是加入***校准项***：

$$
{o}_{l}={v}_{i}\cdot{s}_{l}+{t}_{i}
$$

最后的$o_l$看成是***该窗口最后的objectness得分***。最后就是根据这个得分进行排序来得到前k个proposal的。


## ***特征***

用的***梯度级数（NG）***，很简单的算子,计算方法是：$min(|{g}_{x}|+|{g}_{y}|,255)
$

为了进一步加速运算，使用了***其二进制的估计值，称为BING***。

首先对模型$w$进行二进制估计，算法是：

<p align="center"><img src="http://i.imgur.com/XcOaQJE.png" width="500" ></p>


大概意思是:

- 用符号函数计算目前$w$大致的方向$a_j$
- 将$w$投影到$\alpha_j$上，投影的长度是$\beta_j$
- 从$w$中减去这个投影$\beta_j\alpha_j$
- 重复上述过程，共收集$N_w$个投影，用这些投影来近似表示$w$

之后，能够进一步将每个$a_j$表示成:${a}^{+}_{j}- \overline{a}^{+}_{j}
$

然后对于每个图片的NG，也就是$g_l$，也用二进制来近似：$\mathbf{g}_l=\sum_{k=1}^{N_g}2^{8-k}\mathbf{b}_{k,l}$

从本质上，可以简单理解为***将$g_l$二进制化后，取其最高的$N_g$位***，也就是在量化时候减少了级数。

然后，作者还提出了一种方法，来快速读取NG特征：
<p align="center"><img src="http://i.imgur.com/3dE6XDp.png" width="500" ></p>

大致思想是，如果不进行优化，那么对于每个BING特征我们需要读取64个位置，但其实对于相邻的位置来说，他们的BING特征大部分是共享的，我们不需要每次都重新读取64个位置。
更具体的操作，忘了的话，看回原文。



## ***实验结果***

在VOC2007上面做的，简单来说，效果最好，***然后速度是selective search的3700倍***。而且有不错的泛化性能——在6个类上面做train，另外14类做test，最后效果差不多。

下载源码：http://mmcheng.net/bing/
数据集也能在对应的网页上下到：http://mmcheng.net/bingreadme/
运行了下源代码，需要：

- vs2012环境
- x64 release模式
- 配置好opencv
- 项目属性–C/C++–代码生成–启用增强指令集设为”/arch:SSE2″
