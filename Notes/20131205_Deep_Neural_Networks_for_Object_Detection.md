## [Deep Neural Networks for Object Detection](https://pdfs.semanticscholar.org/713f/73ce5c3013d9fb796c21b981dc6629af0bd5.pdf)

对应网址：http://nips.cc/Conferences/2013/Program/event.php?ID=4018

<iframe id="iframe_container" frameborder="0" webkitallowfullscreen="" mozallowfullscreen="" allowfullscreen="" width="550" height="400" src="https://prezi.com/embed/tick-uwumd-f/?bgcolor=ffffff&amp;lock_to_path=0&amp;autoplay=0&amp;autohide_ctrls=0&amp;landing_data=bHVZZmNaNDBIWnNjdEVENDRhZDFNZGNIUE1WeEV5OG9mcjk1ZjNLZjc3dWMrTE8vdUw1MWtTVVFPUXJINEpFZ1FQST0&amp;landing_sign=J_Yezgm2x9fNL4mZZ_ylJRb6ZSTgnGTA5FExn0N8Z1c"></iframe>

## ***3 DNN-based Detector***

<p align="center"><img src="http://i.imgur.com/vcRfHIs.png" width="600" ></p>

如上图，将图片分成多个网格，***前景趋向1，背景趋向0***。如果将每个网格看成是一种superpixel，那么其实网络做的是***分割的问题***。当我们得到了网络的分割后，***前景像素其实就是物体的检测位置***。
另外，图上标注使用的是***DBN***，而事实上文章使用的是AlexNet，这里应该是原作者笔误。

## 4 Detection as DNN Regression

网络的loss function如下：

The network is trained by minimizing the $L_2$ error for predicting a ground truth mask $m\in [0,1]^N$ for an image $x$:
$$
\min_{\Theta}\sum_{(x,m)\in D}||(Diag(m)+\lambda I)^{1/2}(DNN(x;\Theta)-m)||_2^2
$$

也就是***每个像素看成独立的二范距离***，并且加了一个***正则约束项Diag(m)+λI***，目的是为了让***网络趋向于输出前景***（原因是一般前景像素占的都是网络的一小部分，所以如果没有正则项，那么网络会偏向于输出一个全为0的平凡解）

其他细节，原图分辨率是225×225，粗网格的分辨率是24×24

## ***5 Precise Object Localization via DNN-generated Masks***

上述方法有三个缺点：

***1. 用唯一的mask***，难以区分识别出来的前景是单个物体还是粘连的多个物体
***2. mask分辨率远低于原图分辨率，mask的一个像素对应的是原图16*16的像素，带来定位不准确难以识别小物体，因为小物体激活的神经元很少，所以难以让对应的部分监测为前景像素***
***3. 作者用了对应的三个办法来解决上述三个问题：***

## ***5.1 Multiple Masks for Robust Localization***

作者定义了五种mask={full,bottom,up,left,right}：

<p align="center"><img src="http://i.imgur.com/NoX8ibX.png" width="600" ></p>

作用：full mask无法分开物体，但如果考虑用left的mask，就能够分开两个物体了。——如果是两个孤立的物体，那么在上述五个mask中至少应该有两个是分开的。
注意，每种mask需要训练一个单独的DNN。

<p align="center"><img src="http://i.imgur.com/gcmmPVH.png

上面公式想要说明的是，mask中每个网格的取值不是严格的两极0和1，而是一个0到1之间的数字，直观的物理含义就是该网格中有多少比例是前景像素

## ***5.2 Object Localization from DNN Output***

这里说明是如何从分割结果得到检测结果。

<p align="center"><img src="http://i.imgur.com/NBPj786.png" width="600" ></p>

公式(2)大概意思是，T(i,j)表示在位置(i,j)上DNN预测的结果（0或者1），bb表示预测的bbox，根据这两者的overlap的大小来赋予赋予一个得分。
公式(3)大概意思是，计算上面5中mask的分数，然后加起来。
然后对于公式(3)中bb如何得到，用的是滑动窗口技术+聚类：

<p align="center"><img src="http://i.imgur.com/03nhBDe.png" width="600" ></p>

最后还另外用到了AlexNet的classification网络来做进一步的验证：

<p align="center"><img src="http://i.imgur.com/oYY0AfT.png" width="600" ></p>

## ***5.3 Multi-scale Refinement of DNN Localizer***

两步走：（1）用滑动窗口（2）用一个循环refine

<p align="center"><img src="http://i.imgur.com/cewZdtb.png" width="600" ></p>

***4. 用滑动窗口***：3个scale，可以看成用三个尺度分别是1×,2×,4×大小的窗口去扫描图片，步长控制在让每个窗口的overlap少于20%，总共大概需要40个滑动窗口（作者强调说这不同于滑动窗口，因为它只需要40个窗口已经很少了，- -个人觉得这就是滑动窗口没错）。最后对于每个scale的多个窗口得到的mask，取maximum（也就是，趋向于取前景像素）。最后每个scale得到5个detection结果，共有15个detection结果。

***5. 循环refine***：对于上面15个detection的结果，将bbox扩大1.2倍后从新放到网络中做测试。

<p align="center"><img src="http://i.imgur.com/RnzI2KO.png" width="600" ></p>

最后，整个测试的流程如上图所示，注意到，[本文给每个类别单独训练了一个DNN]()

## 6 DNN Training

1. 对于***每个类别***，都训练了两个DNN，一个用于回归上面的mask的定位模型，另外用于5.2中的验证步骤的分类模型（二分类）。
2. ***Data Augment***：对于每个样本，根据上文滑动窗口设置的size，都crop出数千个样本，最后每类生成了10M量级的样本，60%是负样本，40%是正样本。
3. 先训练分类模型，再用分类模型去pretrain定位模型，然后fine-grained，这里fine-grained是全网络fine-grained（包括卷积层和全连接层）。
4. 最后，网络的学习率用到***ADAGRAD的方法***，能够***自动估计合适的学习率***。

## ***7 Experiments***
这里开始是实验结果，就是各种state-of-the-art。具体不表。附上效果图：

<p align="center"><img src="http://i.imgur.com/Ae4npCG.png" width="600" ></p>
