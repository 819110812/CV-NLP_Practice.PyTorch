# [Makeup-Go: Blind Reversion of Portrait Edit](http://openaccess.thecvf.com/content_ICCV_2017/papers/Chen_Makeup-Go_Blind_Reversion_ICCV_2017_paper.pdf)

文章提出Component Regression Network (CRN)，在不知道美颜具体操作的情况下将美颜后的照片进行还原。以往对图片还原的研究假设图片处理操作是已知的线性的，本文面对的是非线性的美颜操作，而且操作未知。以往的方法在细节还原上效果不佳，如下图1。
![](https://img-blog.csdn.net/20171208111210839?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvY3NreXdpdA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

以往类似的方法使用Euclidean loss function，而由于该损失函数受大的PCA特征值影响较大，在做回归时，小的特征值被忽略，而这些小的特征值可能影响面部细节的表现，使得图片还原效果变差，这就是所谓的Component Dominanteffect。

## Reference
- [Makeup-Go: Blind Reversion of Portrait Edit](http://www.cnblogs.com/yymn/articles/8470163.html)