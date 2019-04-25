# [Multi-View Clustering via Deep Matrix Factorization](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/download/14647/14497)


The key to current multi-view clustering (MVC) is to explore complmentary informaiton to benefit the clustering problem. They present ta deep matrix factorization framework for MVC.The semi-nogeative matrix fractorization is adopted to learn the hierarchical semantics of multi-view
data in a layer-wise fasion.

In general, the main contributions of this paper are as follows:
1. A deep matrix factorization is proposed to learn the hierarchical semantics of multi-view data;
2. Enforing the non-negative representation of each view in the final layer to be the same;
3. Introduce graph regularizers to couple the output representation of deep structures.

## Challenges for MVC
Previous MVC approachs leverage the heterogeneous data to achieve the same goal. Different features characteristize different information from the data set, e.g., an image can be described by color, texture, shape and so on. These multiple types of features can provide useful information from different views.

## Non-negative matrix factorization

![](http://img.blog.csdn.net/20160421172752611)

> NMF的思想：V=WH（W权重矩阵、H特征矩阵、V原矩阵），通过计算从原矩阵提取<font color="red">权重和特征</font>两个不同的矩阵出来。<u>属于一个无监督学习的算法，其中限制条件就是W和H中的所有元素都要大于0</u>。

很好奇的就是为什么分解的矩阵式非负的呢，网上流传一种很有利的解释就是<font color="red">非负为了使数据有效，负数对于数据是无效的</font>。论文作者实际的解释是：
1. 非负性会引发稀疏
2. 非负性会使计算过程进入部分分解

![](http://img.blog.csdn.net/20160421173650310)
