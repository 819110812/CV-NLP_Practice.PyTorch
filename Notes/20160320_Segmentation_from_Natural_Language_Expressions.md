# [Segmentation from Natural Language Expressions](http://arxiv.org/abs/1603.06180)

- [Code release](https://github.com/ronghanghu/text_objseg.git)
- [Author homepage](http://ronghanghu.com/text_objseg/)



本文研究了一种基于自然语言陈述处理图像分割问题的新方法。这不同于传统的在一个预先定义的语义分类之上的语义分割，例如，对于句子“两个人坐在右边的长椅上”只需要分割为两个右边长椅上的人以及没有其他人站在或坐在另一条长椅上。以前适用于这一任务的处理方法都受限于一个固定的类别和/或矩形域。为对语言陈述进行像素式分割，我们提出了一种端到端可训练周期卷积网络模型，这一模型可同时学习处理视觉与语言信息。我们的模型利用一个卷积LSTM网络将指称语编码为一个向量表示，用一个全卷积网络从一幅图像中提取空间特征，并为目标物体输出一个空间响应图谱。我们在一个基准数据集上展示我们的模型可以从自然语言陈述中得到高品质的分割输出，且优于很多的基线处理方法。

<u>Problem :Given an image and a natural language expression that describes a certain part of the image, we want to segment the corresponding region(s) that covers the visual entities described by the expression.</u>

# Introduction

语义图像分割是计算机视觉的核心问题，通过卷积神经网络使用大量视觉数据集和丰富的语言表达，这一领域得到了重大的进步。尽管这些现有的分隔方法能够精确预测诸如“火车”或“猫”之类的查询种类的像素掩膜，但它们不能对更为复杂的查询，例如自然语言陈述“在汽车右侧穿黑色衬衣的两个人”，进行分割预测。 在这篇论文中我们处理了下面的问题：<u>对于给定的一幅图像和一个自然语言陈述，我们希望能分割出涵盖陈述中所表述的视觉实体的相应区域</u>。于对一个预先确定的目标集或事物种类进行像素级的预测（图1，b），以及示例分割，并且附加地识别一个目标类中的不同示例（图1，c）。它也区别于独立于语言的前景分割，前景分割的目的是在前景（或最突出的）目标上生成一个掩膜。不同于如语义图像分割那样为图像中的每一个像素分配一个语义标签，本文的目的是对给定陈述中的视觉实体生成一个分割掩膜。与固定的一个目标集和事物种类不同，自然语言描述可能也包括了“黑”和“平滑”之类的形容词，“跑”之类的动词，“在右边”之类的空间关系，以及不同视觉实体之间的关系如“那个骑着一匹马的人”。

根据自然语言陈述对图像做分割有着广泛的应用，例如建立基于语言的人机交互来向机器人给出“<u>拿起桌上苹果旁边的罐子</u>”之类的指令。这里，重要的是能利用多词指称语来区别不同的物体实例，但相比于一个范围框，能得到精确的分割也很重要，尤其对于非网格对齐的物体（例如图2）。这对于交互式照片编辑同样有用，其中使用者可以用自然语言指示图像的特定区域或事物来进行处理，例如“涂掉穿红色衬衫的人”，或指示你饭菜的某部分来估计其中的营养，“两大块培根”，来决定是否要吃它。 如第二部分所详细讲述的，先前适用于这一任务的方法仅限于在图像中框定范围框，并且/或者仅限于一个先验的固定种类集合。在本文中，<u>我们提出了一种端到端可训练周期卷积网络模型，可以同时学习处理视觉和语言信息，并且为自然语言陈述所描述的目标图像区域生成分割输出，如图2所示。我们通过一个卷积LSTM网络将语言陈述编码进一个固定长度的向量形式，并利用一个卷积网络从图像中提取空间特征图谱。这一编码陈述与特征图谱之后通过一个多层分类网络以全卷积的方式进行处理，生成一个粗糙响应图谱，然后通过反卷积进行上采样来得到一个目标图像区域的像素级分割掩膜。在一个基准数据集上得到的实验结果显示，我们的模型根据自然语言陈述生成高质量的分割预测，并远优于基线方法</u>。

# Relate work

<u><span style="color:blue">Localizing objects with natural language</span>.</u> Their work is related to recent work on object localization with natural language, where the task is to localize a target object in a scene from its natural language description.

The methods reported in <u>["Natural Language Object Retrieval"](http://arxiv.org/abs/1511.04164)</u> and <u>["Generation and Comprehension of Unambiguous Object Descriptions"](http://arxiv.org/abs/1511.02283)</u> build upon image captioning frameworks such as <u>LRCN ["Long-term recurrent convolutional networks for visual recognition and description"](http://arxiv.org/abs/1411.4389) or mRNN [Deep Captioning with Multimodal Recurrent Neural Networks](https://arxiv.org/abs/1412.6632)</u>, and <span style="color:red">localize objects by selecting the bounding box where the expression has the highest probability</span>.

Their model differs from <u>["Natural Language Object Retrieval"](http://arxiv.org/abs/1511.04164) and ["Generation and Comprehension of Unambiguous Object Descriptions"](http://arxiv.org/abs/1511.02283)</u> in that they do not have to learn to generate expressions from image regions.

In <u>["Grounding of textual phrases in images by reconstruction"](https://arxiv.org/abs/1511.03745)</u>, the authors propose a model to localize a textual phrase by attending to a region on which the phrase can be best reconstructed.In <u> ["Flickr30k Entities"](https://arxiv.org/abs/1505.04870)</u>, Canonical Correlation Analysis(典范相关分析) (CCA) is used to learn a joint embedding space of visual features and words, and given a natural language query, the corresponding target object is localized by finding the closest region to the text sequence in the joint embedding space.

All these previous localization methods can <span style="color:red">only return a bounding box of the target object, and no prior work has learned to directly output a segmentation mask of an object given a natural language description as query</span>.

<u><span style="color:blue">Fully convolutional network for segmentation</span>.</u> Fully convolutional networks are convolutional neural networks consisting of only convolutional (and pooling) layers, which are the state-of-the-art method for semantic segmentation over a pre-defined set of semantic categories (["Fully convolutional networks for semantic segmentation"](https://arxiv.org/abs/1411.4038),["Semantic image segmentation with deep convolutional nets and fully connected crfs"](https://arxiv.org/abs/1412.7062),["Conditional random fields as recurrent neural networks"](http://www.robots.ox.ac.uk/~szheng/CRFasRNN.html),["Multi-scale context aggregation by dilated convolutions"](https://arxiv.org/abs/1511.07122))

<span style="color:red">A nice property of fully convolutional networks is that spatial information is preserved in the output, which makes these networks suitable for segmentation tasks that require spatial grid output</span>.

<u><span style="color:blue">Attention and visual question answering</span>.</u>Recently, attention models have been used in several areas including image recognition, image captioning and visual question answering. In ["Neural image caption generation with visual attention"](https://arxiv.org/abs/1511.07122), image captions are generated through focusing on a specific image region for each word. In recent visual question answering models ["Stacked attention networks for image question answering"](http://arxiv.org/abs/1511.02274),["Exploring question-guided spatial attention for visual question answering"](http://arxiv.org/abs/1511.05234), the answer is determined through attending to one or multiple image regions.

The authors of ["Learning to compose neural networks for question answering"](https://arxiv.org/abs/1601.01705) propose a visual question answering method that can learn to answer object reference questions like <span style="color:red">“where is the black cat”</span> through <span style="color:red">parsing the sentence</span> and <span style="color:red">generating attention maps for “black” and “cat”</span>.

These attention models differ from this work as they only learn to <span style="color:red">generate coarse spatial outputs</span> and the purpose of the attention map is to facilitate other tasks such as image captioning, rather than precisely segment out the object.

# Model

Three components :

1. a <span style="color:red">natural language expression encoder</span> based on a recurrent LSTM network,
2. a <span style="color:red">fully convolutional network</span> to extract local image descriptors and generate a spatial feature map
3. a <span style="color:red">fully convolutional classification and upsampling network</span> that takes as input the encoded expression and the spatial feature map and outputs a pixelwise segmentation mask

## Spatial feature map extraction

Use a CNN(VGG-16, which outputs $D_{im}=1000$ dimensional local descriptors) on the image($W\times H$) to obtain a $w\times h$(<span style="color:red">The resulting feature map size is $w = W/s$ and $h = H/s$, where $s = 32$ is the pixel stride on fc8 layer output</span>) spatial feature map(空间特性图), <span style="color:purple">with each position on the feature map containing $D_{im}$ channels ($D_{im}$ dimensional local descriptors)</span>.

For each spatial location on the feature map, they apply L2-normalization to
the $D_{im}$ dimensional local descriptor at that position in order to obtain a more
robust feature representation. In this way, they can extract a $w\times h\times D_{im}$ spatial feature map as the representation for each image.

Also, to allow the model to reason about spatial relationships such as “right
woman”, two extra channels are added to the feature maps: the $x$ and $y$ coordinate of each spatial location.they use <span style="color:purple">relative coordinates</span>, where the upper left corner and the lower right corner of the feature map are represented
as (−1,−1) and (+1, +1), respectively. In this way, they obtain a $w\times h\times (D_{im}+2)$ representation

## Encoding expressions with LSTM network

To achieve this goal, they take the encoder approach in sequence to sequence learning methods (["Sequence to sequence learning with neural networks"](http://arxiv.org/abs/1409.3215),["On the properties of neural machine translation: Encoder-decoder approache"](http://arxiv.org/abs/1409.1259)).

They first embed each word into a vector through a word embedding matrix, and then use a recurrent LSTM network wth $D_{text}$(We use a LSTM network with $D_{text}$ = 1000 dimensional hidden state) dimensional hidden state to scan through the embedded word sequence.

For a text sequence $S = (w_1, ...,w_T )$ with $T$ words (where $w_t$ is the vector embedding for the $t$-th word), at each time step $t$, the LSTM network takes as input the embedded word vector wt from the word embedding matrix. At the final time step $t = T$ after the LSTM network have seen the whole text sequence, <span style="color:red">they use the hidden state $h_T$ in LSTM network as the encoded vector representation of the expression</span>.

## Spatial classification and upsampling

After extracting the spatial feature map from the image and the encoded expression $h_T$. They want to determine whether or not each spatial location on the feature map belongs the foreground. In their model, this is done by a <span style="color:red">fully convolutional classifier over the local image descriptor and the encoded expression</span>. We first tile and concatenate $h_T$ to the local descriptor at each spatial location in the spatial grid to obtain <span style="color:red">a $w\times h\times D^{\star}$ (where $D^{\star} = D_{im}+D_{text}+2)$ spatial map containing both visual and linguistic features</span>.

Then, we train a two-layer classification network, with a $(D_{cls}$ = 500) dimensional hidden layer, which takes at input the $D^{\star}$ dimensional representation and output a score to <span style="color:red">indicate whether a spatial location belong to the target image region or not</span>.

This classification network is applied in a fully convolutional way over the
underlying $w\times h$ feature map as two $1\times 1$ convolutional layers. The fully convolutional classification network outputs a $w\times h$ coarse low-resolution response map containing classification scores, which can be seen as a low-resolution segmentation of the referential expression. In order obtain a segmentation mask with higher resolution, they further perform upsampling through deconvolution. Here they use a $2s \times 2s$ deconvolution filter with stride $s$.

<span style="color:red">They use the pixelwise classification results (i.e. whether or not a value on the response map is greater than 0) as the final segmentation prediction</span>.

The loss function during training is defined as the average over pixelwise loss:

$$
\text{Loss}=\frac{1}{WH}\sum_{i=1}^W\sum_{j=1}^N L(v_{ij},M_{ij})
$$
$$
L(v_{ij},M_{ij}) = \begin{cases}\alpha_f \log(1+\exp(-v_{ij})), \text{ if } M_{ij}=1\\\alpha_b \log(1+\exp(v_{ij})),\text{ if } M_{ij}=0\end{cases}
$$

where $L$ is the per-pixel weighed logistic regression loss, $\alpha_f$ and $\alpha_b$ are loss weights for foreground and background pixels. $v_{ij}$ is the response value (score) on the high resolution response map and $M_{ij}$ is the binary ground-truth label at pixel $(i, j)$.


# Experiment

Dataset : [ReferIt dataset](http://tamaraberg.com/referitgame/)

## Baseline methods

1. Combination of per-word segmentation.
  Each word in the expression is segmented individually, and the per-word segmenta- tion results are then combined to obtain the final prediction.
2. Foreground segmentation from bounding boxes.
  First use a localization method based on natural language input to obtain a bounding box localization of the given expression, and then extract the foreground segmentation from the bounding box using GrabCut.
3. Classification over segmentation proposals.
  First extract a set of candidate segmentation proposals using MCG, and then train a binary classifier to determine whether or not a candidate segmentation proposal matches the expression.

# Conclusion

To generate a pixelwise segmentation output for the image region described by the referential expression. Propose an end-to-end trainable recurrent convolutional neural network model to encode the expression into a vector representation, extract a spatial feature map representation from the image, and output pixelwise segmentation based on fully convolutional classifier and upsampling.
