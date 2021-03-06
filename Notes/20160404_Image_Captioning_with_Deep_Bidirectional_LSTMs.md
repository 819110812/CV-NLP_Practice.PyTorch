# [Image Captioning with Deep Bidirectional LSTMs](https://arxiv.org/pdf/1604.00790.pdf)

accepted by ACMMM 2016 as full paper and oral presentation, [code](https://github.com/deepsemantic/image_captioning),

TLDR: 这项工作提出了采用端对端训练的深度双向LSTM（长短期记忆）模型，进行图像字幕工作。我们的模型建立了深度的卷积神经网络（CNN）和两个独立的LSTM网络。利用历史和未来上下文信息，它能够在高层次语义空间学习长期视觉语言交互。我们增加不同方式的非线性转换的深度，提出了两种新型深度双向变型模型，来学习视觉语言嵌入。提出数据扩增技术，诸如多作物、多尺度和垂直镜方法，以防止在训练深度模型的过度拟合。我们可视化了双向LSTM内部状态随着时间的推移的变化，定性地分析我们的模型如何将图像“翻译”为句子。我们对提出的模型，在三个基准数据集：Flickr8K，Flickr30K和MSCOCO数据集，进行字幕生成和图像句子检索任务的评估。即使没有额外的集成机制（如目标检测，注意力模型等），我们证明双向LSTM模型实现很有竞争力的性能，相对于当前最先进的字幕生成结果，并显著超越近期检索任务的方法。


![](https://www.dropbox.com/s/632c7s1s4cmv2ur/Image%20Captioning%20with%20Deep%20Bidirectional%20LSTMs.png?dl=1)
