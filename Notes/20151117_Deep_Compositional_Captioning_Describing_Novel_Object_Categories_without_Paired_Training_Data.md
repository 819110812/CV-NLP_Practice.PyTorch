# [Deep Compositional Captioning:Describing Novel Object Categories without Paired Training Data](http://www.eecs.berkeley.edu/~lisa_anne/current_projects.html)/[Code](https://github.com/LisaAnne/lisa-caffe-public)/Oral

![](http://www.eecs.berkeley.edu/~lisa_anne/Teaser_figure_cvpr16_4.png)



They present a **Deep [Compositional](http://dict.youdao.com/w/eng/corpora/?spc=Composition#keyfrom=dict.typo) Captioner(DCC)** to describe new objects **which are not present in current caption [copora](http://dict.youdao.com/w/eng/corpora/?spc=corpora#keyfrom=dict.typo)**.

![](https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/f273aa82041a0663fbd30722569da57e3f2fecd9/3-Figure2-1.png)

## Deep [Lexical](http://dict.youdao.com/w/Lexical/#keyfrom=dict.top) Classifier

**Deep Lexical Classifier**: It is a CNN which maps images to semantic concepts. Here, they get feature of image : $$\mathbf{f_I}$$

**Language Model**: The language model learns **sentence structure** using **only unpaired text data** and includes an **embedding layer** which maps a **one-hot-vector word representation** to a lower dimensional space, an LSTM, and **a word prediction layer**.


**Caption Model**: The caption model integrates the **lexical classifier** and the **language model** to learn a joint model for image description.

