
##  [Phrase-based Image Captioning]()<br/>



**Phrase representations initializaition**

They define a metric between the **image $$i$$** and a **phrase $$c$$** as a bilinear operation:
$$
f_{\theta}(c,i)=u_c^TV_{Z_i}$$
where: $$i$$ is a image($$i \in I $$), $$z_i$$ is a feature vector extract from pre-trained CNN. $$c$$ is a pharse, $$\theta$$ is a trainable parameters. And $$U=(u_{c1}, ..., U_{c|c|})$$ and $$V$$ are full-matrix(low-rank)

$$u_c$$ is a vector representation for a phrase $$c=\{w_1,...,w_K\}$$ which is then calculated by averaging its word vector representations: $$u_c= \frac{1}{K}\sum_{k=1}^{K}X_{wk}$$. And each phrase $$c$$ composed of $$K$$ words $$w_k$$ is therefore represented by a vector $$X_{wk}\in R^m$$, this can be producted by word representation model pre-trained on **large unlabeled text corpara**

vecotr representations for all phrases $$c\in C$$ can thus be obtained to initialized the metrix $$U$$. Then $$f_{\theta}(c,i)=u_c^TV_{Z_i}$$ can be represented as : 
$$
f_{\theta}(c,i)=(\frac{1}{K}\sum_{k=1}^{K}X_{wk})^TV_{Z_i}$$
$$V$$ is initialized randomly and trained to encode images($$z_i\in R^n$$) in the same vector space than the phrase used for their descriptions.

(2) Training with negative sampling

Each image $$i$$ is described by a multitude of possible phrases $$C^i$$. Consider $$C$$ classifers attributing a score for each phrase. They train a model to disciminate a traget phrase $$c_j$$ from a set of negative phrases $$c_k\in C$$. The minimize the logistic loss function with respect to $$\theta$$: 
$$
\theta \rightarrow \sum_{i\in I}\sum_{c_j \in C^i}(log(1+e^{-u_{c_j}^T})+\sum_{c_k \in C^{-}}log(1+e^{+u_{c_K}^T V_{z_i}}))$$

(3) Phrases to sentence

The likelihood of a certain sentence is given by $$P(c_1,c_2,..,c_l)=\Pi_{J=1}^l P(c_j|c_1,...,c_{j-1})$$, it can be approximated with a trigram language model: $$P(c_1,c_2,..,c_l)=\Pi_{J=1}^l P(c_j|c_{j-2},c_{j-1})$$<br/>

They constrain the decoding algorithm to include prior knowledge on chunking tages$$t \in \{NP,VP,PP\}$$

$$
\Pi_{j=1}^l \sum_{t}P(c_j|t_j=t,c_{j-2},c_{j-1})P(t_j=t|c_{j-2}c_{j-1})$$
