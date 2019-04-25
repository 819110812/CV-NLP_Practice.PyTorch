## [Improving LSTM-based Video Descriptionwith Linguistic Knowledge Mined from Text](http://arxiv.org/abs/1604.01729)


http://www.cs.utexas.edu/~vsub/


This   paper   investigates   how   [linguistic](http://dict.youdao.com/w/linguistic/#keyfrom=dict.top) knowledge mined from large text corpora can aid the generation of natural language descriptions  of  videos.

Given a sequence of inputs $$(x_1,...,x_T)$$ the LSTM computes the cell memory sequences $$(c_1,...,c_T)$$ and hidden control sequences $$(h_1,...,h_T)$$ as follows:

$$
\begin{cases}
i_t=\text{sigm}(W_{xi}x_t+W_{hi}h_{t-1}+b_i)\\
i_t=\text{sigm}(W_{xf}x_t+W_{hf}h_{t-1}+b_f)\\
i_t=\text{sigm}(W_{xo}x_t+W_{ho}h_{t-1}+b_o)\\
i_t=\text{tanh}(W_{xg}x_t+W_{hg}h_{t-1}+b_g)\\
c_t=f_t\odot c_{t-1}+i_t \odot g_t\\
h_t = o_t \odot \tanh (c_t)
\end{cases} \qquad(1)
$$

During decoding,the model essentially defines a probability over the output sequence $$\vec{y}$$ by decomposing the joint probability into ordered conditionals:

$$
p(\vec{y}|x_1,...,x_T)=\prod_{t=1}^N p(y_t|h_T,y_1,...,y_{t-1})
$$

This is done by applying a softmax function on the  decoder LSTM’sh-sequence.   Hence,  for  a word in the vocabulary (w2V),
$$
p(y_t=w|h_T,\vec{y}<t)=\text{softmax}(W_vh_T+b_v)  \qquad(2)
$$

Thus  the  overall  objective  of  the  network  is  to maximize  log-likelihood  of  the  output  word  sequence.

$$
\log p(\vec{y}|\vec{x})=\sum_{t=1}^N \log p(y_t|h_T, \vec{y}<t) \qquad(3)
$$


