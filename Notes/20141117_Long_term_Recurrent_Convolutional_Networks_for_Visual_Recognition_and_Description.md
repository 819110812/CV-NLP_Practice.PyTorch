## [**Long-term Recurrent Convolutional Networks for Visual Recognition and Description**](http://jeffdonahue.com/lrcn/)/[Code](http://www.eecs.berkeley.edu/~lisa_anne/LRCN_video)  <br/>


They propose a <span style="color:red">*Long-term recurrent convolutional networks(LRCNs)*</span> which combines CNN and long-range temporal recursion and is end-to-end trainable.<br/>

<span style="color:red">Feature extraction: </span>Given visual inputs $v_t, t\in T$, first they use CNN to get feature transformation $\phi_V(v_t)$, where $V$ is the parameter of CNN network, to get a fixed-length vector representation $\phi_t \in R^d; <\phi_1, ..., \phi_T>$.   <br/>
<span style="color:red">Sequence generation: </span> two layer LSTM map the input features to the output $z_t$, which $z_t=h_t$. And $h_1=f_W(x_1,h_0)=f_W(x_1,0)$, then $h_2=f_W(x_2,h_1)$, etc., up to $h_T$.<br/>
<span style="color:red">Final prediction: </span> Use softmax over the ouputs $z_t$. $P(y_t=c)=\frac{exp(W_{zc}z_{t,c}+b_c)}{\sum_{c1\in C}exp(W_{zc}z_{t,c`}+b_c)}$.<br/>
<span style="color:red">Objective function: </span> Minimize the negative log likelihood $L(V,W)=-logP_{V,W}(y_t|x_{1:t},y_{1:t-1})$ of the training data $(x,y)$.<br/>
