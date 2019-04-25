# [Video Paragraph Captioning using Hierarchical Recurrent Neural Networks](http://arxiv.org/abs/1510.07712)/[video](https://www.youtube.com/watch?v=gX9rkJsfp2w)

They present a approach that exploits **hierarchical RNNs to tackle the video captioning problem**. Their hierarchical framework contains a **sentence generator** and a **paragraph generator**. 

The sentence generator produces one simple short sentence that describes a specific short video interval. 

It exploits both temporal and spatial-attention mechanisms to selectively focus on visual elements during generation.

The paragraph generator captures the inter-sentence dependency by taking as input the senential embedding produced by the sentence generator, combining it with the paragraph history, and outputting the new initial state for the sentence generator.
  
 **Sentence Generator**
 
First compute an attention score $$q_m^t$$ or each frame $$m$$, conditioning on the previous hidden state $$h^{t-1}$$: $$
q_m^t = w^T \phi (W_q v_m + U_q h^{t-1} + b_q)
$$ 
After this, they set up a sequential soft-max layer to get the attention weights: 
$$
\beta_m^t = e^{q_m^t} / \sum_{m=1}^{KM} e^{q_{m^{\prime}}^t}$$ 

Finally, a single feature vector is obtained by weighted averaging: 
$$
u^t = \sum_{m=1}^{KM}\beta_m^t v_m$$

The multi-modal layer maps the two features, together with the hidden state $$h^t$$ of the recurrent layer $$I$$, into a 1024 dimensional feature space and add them up:
$$
m^t = \phi(W_{m,o}u_o^t + W_{m,a}u_a^t+U_mh^t+b_m)$$
where $$\phi$$ is set to the element-wise tanh function. 

**Training and Generation**:
They treat the activation value indexed by a training word $$w_t^n$$ int he soft-max layer of their sentence generator as the likelihood of generating that word:

$$
P(w_t^n  | s_{1:n-1}, w_{1:t-1}^n, V)$$
They further define the cost of generating the whole paragraph $$s_{1:N}$$ ($$N$$ is the number of sentences in the paragraph) as : 
$$
PPL(s_{1:N}|V)=-\sum_{n=1}{N}\sum_{t=1}{T_n}log P(w_t^n | s_{1:n-1},w_{1:t-1}^n, V)/\sum_{n=1}^N T_n
$$

Finally, the  cost function over the entire training set is defined as : 
$$
PPL = \sum_{y=1}^Y (PPL(s_{1:N_y}^y | V_y). \sum_{n=1}^{N_y} T_n^y) / \sum_{y=1}^Y \sum_{n=1}^{N_y} T_n^y
$$
