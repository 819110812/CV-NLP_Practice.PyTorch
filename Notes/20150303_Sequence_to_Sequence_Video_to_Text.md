## [Sequence to Sequence – Video to Text](https://www.cs.utexas.edu/~vsub/)/[Code](http://vsubhashini.github.io/s2vt.html#code)

This paper propose a End-to-end sequence-to-sequence model to generate captions for videos.<br/>
Two level LSTMs that learn a representation of a sequence of frames in order to decode it into a sentence that describes the event in the video. **The top LSTM** layer models visual feature inputs. **The second LSTM** layer models language given the text input and the hidden representation of the video sequence. $\< BOS\>$(Begin-of-sentence) and $\<EOS\>$(end-of-sentence), and Zeros are used as a $\<pad\>$ when there is no input at the time step.<br/>

**Training**: Using SGD to optimize the log-likelihood function 

$$
\theta^* = \arg\max_{\theta} \sum_{t=1}^m \log p(y_t| h_{n+t-1},y_{t-1};\theta)
$$


Video and text representation : **RGB frames** + **Optical Flow**

The score of each new word 

$$
p(y_t=y') = \alpha . p_{rgb}(y_t=t')+(1-\alpha).p_{flow}(y_t = y')
$$

where the conditional probability of an output sequence $$(y_1,...,y_m|x_1,...,x_n) = \prod_{t=1}^m p(y_t|h_{n+t-1},y_{n-1})$$, $$p(y_t|h_{n+t})$$ is given by softmax over all the words in the vacabulary. 

$$
p(y|z_t)=\frac{e^{W_y z_t}}{\sum_{y\\in V}e^{W_{y^,}z_t}}
$$

