# [Ask, Attend and Answer: Exploring question-guided spatial attention for visual question answering](http://arxiv.org/abs/1511.05234)

# Brief info

1. They propose a model they call the Spatial Memory Network and apply itto the VQA task.
2. Memory networks are RNN with an explicit attention mechanism that selects certain parts of the information stored in memory.
3. Their Spatial Memory Network stores neuron activations from different spatial regions of the image in its memory, and uses the question to choose relevant regions for computing the answer, aprocess of which constitutes a single "hop" in the network.
4. Propose anovel spatial attention architecture that aligns words with image patchesin the first hop, and obtain improved results by adding a second attention hop which considers the whole question to choose visual evidencebased on the results of the  first hop.
5. To better understand the inferenceprocess learned by the network, they design synthetic questions that specifically require spatial inference and visualize the attention weights.
6. They evaluate their model on two published visual question answering datasets, DAQUAR ["A multi-world approach to question answering aboutreal-world scenes based on uncertain input"]() and VQA ["VQA: visual question answering"](), and obtain improved results compared to astrong deep baseline model (iBOWIMG) which concatenates image andquestion features to predict the answer ["Simple  baseline  forvisual question answering"]().


# Related works

Before  the  popularity  of  VQA,  QA had already been established as a mature research problem in thearea  of  natural  language  processing.

Previous  QA  methods  include  :
1. searching for the key words of the question in a search engine ["Natural language questions for the web of data"]();
2. parsing the questionas a knowledge base (KB) query ["Semantic parsing via paraphrasing"];
3. embedding the question and using asimilarity measurement to  nd evidence for the answer ["Question answering with subgraph embeddings"].

Recently, memory networks  were  proposed  for  solving  the  QA  problem.
1. ["Memory  networks"]   1-st  introduces  the memory network as a general model that consists of a memory and four compo-nents: input feature map, generalization, output feature map and response. The model is investigated in the context of question answering, where the long-term memory acts as a dynamic knowledge base and the output is a textual response.
2. ["End-to-end memory networks"]() proposes a competitive memory network model that uses less supervision,called end-to-end memory network, which has a recurrent attention model over alarge external memory.
3. The Neural Turing Machine (NTM) ["Neural  turing  machine"]() couples a neuralnetwork to external memory and interacts with it by attentional processes to in-fer simple algorithms such as copying, sorting, and associative recall from inputand output examples.

The neural attention mechanism has been widely used in di erent areas ofcomputer  vision  and  natural  language  processing,  see  for  example  the  attention models in image captioning ["Show, attend and tell: Neural image caption generation with visual attention"]*, video description generation ["Describing videos by exploiting temporal structure"](), machine translation [Neural machine translation by jointly learningto align and translate.]()["Effective approaches to attention-basedneural machine translation"]() and machine reading systems [:eaching machines to read and comprehend]().

Most methods use the soft attention mechanism  first proposed in ["Neural machine translation by jointly learningto align and translate"](), which adds a layer to the network that predicts soft weights and uses them to compute a weighted combination ofthe items in memory.

The two main types of soft attention mechanisms differ in the function that aligns the input feature vector and the candidate feature vectors in order to compute the soft attention weights.
1. The first type uses an alignment function based on "concatenation" of the input and each candidate(we use the term "concatenation" as described [" ective approaches to attention-basedneural machine translation"]()),
2. The second type usesan alignment function based on the dot product of the input and each candidate. The "concatenation" alignment function adds one input vector (e.g. hiddenstate vector of the LSTM) to each candidate feature vector, embeds the result-ing vectors into scalar values, and then applies the softmax function to generatethe attention weight for each candidate.
3. [19][20][21][23] use the "concatenation" alignment  function  in  their  soft  attention  models  and  [24]  gives  a  literaturereview  of  such  models  applied  to  di erent  tasks.  On  the  other  hand,  the  dotproduct alignment function  rst projects both inputs to a common vector em-bedding space, then takes the dot product of the two input vectors, and appliesa softmax function to the resulting scalar value to produce the attention weightfor each candidate.
4. The end-to-end memory network [13] uses the dot product alignment function.
5. In [22], the authors compare these two alignment functionsin  an  attention  model  for  the  neural  machine  translation  task,  and   nd  thattheir implementation of the "concatenation" alignment function does not yieldgood performance on their task.


Motivated by this, in this paper they use the dot product alignment function in our Spatial Memory Network.VQA is related to image captioning. Several early papers about VQA directly adapt the image captioning models to solve the VQA problem ["Exploring models and data for image questionanswering"]()["Ask your neurons: A neural-based ap-proach  to  answering  questions  about  image"]() by generating the answer using a recurrent LSTM network conditioned on the CNN output. But these models' performance is still limited (["Exploring models and data for image questionanswering"]()["Ask your neurons: A neural-based ap-proach  to  answering  questions  about  image"]()).
1. ["isual7w: Grounded question an-swering in images"]() proposes anew dataset and uses a similar attention model to that in image captioning ["Show, attend and tell: Neural image caption generation with visual attention"](),but does not give results on the more common VQA benchmark.
2. ["Simple  baseline  forvisual question answering"]() summarizes several recent papers reporting results on the VQA dataset and gives a simple but strong baseline model (iBOWIMG) on this dataset. This simple baseline concatenates the image features with the bag of word embedding question representation and feeds them into a softmax classifier to predict the answer.
3. The iBOWIMG model beats most VQA models considered in the paper.
4. Here, we compare our proposed model to the VQA models(namely, the ACK model ["Ask  me  anything:  Free-form visual question answering based on knowledge from external sources"]() and the DPPnet model ["mage question answering using convolutional neu-ral network with dynamic parameter prediction"]()) which have comparable or better results than the iBOWIMG model.
5. The ACK model in ["Ask  me  anything:  Free-form visual question answering based on knowledge from external sources"]() is essentially the same as the LSTM model in ["Exploring models and data for image questionanswering"](), except that it uses image at-tribute features, the generated image caption and relevant external knowledgefrom a knowledge base as the input to the LSTM's  rst time step.
6. The DPPnetmodel in ["mage question answering using convolutional neu-ral network with dynamic parameter prediction"]() tackles VQA by learning a convolutional neural network (CNN)with some parameters predicted from a separate parameter prediction network.Their parameter prediction network uses a Gate Recurrent Unit (GRU) to gen-erate  a  question  representation,  and  based  on  this  question  input,  maps  thepredicted weights to CNN via hashing.
7. Neither of these models (["Image question answering using convolutional neu-ral network with dynamic parameter prediction"](),[Ask  me  anything:  Free-form visual question answering based on knowledge from external sources]()) containa spatial attention mechanism, and they both use external data in addition tothe  VQA  dataset,  e.g.  the  knowledge  base  in  ["Image question answering using convolutional neu-ral network with dynamic parameter prediction"]()  and  the  large-scale  textcorpus used to pre-train the GRU question representation ["mage question answering using convolutional neu-ral network with dynamic parameter prediction"]().
8. In this paper, weexplore a complementary approach of spatial attention to both improve perfor-mance and visualize the network's inference process, and obtain improved resultswithout using external data compared to the iBOWIMG model ["Simple  baseline  forvisual question answering"]() as well as the ACK model ["Ask  me  anything:  Free-form visual question answering based on knowledge from external sources"]() and the DPPnet model ["mage question answering using convolutional neu-ral network with dynamic parameter prediction"]() which use external data.



1. In one of the early works ["A multi-world approach to question answering aboutreal-world scenes based on uncertain input"](), VQA is seen as a Turing test proxy. The authors propose an approach based on handcrafted features using a semantic parse of the question and scene analysis of the image combined in a latent-world Bayesian framework.
2. More recently, several end-to-end deep neural networks that learnfeatures directly from data have been applied to this problem (["sk your neurons: A neural-based ap-proach  to  answering  questions  about  image"](),["Exploring models and data for image questionanswering"]()).
3. Most ofthese are directly adapted from captioning models(["Long-term  recurrent  convolutional  networks  for  visualrecognition and description"](),["Show  and  tell:  A  neural  imagecaption generator"](),["eep fragment embeddings for bidirectionalimage sentence mapping"]()), and utilize a recurrent LSTM network, which takes the question and Convolutional Neural Net (CNN) image  features  as  input,  and  outputs  the  answer.
4. Though  the  deep  learning methods in (["sk your neurons: A neural-based ap-proach  to  answering  questions  about  image"](),["xploring models and data for image questionanswering"]()) have shown great improvement compared to the hand crafted feature method ["A multi-world approach to question answering aboutreal-world scenes based on uncertain input"](), they have their own drawbacks. These models based on theLSTM reading in both the question and the image features do not show a clear improvement compared to an LSTM reading in the question only (["sk your neurons: A neural-based ap-proach  to  answering  questions  about  image"](),["Exploring models and data for image questionanswering"]()).
5. Furthermore, the rather complicated LSTM models obtain similar or worse accuracy to a baseline model which concatenates CNN features and a bag-of-words ques-tion embedding to predict the answer, see the IMG+BOW model in ["Exploring models and data for image questionanswering"]() and the iBOWIMG model in ["Simple  baseline  forvisual question answering"]().

# contributions

## drawback of existing models

1. A major drawback of existing models is that they do not have any explicit notion  of  object  position,  and  do  not  support  the  computation  of  intermediate  results  based  on  spatial  attention.
2. Their  intuition  is  that  answering  visualquestions often involves looking at different spatial regions and comparing theircontents and/or locations.

<p align="center"><img src="https://dl.dropboxusercontent.com/s/x1olgzae6ty0cvx/Screenshot%20from%202016-05-26%2014%3A41%3A01.png?dl=0" width="500" ></p>

<font size="2">For example, to answer the questions in Fig. 1, we need to look at a portion of the image, such as the child or the phone booth. Similarly, to answer the question "Is there a cat in the basket?" in Fig. 2, wecan  rst  nd the basket and the cat objects, and then compare their locations.</font>

They propose a new deep learning approach to VQA that incorporates explicit spatial  attention,  which  we  call  the  Spatial  Memory  Network  VQA  (SMem-VQA).

Their approach is based on memory networks, which have recently been proposed for text Question Answering (QA). Memory networks combine learned text embeddings with an attention mechanism and multi-step inference.The text QA memory network stores textual knowledge in its "memory" in the form of sentences, and selects relevant sentences to infer the answer. However,in VQA, the knowledge is in the form of an image, thus the memory and thequestion come from di erent modalities.

We adapt the end-to-end memory network to solve visual question answering by storing the convolutional networkoutputs obtained from di erent receptive  elds into the memory, which explicitlyallows spatial attention over the image. We also propose to repeat the processof gathering evidence from attended regions, enabling the model to update theanswer based on several attention steps, or "hops". The entire model is trained end-to-end and the evidence for the computed answer can be visualized usingthe attention weights.

To summarize their contributions:
1. In this paper they propose  a  novel  multi-hop  memory  network  with  spatial  attention  for  the VQA task which allows one to visualize the spatial inference process used by the deep network.
2. they design an attention architecture in the first hop which uses each word em-bedding to capture  fine-grained alignment between the image and question,
3. they create a series of synthetic questions that explicitly require spatial inferenceto  analyze  the  working  principles  of  the  network,  and  show  that  it  learns logical inference rules by visualizing the attention weights,
4. they provide an extensive evaluation of several existing models and their own model on the same publicly available datasets.

# Spatial Memory Network for VQA

<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/1cf6bc0866226c1f8e282463adc8b75d92fba9bb/2-Figure2-1.png" width="700" ></p>

<font size="2">Figure 2. their proposed spatial-attention memory network for visual question answering.</span>

$a$. Overview of <span style="color:red">one-hop network</span>: The <span style="color:red">question word vectors $v_j$</span> and the <span style="color:red">CNN activation vectors $\mathbf{S}=s_i$</span> at each <span style="color:red">image location $i$</span> are embedded into a common semantic space of the question word vectors $v_j$ using the "attention" visual embedding $\mathbf{W_A}$, and then used to infer spatial attention weights $\mathbf{W}_{att}$ via $(b)$ or $(c)$. These weights are then used to compute a weighted sum over the visual features embedded using a separate transformation $\mathbf{W_E}$ . The resulting ”evidence” vector $\mathbf{S}_{att}$ is combined with the question embedding $Q$ and the answer is predicted. An additional hop can repeat the process to gather additional evidence.

$b$. The sentence-guided attention model, which computes a weighted sum over the word vectors and uses its correlation with the spatial features to produce attention weights.

$c$. The finer-grained word-guided attention model, which selects the word with the maximum correlation at each spatial location. See text for more details.

The  input  to  network  is  a  question  comprised  of  a  variable-length  se-quence of words, and an image of  fixed size. Each word in the question is  first represented  as  a  one-hot  vector  in  the  size  of  the  vocabulary,  with  a  valueof  one  only  in  the  corresponding  word  position  and  zeros  in  the  other  posi-tions.  Each  one-hot  vector  is  then  embedded  into  a  real-valued  word  vector, $V=\{v_j|v_j\in \mathbb{R}^N ;j= 1,..., T\}$, where $T$ is the maximum number of wordsin the question and $N$ is the dimensionality of the embedding space. Sentences with length less than $T$ are padded with special1 value, which are embedded to all-zero word vector.The words in questions are used to compute attention over the visual memory, which contains extracted image features. The input image is processed by a CNN to extract high-levelM-dimensional visual features on a grid of spatial locations. Specifically, we use $S=\{s_i|s_i \in \mathbb{R}^M;i= 1,..., L\}$ to represent the spatial CNN features at each of theLgridlocations. In this paper, the spatial feature outputs of the last convolutional layer of GoogLeNet (inception_5b=output) ["Going deeper with convolutions"]() are used as the visual features for theimage.

The convolutional image feature vectors at each location are embedded into a common semantic space with the word vectors. Two diffient embeddings areused: the "attention" embedding $W_A$ and the "evidence" embedding $W_E$. The attention embedding projects each visual feature vector such that its combination with the embedded question words generates the attention weight at that location. The evidence embedding detects the presence of semantic concepts orobjects,  and  the  embedding  results  are  multiplied  with  attention  weights  andsummed over all locations to generate the visual evidence vector $S_{att}$.

Finally, the visual evidence vector is combined with the question represen-tation and used to predict the answer for the given image and question. In thenext section, we describe the one-hop Spatial Memory network model and thespeci c attention mechanism it uses in more detail.

## Word Guided Spatial Attention in One-Hop Model

Rather than using the bag-of-words question representation to guide attention,
the attention architecture in the first hop (Fig.~\ref{fig:illus}(b)) uses each word vector separately to extract correlated visual features in memory.
The intuition is that the BOW representation may be too coarse, and letting each word select a related region may provide more fine-grained attention.
The correlation matrix $C \in \mathbb{R}^{T\times L}$ between word vectors $V$ and visual features $S$ is computed as

$$
C = V \cdot (S \cdot W_A + b_A)^T
$$

where $W_A \in \mathbb{R}^{M\times N}$ contains the attention embedding weights of visual features $S$, and $b_A \in \mathbb{R}^{L\times N}$ is the bias term.
This correlation matrix is the dot product result of each word embedding and each spatial location's visual feature, thus each value in correlation matrix $C$ measures the similarity between each word and each location's visual feature.

The spatial attention weights $W_{att}$ are calculated by taking maximum over the word dimension $T$ for the correlation matrix $C$, selecting the highest correlation value for each spatial location, and then applying the softmax function

$$
W_{att} = \text{softmax}(\max_{i=1,\cdots,T}(C_i)), ~C_i \in \mathbb{R}^L
$$

The resulting attention weights $W_{att} \in \mathbb{R}^{L}$ are high for selected locations and low for other locations, with the sum of weights equal to $1$. For instance, in the example shown in Fig.~\ref{fig:illus}, the question ``Is there a cat in the basket?'' produces high attention weights for the location of the basket because of the high correlation of the word vector for \textit{basket} with the visual features at that location.

The evidence embedding $W_E$ projects visual features $S$ to produce high activations for certain semantic concepts. E.g., in Fig.~\ref{fig:illus}, it has high activations in the region containing the cat. The results of this evidence embedding are then multiplied by the generated attention weights $W_{att}$, and summed to produce the selected visual ``evidence'' vector $S_{att} \in \mathbb{R}^N$,

$$
S_{att} = W_{att} \cdot (S \cdot W_E + b_E)
$$

where $W_E \in \mathbb{R}^{M\times N}$ are the evidence embedding weights of the visual features $S$, and $b_E \in \mathbb{R}^{L\times N}$ is the bias term.
In our running example, this step accumulates \textit{cat} presence features at the \textit{basket} location.

Finally, the sum of this evidence vector $S_{att}$ and the question embedding $Q$ is used to predict the answer for the given image and question.
For the question representation $Q$, we choose the bag-of-words (BOW). Other question representations, such as an LSTM, can also be used, however, BOW has fewer parameters yet has shown good performance. As noted in~\cite{shih2015look}, the simple BOW model performs roughly as well if not better than the sequence-based LSTM for the VQA task. Specifically, we compute

$$
Q = W_Q \cdot V + b_Q
$$

where $W_Q \in \mathbb{R}^T$ represents the BOW weights for word vectors $V$, and $b_Q \in \mathbb{R}^{N}$ is the bias term. The final prediction $P$ is

$$
P = \text{softmax}(W_P \cdot f(S_{att} + Q) + b_P)
$$

where $W_P \in \mathbb{R}^{K\times N}$, bias term $b_P \in \mathbb{R}^{K}$, and $K$ represents the number of possible prediction answers. $f$ is the activation function, and we use ReLU here.
In our running example, this step adds the evidence gathered for \textit{cat} near the basket location to the question, and, since the cat was not found, predicts the answer ``no''.
The attention and evidence computation steps can be optionally repeated in another hop, before predicting the final answer, as detailed in the next section.


## Spatial Attention in Two-Hop Model

We can repeat hops to promote deeper inference, gathering additional evidence at each hop. Recall that the visual evidence vector $S_{att}$ is added to the question representation $Q$ in the first hop to produce an updated question vector,

$${O_{hop1} = S_{att} + Q}$$

On the next hop, this vector $O_{hop1} \in \mathbb{R}^{N}$ is used in place of the individual word vectors $V$ to extract additional correlated visual features to the whole question from memory and update the visual evidence.

The correlation matrix $C$ in the first hop provides fine-grained local evidence from each word vectors $V$ in the question, while the correlation vector $C_{hop2}$ in next hop considers the global evidence from the whole question representation $Q$.
The correlation vector $C_{hop2} \in \mathbb{R}^L$ in the second hop is calculated by

$$
C_{hop2} = (S \cdot W_E + b_E) \cdot O_{hop1}
$$

where $W_E \in \mathbb{R}^{M\times N}$ should be the attention embedding weights of visual features $S$ in the second hop and $b_E \in \mathbb{R}^{L\times N}$ should be the bias term. Since the attention embedding weights in the second hop are shared with the evidence embedding in the first hop, so we directly use $W_E$ and $b_E$ from first hop here.

The attention weights in the second hop $W_{att2}$ are obtained by applying the softmax function to the correlation vector $C_{hop2}$.

$$
W_{att2} = \softmax(C_{hop2})
$$


Then, the correlated visual information in the second hop $S_{att2} \in \mathbb{R}^N$ is extracted using attention weights $W_{att2}$.

$$
S_{att2} = W_{att2} \cdot (S \cdot W_{E_2} + b_{E_2})
$$

where $W_{E_2} \in \mathbb{R}^{M\times N}$ are the evidence embedding weights of visual features $S$ in the second hop, and $b_{E_2} \in \mathbb{R}^{L\times N}$ is the bias term.

The final answer $P$ is predicted by combining the whole question representation $Q$, the local visual evidence $S_{att}$ from each word vector in the first hop and the global visual evidence $S_{att2}$ from the whole question in the second hop,

$$
P = \softmax(W_P \cdot f(O_{hop1} + S_{att2}) + b_P)
$$

where $W_P \in \mathbb{R}^{K\times N}$, bias term $b_P \in \mathbb{R}^{K}$, and $K$ represents the number of possible prediction answers. $f$ is activation function. More hops can be added in this manner.

The entire network is differentiable and is trained using stochastic gradient descent via standard backpropagation, allowing image feature extraction, image embedding, word embedding and answer prediction to be  jointly optimized on the training image/question/answer triples.



<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/1cf6bc0866226c1f8e282463adc8b75d92fba9bb/7-Table1-1.png" width="500" ></p>
Table 1. Results on the VQA and the DAQUAR datasets (in percentage). The column marked ∗ are results reported in other papers.



<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/1cf6bc0866226c1f8e282463adc8b75d92fba9bb/4-Figure3-1.png" width="500" ></p>
Figure 3. Object presence experiment: for each image and question pair, we show the original image (left), the evidence embeddingWE of the convolutional layer (middle), and the attention weights Watt of the one-hop model (right). The evidence embedding WE almost always has a high response on the whole cat or dog object, and low response elsewhere. We see that the model learns the following two inference rules to answer the question: if the question word does not match the animal (i.e. the answer is ”no”), it uses attention to gather visual features from the region including the face of the cat or dog (bottom row); if the question word matches the animal (i.e. the answer is ”yes”), it gathers features from regions other than the face (top row). It then uses the gathered features to predict the answer: if the features are animal face features, predict ”no”, otherwise, predict ”yes”.

<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/1cf6bc0866226c1f8e282463adc8b75d92fba9bb/5-Figure4-1.png" width="500" ></p>

Figure 4. Absolute position experiment: for each image and question pair, we show the original image (left) and the attention weights Watt of the one-hop model (right). The attention weights follow the following two rules. The first rule (top row) looks at the position specified in the question (top|bottom|right|left), if it contains a square, answer ”yes”; otherwise answer ”no”. The second rule (bottom row) looks at the region where there is a square, and answers ”yes” if the question contains that position and ”no” if it contains one of the other three positions.

<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/1cf6bc0866226c1f8e282463adc8b75d92fba9bb/5-Figure5-1.png" width="500" ></p>
Figure 5. Relative position experiment: for each image and question pair, we show the original image (left), the evidence embedding WE of the convolutional layer (middle) and the attention weightsWatt (right). The evidence embeddingWE has high activations on both the cat and red square. The attention weights follow the inference rules similar to those in Sec. 4.1.2, with the difference that the attention position is around the cat.

<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/1cf6bc0866226c1f8e282463adc8b75d92fba9bb/7-Figure6-1.png" width="500" ></p>
Figure 6. Visualization of the spatial attention weights in the one-hop and two-hop model on VQA (top two rows) and DAQUAR (bottom row) datasets. For each image and question pair, we show the original image, the attention weights Watt of the one-hop model, and the two attention weightsWatt andWatt2 of the two-hop model in order.


<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/1cf6bc0866226c1f8e282463adc8b75d92fba9bb/10-Figure7-1.png
Figure 7. Visualization of the attention wei" width="500" ></p>ghts in the one-hop and two-hop models on the DAQUAR dataset. For each image and question pair, we show the original image, the attention weightsWatt of the one-hop model, and the two sets of attention weights (Watt andWatt2) of the two-hop model, in that order. For some examples only the two-hop model predicts the correct answer (Rows a, b and c), while for other examples both models predict the correct answer (Row d).

<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/1cf6bc0866226c1f8e282463adc8b75d92fba9bb/11-Figure8-1.png" width="500" ></p>
Figure 8. Visualization of the attention weights in the one-hop and two-hop models on the VQA dataset. For each image and question pair, we show the original image, the attention weightsWatt of the one-hop model, and the two sets of attention weights (Watt andWatt2) of the two-hop model, in that order. We show several examples of different types of questions.
