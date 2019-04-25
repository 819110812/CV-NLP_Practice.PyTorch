## [Compositional Memory for Visual Question Answering](http://arxiv.org/abs/1511.05676)


Visual Question Answering (VQA) emerges as one of the most fascinating topics in computer vision recently. Many state of the art methods naively use holistic visual features with language features into a Long Short-Term Memory (LSTM) module, neglecting the sophisticated interaction between them. This coarse modeling also blocks the possibilities of exploring finer-grained local features that contribute to the question answering dynamically over time.

This paper addresses this fundamental problem by directly modeling the temporal dynamics between language and all possible local image patches. When traversing the question words sequentially, our end-to-end approach explicitly fuses the features associated to the words and the ones available at multiple local patches in an attention mechanism, and further combines the fused information to generate dynamic messages, which we call episode. We then feed the episodes to a standard question answering module together with the contextual visual information and linguistic information. Motivated by recent practices in deep learning, we use auxiliary loss functions during training to improve the performance. Our experiments on two latest public datasets suggest that our method has a superior performance. Notably, on the DAQUAR dataset we advanced the state of the art by 6$\%$, and we also evaluated our approach on the most recent MSCOCO-VQA dataset.

# Introduction

Given an image and a question, the goal of Visual Question Answering (VQA) is to directly infer the answer(s) automatically from the image. This is undoubtedly one of the most interesting, and arguably one of the most challenging, topics in computer vision in recent times.

Almost all state-of-the-art methods use predominantly holistic visual features in their systems. For example, Malinowski et.al.  (["Ask your neurons: A neural-based approach to answering questions about images"]()) used the concatenation of linguistic feature and visual feature extracted by a Convolutional Neural Network (CNN), and Ren et.al.  (["Exploring models and data for image question answering"]()) considered the visual feature as the first word to initialize the sequential learning.

While the use of holistic approach is straightforward and convenient, it is, however, debatably problematic. For example, in the VQA problems many answers are directly related to the contents of some image regions. Therefore, it is dubious if the holistic features are rich enough to provide the information only available at regions. Also, it may hinder the exploration of finer-grained local features for VQA.

In this paper we propose a Compositional Memory for an end-to-end training framework. Our approach takes the advantage of the recent progresses in image captioning (["Deep visual-semantic alignments for generating image descriptions"]()), natural language processing (["Ask me anything: Dynamic memory networks for natural language processing"]()), and computer vision to advance the study of the VQA. Our goal is to fuse local visual features and the linguistic information over time, in a Long Short-Term Memory (LSTM) based framework. The fused information, which we call "episodes", characterizes the interaction and dynamics between vision and language.

Explicitly addressing the interaction between question words and local visual features has a number of advantages. To begin with, regions provide rich info towards capturing the dynamics in question. Intuitively, parts of an image serve as "candidates" that may have varying importance at different time when parsing a question sentence. Recent study of image captioning (["Show, attend and tell: Neural image caption generation with  visual attention"]()), a closely related research topic, suggests that visual attention mechanism is very important in generating good descriptions. Obviously, this idea will also improve the accuracy in question answering.

Going deeper, this candidacy is closely related to the concept of semantic "facts" in reasoning. For example, one often begins to dynamically search useful local visual evidences at the same time when (s)he reads words. The use of facts has been explored in the natural language processing recently (["Ask me anything: Dynamic memory networks for natural language processing"]()), but this useful concept cannot be explored without local visual information in computer vision.

While the definition of "visual facts" is still elusive, we can approach the problem through modeling interactions between vision and language. This "sensory interaction" plays a phenomenal role in the information processing and reasoning. It has a significant meaning in memory study as well. Eichenbaum and Cohen argued that part of the human memory needs to be modeled as a form of relationship between spatial, sensory and temporal information (["Memory, Amnesia, and the Hippocampal System"]()).

%The above reasons makes it compelling to use regional information as the building blocks for VQA, however, they also bring us two major problems: 1) where to extract, and 2) what to extract?

%The first problem (where) is a chicken and egg problem: we know the answer immediately if we know where to look at.
%To bypass this problem, we propose to use a fixed number of regions proposals. As a result, instead of dynamically finding important regions at each time, we opt to modify the information based on the dynamics in the previous processing.
%This remedy naturally allows us to solve the second question (what).
%Our solution is then rooted on the study of the dynamics information extracted from local image patches, which we call "Visual Facts" in this paper.

Specifically, our method traverses the words in a question sequentially, and explicitly fuses the linguistic and the ones available at local patches to episodes. An attention mechanism  is used to re-weight the importance of the regions (["Show, attend and tell: Neural image caption generation with visual attentio"]()).
The fused information is fed to a dynamic network to generate episodes.
We then feed the episodes to a standard question answering module together with the contextual visual information and linguistic information.

The use of local features inevitably leads to the quest about region selection.
In principle, the regions can be 1) patches generated by object proposals, such as those obtained by edgebox (["Edge boxes: Locating object proposals from edges"]()) and faster-RCNN (["Faster {R-CNN}: Towards real-time object detection with region proposal network"]()), and 2) overlapping patches that cover most important contents in image. In this paper, we choose the latter and use the features of the last convolutional layer in the CNNs.

Our experiments on two latest public datasets suggest that our method outperforms the other state of the art methods. We tested on the  DAQUAR (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) and MSCOCO-VQA (["{VQA:} visual question answering"]()). Notably, on the DAQUAR dataset we advanced the state of the art by 6$\%$.
We also compared a few variants of our method and demonstrated the usefulness of the Compositional Memory for VQA.
We further verified our idea on the latest MSCOCO-VQA dataset.

The main contributions of the paper are:
1. We present an end-to-end approach that explores the local fine grained visual information for VQA tasks,
2. We develop a Compositional Memory that explicitly models the interactions of vision and language,
3. Our method has a superior performance and it outperforms the state of the art methods.

# Related Work

#### CNN, RNN, and LSTM
Recently, deep learning has achieved great success on many computer vision tasks. For example, CNN has set records on standard object recognition benchmarks (["Imagenet classification with deep  convolutional neural networks"]()). With a deep structure, CNN can effectively learn complicated mappings from raw images to the target, which requires less domain knowledge compared to handcrafted features and shallow learning frameworks.

Recurrent Neural Networks (RNN) have been used for modeling temporal sequences and gained attention in speech recognition (["Speech recognition with deep recurrent neural networks"]()), machine translation (["Neural machine translation by jointly learning to align and translate"]()), image captioning (["Deep visual-semantic alignments for generating image descriptions"]()). The recurrent connections are feedback loops in the unfolded network, and because of these connections, RNNs are suitable for  modeling time series with strong nonlinear dynamics and long time correlations. The traditional RNN is hard to train due to the vanishing gradient problem, $\textbf{i.e.} the weight updates computed via error backpropagation through time may become very small.

Long Short Term Memory model (["Long short-term memory"]()) has been proposed as a solution to overcome these problems. The LSTM architecture uses memory cells with gated access to store and output information, which alleviates the vanishing gradient problem in backpropagation over multiple time steps. Specifically, in addition to the hidden state, the LSTM also includes an input gate, a forget gate, an output gate, and the memory cell.
In this architecture, input gate and forget gate are sigmoidal gating functions, and these two terms learn to control the portions of the current input and the previous memory that the LSTM takes into consideration for overwriting the previous state. Meanwhile, the output gate controls how much of the memory should be transferred to the hidden state. These mechanisms allow LSTM networks to learn temporal dynamics.

%In this architecture, $i_t$ and $f_t$ are sigmoidal gating functions, and  these two terms learn to control the portions of the current input and the previous memory that the LSTM takes into consideration for overwriting the previous state. Meanwhile, the output gate $o_t$ controls how much of the memory should be transferred to the hidden state. These mechanisms allow LSTM networks to learn temporal dynamics with long time constants.

#### Language and vision

The effort of combining language and vision attracts a lot of attention recently.
Image captioning and VQA are two most intriguing problems.

Question answering (QA) is a classical problem in natural language processing (["Ask me anything: Dynamic memory networks for natural language processing"]()). When images are involved, the goal of VQA is to infer the answer of a question directly from the image (["{VQA:} visual question answering"]()). Multiple questions and answers can be associated to the same image during training.

It has been shown that VQA can borrow the idea from image captioning.
Being a related area, image captioning also uses RNN for sentence generation (["Long-term recurrent convolutional networks for  visual recognition and description"]()).
Attention mechanism is recently adopted in image captioning and proves to be a useful component (["Show, attend and tell: Neural image caption generation with visual attention"]()).

#### LSTM for VQA

Because a VQA system needs to process language and visual information simultaneously, most recent work adopted the LSTM in their approaches.
A typical LSTM-VQA uses holistic image features extracted by CNNs as "visual words".

The visual word features are used either as the first or at the end of question sequence (["Exploring models and data for image question answering"]()) or they are concatenated together with question word vectors into a LSTM (["Are you talking to a machine? dataset and methods for multilingual image question answering"]()) .

<p align="center"><img src="https://dl.dropboxusercontent.com/s/raiijwp1pnd9ezk/basiclstm-vqa.jpg?dl=0" width="500" ></p>

This LSTM-VQA framework is straightforward and useful.
However, treating the feature interaction as feature vector concatenation lacks the capability that explicitly extracts finer-grained information. As we discussed before, details of facts may be neglected if global visual features are used. This leads to the quest for more effective information fusion model for language and image in VQA problems.


# Our Approach

We present our approach in this section. First, we present our model overview. Then, we discuss the technical details and explain the training process.

## Our End-to-End VQA model

### Overview and notation

Compared to the basic LSTM approach for VQA , We made two major improvements of the model.
The diagram of the network is shown in Figure below.

<p align="center"><img src="https://dl.dropboxusercontent.com/s/lkfgjmiyn2nuep7/dynamicmemorynetwork.jpg?dl=0" width="500" ></p>

The first, and the most important addition, is a Compositional Memory.
It reasons over those input features to produce an image-level representation for answer module.
It reflects an experience over image contents.

The second addition is the LSTM module for parsing (factorizing) the question.
It provides input for both the Compositional Memory and the question answering LSTM.
In the experiment we will show the importance of this module for the VQA tasks.

In part, this implementation is aligned with the findings in cognitive neuroscience.
It is well known that semantic (e.g. , visual features and classifiers) and episodic memory (e.g. , temporal questioning sentence) together make up the declarative memory of human beings, and the interactions among them become the key in representation and reasoning (["Episodic and semantic memory"]()).
Our model captures this interaction naturally.

### Compositional Memory

<p align="center"><img src="https://dl.dropboxusercontent.com/s/df4qi0jsdbi2653/compositelstmunit.jpg?dl=0" width="500" ></p>

We present the details of our Compositional Memory in this section.
This unit consists of a number of Region-LSTM and $\alpha$ gates.
All these Region-LSTM share the same set of parameters, and their results are combined to generate an episode at each time step.

The region-LSTMs are mainly in charge of processing input image region contents in parallel. It dynamically generates language-embedded visual information for each region, conditioned on previous episodes, local visual feature, and current question word.
The state, gates and cells of of each Region-LSTM are updated as follows:
$$
\begin{cases}
 & i_t ^k = \sigma(W_{qi}q_t + W_{hi}h_{t-1} + W_{xi} X_k + b_i)  \\
 & f_t ^k= \sigma(W_{qf}q_t + W_{hf}h_{t-1} + W_{xf} X_k + b_f) \\
 & o_t ^k= \sigma(W_{qo}q_t + W_{ho}h_{t-1} + W_{xo} X_k + b_o) \\
 & g_t ^k= \tanh(W_{qg}q_t + W_{hg}h_{t-1} + W_{xg} X_k + b_g) \\
 & c_t ^k= f_t ^k \odot c_{t-1} ^k + i_t ^k \odot g_t ^k\\
 & m_t ^k= o_t ^k \odot \tanh(c_t ^k)
\end{cases}
$$
where the hidden state $h_t$ denotes an episode at every time step, $q_t$ is the language information (e.g.  features generated by $\textbf{word2vec}$), and $X_k$ is the $k^{th}$ regional CNN features.
$m_t$ is the output of region-LSTM $t$.
Please note that the superscripts are omitted in the above notations for simplicity.

In Region-LSTM , $c_t$ denotes the memory cell, $g_t$ denotes an input modulation gate.
$i_t$ and $f_t$ are input and forget gates, which control the portions of the current input and the previous memory that LSTM takes into consideration.
$o_t$ is output gate that determines how much of the memory to transfer to the hidden state.
$\odot$ is the element-wise multiplication operation, and $\sigma()$ and $tanh()$ denote the sigmoidal and tanh operations, respectively.
These mechanisms allow LSTM to learn long-term temporal dynamics.

The implementation of Region-LSTM is similar to that of traditional LSTM. However, the important differences lie in the parallel strategy that all parameters ($W_{q*}, W_{h*}, W_{x*}, b_{*}$) are shared across different regions. Please note that each Region-LSTM has its own gate and cell state, respectively.

The $\alpha$ gate is also conditioned on previous episode $h_{t-1}$, region feature $X_k$, and current input language feature $q_t$. It returns a single scalar for each region.
This gate is mainly used for weighted combination of region messages for generating episodes, which are dynamically pooled into image-level information.
Similar to $W_{zq}, W_{zh}, W_{zx}, b_z, W_\alpha, b_\alpha$, the parameters of $\alpha$ gate, are also shared across regions. At every time step $t$, $\alpha$ gate dynamically generates values $\alpha _k ^t$ for $k^{th}$ region.
$$
\begin{array}{l}
{z_k ^t} = \tanh \left( {{W_{zq}}{q_t} + {W_{zh}}{h_{t - 1}} + {W_{zx}}{x_k} + {b_z}} \right)\\
{\alpha _k ^t} = \sigma \left( {{W_\alpha }{z_k ^t} + {b_\alpha }} \right)
\end{array}
$$

In order to summarize region-level information to an image-level feature, we employ a modified pooling mechanism of gated recurrent style (["Empirical evaluation of gated recurrent neural networks on sequence modeling"]()). The episode $h_t$ acts as dynamic feature as well as hidden state of Compositional Memory unit. It is updated to renew the input information of Region-LSTMs together with language input at every time step.

$$
\begin{array}{l}
 \beta  = 1 - \frac{1}{K}\sum\limits_k {{\alpha _k}}\\
 {h_t} =  \beta {h_{t - 1}} + \frac{1}{K} \sum\limits_k {{\alpha _k ^t}m_t^k}
\end{array}
$$
where $K$ is the total number of regions.

One of the advantages of Compositional Memory is that it goes beyond traditional static image feature. The unit incorporates both the merits of LSTM and attention mechanism, and thus it is suitable to generate dynamic episodic messages for visual question answering applications.

### Language LSTM

We use a Language LSTM to process linguistic inputs.
We consider this representation as "factorization of questions", which captures the recurrent relations of question word sequence, and stores semantic memory information about the questions. This strategy for processing language has also been proven to be  important in image captioning (["Long-term recurrent convolutional networks for visual recognition and description"]()).

### Other components

#### Answer generation

The answer generation module takes dynamic episodes, together with language and static visual context generated by CNNs to generate the answers in a LSTM framework.

#### CNN

Our approach is compatible with all the major CNN network, such as AlexNet (["Imagenet classification with deep convolutional neural networks"]()) and GoogLeNet (["Going deeper with convolutions"]()).
In each CNN, we use the last convolution layer as region selections.
For example, GoogleNet, we use the $1024 \times 7 \times 7$ feature map from the inception\_5b/output. This means that Compositional Memory operates on 49 regions, each of which is represented by 1024-dim feature vector.
Following the current practice, we used the output of the last CNN layer as visual context.

## Training

Generally, there are three types of the VQA tasks: 1) Single Word, 2) Multiple Words (or called "Open-ended"), and 3) Multiple Choice. The single word category restricts the answer to have only one single word. Multiple Words VQA is an open-ended task, where the length of answer is not limited and the VQA system needs to sequentially generate possible answer words until generating an answer-ending mark. The multiple choice task refers to select most probable answer from a set of answer candidates.

Among these three, the "Open-ended" VQA task is the most difficult one, thus we chose this category to demonstrate the performance of our approach.
In this section, we briefly present the training procedure, the loss function, and other implementation details.

### Protocol
During the training of our VQA system, CNN and LSTMs are jointly learned in an end-to-end way. The unfolded network of our proposed model is shown in Figure below.

<p align="center"><img src="https://dl.dropboxusercontent.com/s/s98uq56g0swui0g/unfoldednetwork.jpg?dl=0" width="700" ></p>


In this Open-ended category of VQA, questions may have multiple word answers.
We consequently decompose the problem to predict a set of answer words $A = \left\{ {{a_1},{a_2},...,{a_M}} \right\}$, where $a_i$ are words from a finite vocabulary $\Omega '$ and M is the number of answer words for a given question and image. To deal with open-ended VQA task, we add an extra token $\langle EOA \rangle$ into the vocabulary $\Omega  = \Omega '\cup \{\langle EOA \rangle\}$. The $\langle EOA \rangle$ indicates the end of the answer sequence. Therefore, we formulate the prediction procedure recursively as:
$$
{\widehat a_t} = \arg \max p(a|X,q,{\widehat A_{t - 1}};\vartheta )
$$
where ${\widehat A_{t - 1}}$ is the set of previously predicted answer words, with ${\widehat A_{0}} = \{\}$ at start. The prediction procedure is terminated when ${\widehat a_t} = \langle EOA \rangle$.

Feed the VQA system with a question as a sequence of words, $\textbf{i.e.} $q = [{q_1},{q_2},...,{q_{n - 1}},\left[\kern-0.15em\left[ ? \right]\kern-0.15em\right]]$, where $\left[\kern-0.15em\left[ ? \right]\kern-0.15em\right]$ encodes the end of question. In the training phase, we augment the question word sequence with the corresponding ground truth answer sequence $a$, $\textbf{i.e.}$ $\widehat q: = [q,a]$. During the test phase, at $t^{th}$ time step we augment question $q$, with previously predicted answer words ${\widehat q_t}: = [q,{\widehat a_{1,...,t - 1}}]$.

### Loss function

All the parameters are jointly learned with cross-entropy loss. The output predictions that occur before the ending question mark $\left[\kern-0.15em\left[ ? \right]\kern-0.15em\right]$ are excluded from the loss computation, so that the model is solely penalized based on the predicted answer words.

Motivated by the recent success of GoogLeNet, We adopt a multi-task training strategy for learning the parameters of our network.
Specifically, in additional to the question answering LSTM, we add a "Language Only" loss layer on the Language LSTM, and an "Episode Only" loss layer on the Compositional Memory.
These two auxiliary loss functions are added during training to improve the performance, and they are removed during testing.

### Implementation

We implemented our end-to-end VQA network using Caffe\footnote{\url{http://caffe.berkeleyvision.org/}}. The CNN models are pre-trained, and then fine-tuned in our recurrent network training. The source code of our implementation will be available in public.

# Experiments
We test our approach on two large data sets, namely, DAQUAR (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) and MSCOCO-VQA.
In the experiments on these two data sets, our method outperforms the state of the arts in different well recognized metrics.

## Datasets

$\textbf{DAQUAR}$ contains 12,468 human question answer pairs on 1,449 images of indoor scene. The training set contains 795 images and 6,793 question answer pairs, and the testing set contains 654 images and 5,675 question answer pairs.

We run experiments for the full dataset with all classes, instead of their "reduced set" where the output space is restricted to only 37 object categories and 25 test images in total.
This is because the full dataset is much more challenging and the results are more meaningful in statistics.
The performance is reported using the "Multiple Answers" category but the answers are generated using open-ended approach.

$\textbf{MSCOCO-VQA}$ is the latest VQA dataset that contains open-ended questions about images.
This dataset contains 369,861 questions and 3,698,610 ground truth answers based on 123,287 MSCOCO images. These questions and answers are sentence-based and open-ended.
The training and testing split follows MSCOCO-VQA official split. Specifically, we use 82,783 images for training and 40,504 validation images for testing.


## Evaluation criteria

#### DAQUAR

On the DARQUAR dataset, we  use the Wu-Palmer Similarity (WUPS) (["Verbs semantics and lexical selection"]()) score at different thresholds for comparison.

$$
\frac{1}{N}\sum\limits_{i = 1}^N {\min \{ {\prod\limits_{a \in {A_i}} {\mathop {\max }\limits_{t \in {T_i}} \mu ( {a,t} )} ,\prod\limits_{t \in {T_i}} {\mathop {\max }\limits_{a \in {A_i}} \mu ( {t,a} )} } \}}
$$

where, $A_i$ is the predicted answer set of question \emph{i}. $T_i$ is its ground truth answer set. $\mu$ is membership measure, for instance, Wu-Palmer similarity(WUP). $N$ is the total num of questions.
There are three metrics: Standard Metric, Average Consensus Metric and Min Consensus Metric. The Standard Metric is the basic score. The last two metrics are used to study the effects of consensus in question answering tasks.
Please refer to (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) for the details.

$$
\begin{cases}
& \frac{1}{N}\sum\limits_{i = 1}^N {\sum\limits_{k = 1}^K {\min \{ {\prod\limits_{a \in {A_i}} {\mathop {\max }\limits_{t \in {T_i}^k} \mu ( {a,t} )} ,\prod\limits_{t \in {T_i}^k} {\mathop {\max }\limits_{a \in {A_i}} \mu ( {t,a} )} } \}} } \label{Eval:AverageConsensus}\\
& \frac{1}{N}\sum\limits_{i = 1}^N {\mathop {\max }\limits_{k = 1}^K ( {\min \{ {\prod\limits_{a \in {A_i}} {\mathop {\max }\limits_{t \in {T_i}^k} \mu ( {a,t} )} ,\prod\limits_{t \in {T_i}^k} {\mathop {\max }\limits_{a \in {A_i}} \mu ( {t,a} )} } \}} )} \label{Eval:min-consensus}
\end{cases}
$$
where, $T_i ^k$ is the k-th possible human answer corresponding to the k-th interpretation of question.

#### MSCOCO-VQA

On the MSCOCO-VQA dataset, we use the evaluation criteria provided by the organizers.
For the open-ended tasks, the generated answers are evaluated using accuracy metric. It is computed as the percentage of answers that exactly agree with the ground truth provided by human.

## Experimental settings
We choose AlexNet and GoogLeNet in our experiments, respectively.
For AlexNet, the region features are from Pool5 layer. For GoogLeNet we use the inception\_poo5b/output layer.
That means our Compositional Memory processes flattened $36$ regions for AlexNet, and $49$ for GoogLeNet.

In our current implementation, the parameters of region-LSTMs and $\alpha$ gates in Compositional Memory are shared across regions, therefore the computational burden is minimal. However, as regions have to store their respective memory states, the storage space is more than traditional LSTM. In our experiments, the dimension of region inputs $d$ is 1024 for GoogLeNet and 256 for AlexNet. The dimension of the episodes is set to 200 for all LSTMs (including our Compositional Memory) on the DAQUAR dataset and the MSCOCO-VQA.

We used separate dictionaries for training and testing. To generate the dictionary, we first remove the punctuation marks, except those used in timing and measurements, separate them and convert them to lower cases.

We stop our training procedure after 100,000 iterations. The base learning rate is set to be 0.01.
During training, it takes about 0.4 sec for one iteration on GTX Nvidia 780.

## Results on DAQUAR

### Comparisons with state-of-the-art methods
We compare our proposed model with (["A multi-world approach to question answering  about real-world scenes based on uncertain input"]()).
Because they used different CNN methods, we tested both AlexNet and GoogLeNet.
The performance on "Multiple Answers" ("Open-ended") category are shown in Table~\ref{DAQUAR:Comparisons}.

The statistical results shows that the performance of our model substantially is better than the state of the art. On the WUPS$@$0.9, our method is $6\%$ higher (from $23.31\%$ to $29.77\%$).
When we lower the threshold in the WUPS, we are $5.25\%$ superior than the state of the art.

In other two measurements, where "consensus" is calculated on the answers from multiple subjects, our method also outperforms the state of the art $2\%$ to $7\%$.
This further confirms our system is more accurate and robust.

We also find that our method has better performance when GoogLeNet is used in our framework, although the difference is marginally noticeable.


### Effectiveness of Compositional Memory module
We further present our study on the effectiveness of Compositional Memory and the Language LSTM in our VQA system.
Specifically, we show the comparison in performance when we toggle on and off these components.

We consider the configuration where all modules are used as the "full model", and also name the configuration of traditional approach as "baseline".
We then introduce three variants, where only language is used ("Factorized Language Only"), only Compositional Memory is used ("Episodes Only"), and both are used together ("Language+Episodes").

## Results on MSCOCO-VQA
Compared to DAQUAR, MSCOCO-VQA is the latest VQA dataset. It is  much larger and contains more scenes and question types that are not covered by DARQUAR.

Possibly because this is the latest outcome, there are different ways of evaluating the performances and reporting the results.
For example, while the measurement of accuracy is well defined, the evaluation protocols are not standardized.
Some practitioners use the organizer's previous release for training and validating, and further split the validation sets.
Only until recently the organizers release their \texttt{test-dev} set online, however, there are still many ways of handling the input.
For example, the official version (["{VQA:} visual question answering"]()) selects the most frequent $1000$ answers, which covers only $82.67\%$ of the answer set.
Different selections of dictionary can lead to fluctuations in the accuracy.
Finally, the tokenizers used in different practitioners may lead to other uncertainties in accuracy.

Due to the above concerns, we conclude that it is in the early stage of the evaluation, and would like to clearly outline our practices when readers examine the numbers.

- We used a naive tokenizer as specified in Sec. \ref{sec:setting}.
- We used 13,880 words appeared in the training + validation answer set as our answer dictionary.
- We report results on both the \texttt{test-dev} and the full validation set.

We first show results of our method in Fig. ~\ref{fig:VQAexamples_VQA}.
Compared to Fig. \ref{fig:VQAexamples_DAQUAR}, one can see that MSCOCO-VQA is more diversified.
More results are shown in Supplementary Materials.

### Statistical results
MSCOCO-VQA is grouped to a number of categories based on the types of the questions, and the types of answers.
We show the statistics on both categories in this section.

#### Answer type
We report the overall accuracy and those of different answer types using both \texttt{text-dev} and the full validation set .
Please note that we used a larger answer dictionary, which means potentially it is more difficult to deliver correct answers, but still our method achieved similar performance of the state of the art.

One can notice that the accuracy of simple answer type (e.g.  "yes/no") is very high, but the accuracies drop significantly when the answers become more sophisticated.
This indicates the potential issues and directions of our method.


#### Question type
We use validation set to report the accuracy of our method when question type varies (Figure~\ref{fig:vqa_per_type_accuracy}).
The colored bar chart and sorted according to the accuracy, with the numbers displayed next to the bars.

It is interesting to see a significant drop when the questions ascend from simple forms (e.g. , "is there?") to complicated ones (e.g. , "what","how").
This suggest that a practical VQA system needs to take this prior into consideration.



# Conclusion}
In this paper we propose to use the Compositional Memory as the core element in the VQA.
Our end-to-end approach is capable of dynamically extracting local the features.
The Long Short-Term Memory (LSTM) based approach fuses image regions and language, and generates the episodes that is effective for high level reasoning.
Our experiments on the latest public datasets suggest that our method has a superior performance.
