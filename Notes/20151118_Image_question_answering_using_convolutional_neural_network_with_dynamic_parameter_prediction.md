## [Image question answering using convolutional neural networkwith dynamic parameter prediction](http://arxiv.org/abs/1511.05756)


Home : http://cvlab.postech.ac.kr/research/dppnet/

<p align="center"><img src="https://camo.githubusercontent.com/630c2499f26c6685f41caa61508a8a7f1179b9dd/687474703a2f2f7333322e706f7374696d672e6f72672f633677646e623561642f53637265656e5f53686f745f323031365f30355f30385f61745f365f30395f33305f504d2e706e67" width="500" ></p>

- This is a very creative paper that is different from others
- Solves only for single answer question, to reduce the main network to solve for a classification problem
- Other architectures have LSTM as the main framework, this paper uses the convnet / regular DNN as the main body
- Question is embedded via GRUs to dynamically control the weights of the second last layer of the CNN / DNN, which is very original
- To predict a large number of weights in the dynamic parameter layer effectively and efficiently, applies hashing trick, which reduces the number of parameters significantly with little impact on network capacity
- Shows that the hashing trick does not deteriorate the accuracy of the classification network

# ABSTRACT

They tackle <span style="color:red">image question answering (ImageQA)</span> problem by learning a convolutional neural network (CNN) with a dynamic parameter layer whose weights are determined adaptively based on questions.

For the adaptive parameter prediction, they employ a <span style="color:red">separate parameter prediction network</span>, which consists of <span style="color:red">gated recurrent unit (GRU)</span> taking a question as its input and a fully-connected layer generating a set of candidate weights as its output.

Since the dynamic parameter layer is a fully connected layer, it is challenging to predict a large number of parameters in the layer to construct the CNN for ImageQA.

However, it is challenging to construct a parameter prediction network for a large number of parameters in the fully-connected dynamic parameter layer of the CNN.

They reduce the complexity of this problem by incorporating a hashing technique, where the candidate weights given by the parameter prediction network are selected using a predefined hash function to determine individual weights in the dynamic parameter layer.

The proposed network---joint network with the CNN for ImageQA and the parameter prediction network---is trained end-to-end through back-propagation, where its weights are initialized using a pre-trained CNN and GRU.

The proposed algorithm illustrates the state-of-the-art performance on all available public ImageQA benchmarks. such as DAQUAR, COCO-QA and VQA.

# introduction}

## [Holistic scene understanding]()

One of the ultimate goals in computer vision is holistic scene understanding (["Describing the scene as a whole: Joint object detection, scene classification and semantic segmentation"]()), which requires a system to capture various kinds of information such as objects, actions, events, scene, atmosphere, and their relations in many different levels of semantics.

Although significant progress on various recognition tasks has been made in recent years, these works focus only on solving relatively simple recognition problems in controlled settings, where each dataset consists of concepts with similar level of understanding (e.g. object, scene, bird species, face identity, action, texture etc).
There has been less efforts made on solving various recognition problems simultaneously, which is more complex and realistic, even though this is a crucial step toward holistic scene understanding.

## [ImageQA - tackles previously mentioned challenge]()

ImageQA aims to solve the holistic scene understanding problem by proposing a task unifying various recognition problems.

The critical challenge of this problem is that different questions require different types and levels of understanding of an image to find correct answers.

For example, to answer the question like "how is the weather?" we need to perform classification on multiple choices related to weather, while we should decide between yes and no for the question like "is this picture taken during the day?"

For this reason, not only the performance on a single recognition task but also the capability to select a proper task is important to solve ImageQA problem.

Since different question require different type or level of understanding to answer, not only the performance on single recognition task but also the ability to perform proper recognition task selectively is improtant to solve this problem.

## [Existing methods based on a single classifier]()

ImageQA problem has a short history in computer vision and machine learning community, but there already exist several approaches(["Are you talking to a machine? dataset and methods for multilingual image question answering"](),["Learning to answer questions from image using convolutional neural network"](),["A multi-world approach to question answering about real-world scenes based on uncertain input"](),["Ask your neurons: A neural-based approach to answering questions  about images"](),["Exploring models and data for image question answering"]()).


These approaches extract image features using a CNN, and use CNN or bag-of-words to obtain feature descriptors from question.

They can be interpreted as a method that the answer is given by the co-occurrence of a particular combination of features extracted from an image and a question.

# [Their approach]()

Contrary to the existing approaches, they define a different recognition task depending on a question.

To realize this idea, we propose a deep CNN with a dynamic parameter layer whose weights are determined adaptively based on questions.

they claim that a single deep CNN architecture can take care of various tasks by allowing adaptive  weight assignment in the dynamic parameter layer.

For the adaptive parameter prediction, they employ a parameter prediction network, which consists of gated recurrent units (GRU) taking a question as its input and a fully-connected layer generating a set of candidate weights for the dynamic parameter layer.

Since the dynamic parameter layer is a fully connected layer, it is challenging to predict a large number of parameters in the layer to construct the CNN for ImageQA.

they reduce the complexity of this problem by incorporating a hashing technique, where the candidate weights given by the parameter prediction network are selected using a predefined hash function to determine individual weights in the dynamic parameter layer.

The entire network including the CNN for ImageQA and the parameter prediction network is trained end-to-end through back-propagation, where its weights are initialized using pre-trained CNN and GRU.

Their main contributions in this work are summarized below:

- We successfully adopt a deep CNN with a dynamic parameter layer for ImageQA, which is a fully-connected layer whose parameters are determined dynamically based on a given question.
-  To predict a large number of weights in the dynamic parameter layer effectively and efficiently, we apply hashing trick["Compressing neural networks with the hashing trick"](), which reduces the number of parameters significantly with little impact on network capacity.
-  We fine-tune GRU pre-trained on a large-scale text corpus["Skip-thought vectors"]() to improve generalization performance of our network. Pre-training GRU on a large corpus is natural way to deal with a small number of training data, but no one has attempted it yet to our knowledge.
-  This is the first work to report the results on all currently available benchmark datasets such as DAQUAR, COCO-QA and VQA.


# Related Work

## [Bayesian approach]()

There are several recent papers to address ImageQA; the most of them are based on deep learning except.

Malinowski and Fritz["A multi-world approach to question answering about real-world scenes based on uncertain input"]() propose a Bayesian framework, which exploits recent advances in computer vision and natural language processing.

Specifically, it employs semantic image segmentation and symbolic question reasoning to solve ImageQA problem.

However, this method depends on a pre-defined set of predicates, which makes it difficult to represent complex models required to understand input images.

## [Deep learning]()

Deep learning based approaches demonstrate competitive performances in ImageQA

LSTM is used to generate answer phrases.

Most approaches based on deep learning commonly use CNNs to extract features from image while they use different strategies to handle question sentences.

Some algorithms employ embedding of joint features based on image and question.

However, learning a softmax classifier on the simple joint features---concatenation of CNN-based image features and continuous bag-of-words representation of a question---performs better than LSTM-based embedding on COCO-QA dataset.

Another line of research is to utilize CNNs for feature extraction from both image and question and  combine the two features["Learning to answer questions from image using convolutional neural  network"](); this approach demonstrates impressive performance enhancement on DAQUAR ["A multi-world approach to question answering about real-world scenes based on uncertain input"]() dataset by allowing fine-tuning the whole parameters.
%However, predicting the parameters of the network for ImageQA has not been tried.

## [Predicting parameter of neural network]()

Discuss hashing trick here together, but the current paragraph should be reduced a lot and hashing need to be discussed with more importance.

The prediction of the weight parameters in deep neural networks has been explored in ["Predicting deep zero-shot convolutional neural networks using textual descriptions"]() in the context of zero-shot learning.

To perform classification of unseen classes, it trains a multi-layer perceptron to predict a binary classifier for class-specific description in text.
However, this method is not directly applicable to ImageQA since finding solutions based on the combination of question and answer is a more complex problem than the one discussed in ["Predicting deep zero-shot convolutional neural networks using textual descriptions"](), and ImageQA involves a significantly larger set of candidate answers, which requires much more parameters than the binary classification case.

Recently, a parameter reduction technique based on a hashing trick is proposed by Chen ["Compressing neural networks with the hashing trick"]() to fit a large neural network in a limited memory budget.

However, applying this technique to the dynamic prediction of parameters in deep neural networks is not attempted yet to our knowledge.

1. First, contrary to zero-shot learning where each texture-description blongs to one object class, ImageQA have to figure out more complex relation between question and answer.}
2. Second, ImageQA cope with a lot more number of possible classes compare to the mentioned algorithm ["Predicting deep zero-shot convolutional neural networks using textual descriptions"](); the mentioned algorithm is tested on dataset with 200 classes but the largest ImageQA dataset contains more than 20K possible answers.


However, applying their approach to the image question answering is not trivial as a lot more parameters should be involved to perform various operation according to given question compare to binary classification problem.

## learning with multimodal input?

# Algorithm Overview

Briefly describe the motivation and formulation of our approach in this section.

<p align="center"><img src="https://dl.dropboxusercontent.com/s/klivjxulp43bcka/figure2.png?dl=0" width="700" ></p>


# Motivation

Existing approaches (["Learning to answer questions from image using convolutional neural  network"](),["Are you talking to a machine? dataset and methods for multilingual image question answering"](),["Ask your neurons: A neural-based approach to answering questions about images"]()) typically interpret ImageQA as a set of heterogeneous visual recognition problems.

Although ImageQA requires different types and levels of image understanding, existing approaches(["Learning to answer questions from image using convolutional neural  network"](),["Are you talking to a machine? dataset and methods for multilingual image question answering"](),["Ask your neurons: A neural-based approach to answering questions about images"]()) pose the problem as a flat classification task.

However, we believe that it is difficult to solve ImageQA using a single deep neural network with fixed parameters.

In many CNN-based recognition problems, it is well-known to fine-tune a few layers for the adaptation to new tasks.

In addition, some networks are designed to solve two or more tasks jointly by constructing multiple branches connected to a common CNN architecture.

In this work, we hope to solve the heterogeneous recognition tasks using a single CNN by adapting the weights in the dynamic parameter layer.

Since the task is defined by the question in ImageQA, the weights in the layer are determined depending on the question sentence.

In addition, a hashing trick is employed to predict a large number of weights in the dynamic parameter layer and avoid parameter explosion.


# Problem Formulation

ImageQA systems predict the best answer $\hat{a}$ given an image $I$ and a question $q$.
Conventional approaches~\cite{Convqa, mren2015} typically construct a joint feature vector based on two inputs $I$ and $q$ and solve a classification problem for ImageQA using the following equation:

$$
\hat{a} = \underset{a\in{\Omega}}{\operatorname{argmax}} \hspace{0.1cm} p(a \vert {I}, q;{\mathbf{\theta}})
$$
where $\Omega$ is a set of all possible answers and ${\mathbf{\theta}}$ is a vector for the parameters in the network.

On the contrary, we use the question to predict weights in the classifier and solve the problem.
We find the solution by
$$
\hat{a} = \underset{a\in{\Omega}}{\operatorname{argmax}} \hspace{0.1cm} p(a \vert I ;{\mathbf{\theta}}_{s},{\mathbf{\theta}}_{d} (q))
$$
where ${\mathbf{\theta}}_{s}$ and ${\mathbf{\theta}}_{d}(q)$ denote static and dynamic parameters, respectively.
Note that the values of ${\mathbf{\theta}}_{d}(q)$ are determined by the question $q$.



# Network Architecture

## architecture

The network is composed of two sub-networks: classification network and parameter prediction network.

The classification network is a CNN.

One of the fully-connected layers in the CNN is the dynamic parameter layer, and the weights in the layer are determined adaptively by the parameter prediction network.

The parameter prediction network has GRU cells and a fully-connected layer.
It takes a question as its input, and generates a real-valued vector, which corresponds to candidate weights for the dynamic parameter layer in the classification network.

Given an image and a question, our algorithm estimates the weights in the dynamic parameter layer through hashing with the candidate weights obtained from the parameter prediction network.

Then, it feeds the input image to the classification network to obtain the final answer.

## Classification Network

### [Detailed description of classification network]()

The classification network is constructed based on VGG 16-layer net

We remove the last layer in the network and attach three fully-connected layers.

The second last fully-connected layer of the network is the dynamic parameter layer whose weights are determined by the parameter prediction network, and the last fully-connected layer is the classification layer whose output dimensionality is equal to the number of possible answers.
The probability for each answer is computed by applying a softmax function to the output vector of the final layer.


###  [Why not putting dynamic parameters at the end]()

We put the dynamic parameter layer in the second last fully-connected layer instead of the classification layer because it involves the smallest number of parameters.

As the number of parameters in the classification layer increases in proportion to the number of possible answers, predicting the weights for the classification layer may not be a good option to general ImageQA problems in terms of scalability.

Our choice for the dynamic parameter layer can be interpreted as follows.

By fixing the classification layer while adapting the immediately preceding layer, we obtain the task-independent semantic embedding of all possible answers and use the representation of an input embedded in the answer space to solve an ImageQA problem.

Therefore, the relationships of the answers globally learned from all recognition tasks can help solve new ones involving unseen classes, especially in multiple choice questions.
For example, when not the exact ground-truth word (e.g.,, kitten) but similar words (e.g.,, cat and kitty) are shown at training time, the network can still predict the close answers (e.g.,, kitten) based on the globally learned answer embedding.
Even though we could also exploit the benefit of answer embedding based on the relations among answers to define a loss function, we leave it as our future work.
% Since the embedding of answers is globally learnt from all recognition tasks, it helps the network answer unseen words in a specific recognition task.
%Especially in a multiple choice problem, although not the exact ground truth word (e.g. kitten) but similar words (e.g. cat, kitty) were shown at training time for the given recognition task, the network can still predict the closest answer in the choices based on the globally learnt answer embedding.
%We can also utilize the characteristics of answer embedding and relations among answers in a loss function even though we leave it as our future work.}


## Parameter Prediction Network

### paramter

#### [What is dynamic parameter layer]()

As mentioned earlier, our classification network has a dynamic parameter layer.
That is, for an input vector of the dynamic parameter layer ${\bf{f}}^{i}=\left[f^{i}_1,\dots,f^{i}_{\scriptscriptstyle N}\right]^{\scriptscriptstyle T}$, its output vector denoted by ${\bf{f}}^{o}=\left[f^{o}_1,\dots,f^{o}_{\scriptscriptstyle M}\right]^{\scriptscriptstyle T}$ is given by

$$
{\bf{f}}^{o} = {\bf{W}}_{d}(q){\bf{f}}^{i}+{\bf{b}}
$$


where ${\bf{b}}$ denotes a bias and ${\bf{W}}_{d}(q)\in\mathbb{R}^{\scriptscriptstyle M \times N}$ denotes the matrix constructed dynamically using the parameter prediction network given the input question.
%by applying a hash function to the output of the parameter prediction network the given question as its input.
In other words, the weight matrix corresponding to the layer is parametrized by a function of the input question $q$.

### [What is the parameter prediction network]}

The parameter prediction network is composed of GRU cells~\cite{chung2014empirical} followed by a fully-connected layer, which produces the candidate weights to be used for the construction of weight matrix in the dynamic parameter layer within the classification network.

The number of GRU units is equal to the number of words in the question sentence, and the fully-connected layer produces the candidate weights to be used for the prediction of weight parameters in the dynamic parameter layer within the classification network.
GRU, which is similar to LSTM, is designed to model dependency in multiple time scales.


However, contrary to LSTM, which maintains a separate memory cell explicitly, GRU directly updates its hidden states with a reset gate and an update gate.
The detailed procedure of the update is described below.


### [Description of GRU]()

Let $w_{1} ,...,w_{\scriptscriptstyle T}$ be the words in a question $q$, where $T$ is the number of words in the question.
In each time step $t$, given the embedded vector ${\bf{x}}_t$ for a word $w_t$, the GRU encoder updates its hidden state at time $t$, denoted by ${\bf h}_t$, using the following equations:
$$
\begin{cases}
{\bf{r}}_{t} &= \sigma({\bf{W}}_r{\bf{x}}_{t}+{\bf U}_r{\bf{h}}_{t-1}) \\
{\bf{z}}_{t} &= \sigma({\bf{W}}_z{\bf{x}}_{t}+{\bf U}_z{\bf{h}}_{t-1}) \\
\bar{{\bf{h}}}_{t} &= \tanh({\bf{W}}_h{\bf{x}}_{t} + {\bf U}_h({\bf{r}}_{t}\odot{\bf{h}}_{t-1})) \\
{\bf{h}}_{t} &= (1-{\bf{z}}_{t})\odot{\bf{h}}_{t-1}+{\bf{z}}_t\odot\bar{{\bf{h}}}_{t}
\end{cases}
$$
where ${\bf{r}}_{t}$ and ${\bf{z}}_{t}$ respectively denote the reset and update gates at time ${t}$, and $\bar{{\bf{h}}}_{t}$ is candidate activation at time $t$.
In addition, $\odot$ indicates element-wise multiplication operator and $\sigma(\cdot)$ is a sigmoid function.
Note that the coefficient matrices related to GRU such as ${\bf{W}}_r$, ${\bf{W}}_z$, ${\bf{W}}_h$, ${\bf{U}}_r$, ${\bf{U}}_z$, and ${\bf{U}}_h$ are learned by our training algorithm.
By applying this encoder to a question sentence through a series of GRU cells, we obtain the final embedding vector ${\bf{h}}_{\scriptscriptstyle T}\in\mathbb{R}^{\scriptscriptstyle L}$ of the question sentence.


### [Generating output of parameter prediction network]()

Once the question embedding is obtained by GRU, the candidate weight vector, ${\bf{p}}=\left[p_{1},\dots,p_{\scriptscriptstyle K}\right]^{\rm T}$, is given by applying a fully-connected layer to the embedded question ${\bf{h}}_{\scriptscriptstyle T}$ as
$$
{\bf{p}} = {\bf{W}}_{p}{\bf{h}}_{\scriptscriptstyle T}
$$
where ${\bf{p}} \in \mathbb{R}^{\scriptscriptstyle K}$ is the output of the parameter prediction network, and ${\bf{W}}_{p}$ is the weight matrix of the fully-connected layer in the parameter prediction network.
Note that even though we employ GRU for a parameter prediction network since the pre-trained network for sentence embedding---skip-thought vector model~\cite{Skipthought}---is based on GRU, any form of neural networks, e.g.,, fully-connected and convolutional neural network, can be used to construct the parameter prediction network.

## Parameter Hashing

### [Why do we need to reduce parameters?]()

The weights in the dynamic parameter layers are determined based on the learned model in the parameter prediction network given a question.

The most straightforward approach to obtain the weights is to generate the whole matrix ${\bf{W}}_{d}(q)$ using the parameter prediction network.

However, the size of the matrix is very large, and the network may be overfitted easily given the limited number of training examples.

In addition, the number of parameters to construct ${\bf{W}}_{d}(q)$ is closely related to the number of parameters in the parameter prediction network.

In addition, since we need quadratically more parameters between GRU and the fully-connected layer in the parameter prediction network to increase the dimensionality of its output, it is not desirable to predict full weight matrix using the network.

Therefore, it is preferable to construct ${\bf{W}}_{d}(q)$ based on a small number of candidate weights using a hashing trick.

In this case, the output dimensionality of the parameter prediction layer increases in proportion to the product of the input and output dimensionalities of the dynamic parameter layer.

In practice, since the second fully-connected layer in our classification network needs the least parameters among the three fully-connected layers, it is more desirable to determine the weight parameters adaptively for the layer.

This issue is also related to the limited number of candidate weights denoted by $K$; it is better to have a small ratio of the number of parameters predicted to the number of candidate parameters.

Since we need quadratically more parameters between GRU and the fully-connected layer in the parameter prediction network to increase the number of weight candidates, it is also better to maintain a small number of candidate weights.


### [Hash based K reduction]()

We employ the recently proposed random weight sharing technique based on hashing~\cite{Hashing} to construct the weights in the dynamic parameter layer.

Specifically, a single parameter in the candidate weight vector ${\bf{p}}$ is shared by multiple elements of ${\bf{W}}_{d}(q)$, which is done by applying a predefined hash function that converts the 2D location in ${\bf{W}}_{d}(q)$ to the 1D index in ${\bf{p}}$.
By this simple hashing trick, we can reduce the number of parameters in ${\bf{W}}_{d}(q)$ while maintaining the accuracy of the network~\cite{Hashing}.


### [Formal description of hashing]()

Let $w^{d}_{mn}$ be the element at $(m,n)$ in ${\bf{W}}_{d}(q)$, which corresponds to the weight between $m^{\rm th}$ output and $n^{\rm th}$ input neuron.
Denote by $\psi(m,n)$ a hash function mapping a key $(m,n)$ to a natural number in $\left\{1,\dots,K \right\}$, where $K$ is the dimensionality of ${\bf p}$.
The final hash function is given by
$$
w^{d}_{mn}={p}_{\psi(m,n)} \cdot \xi(m,n)
$$
where $\xi(m,n):\mathbb{N}\times\mathbb{N}\rightarrow \{+1, -1 \}$ is another hash function independent of $\psi(m,n)$.
This function is useful to remove the bias of hashed inner product~\cite{Hashing}.
In our implementation of the hash function, we adopt an open-source implementation of {\it{xxHash}}.

### [Why hashing works]()

We believe that it is reasonable to reduce the number of free parameters based on the hashing technique as there are many redundant parameters in deep neural networks ["Predicting parameters in deep learning"]() and the network can be parametrized using a smaller set of candidate weights.
Instead of training a huge number of parameters without any constraint, it would be advantageous practically to allow multiple elements in the weight matrix to share the same value.
It is also demonstrated that the number of free parameter can be reduced substantially with little loss of network performance .



# Training Algorithm

## training


This section discusses the error back-propagation algorithm in the proposed network and introduces the techniques adopted to enhance performance of the network.


## Training by Error Back-Propagation

The proposed network is trained end-to-end to minimize the error between the ground-truths and the estimated answers.
The error is back-propagated by chain rule through both the classification network and the parameter prediction network and they are jointly trained by a first-order optimization method.


## [Notations]()

Let ${\mathcal{L}}$ denote the loss function.
The partial derivatives of ${\mathcal{L}}$ with respect to the $k^{\rm th}$ element in the input and output of the dynamic parameter layer are given respectively by
$$
{\delta}^{i}_k \equiv \frac{\partial\mathcal{L}}{\partial {f}^{i}_k}  ~~~~\text{and}~~~~
{\delta}^{o}_k \equiv \frac{\partial\mathcal{L}}{\partial {f}^{o}_k}.
$$
The two derivatives have the following relation:
$$
{\delta}^{i}_n = \sum _{ m=1 }^{ M }{ w^d_{mn}\delta^o_{m} }
$$
Likewise, the derivative with respect to the assigned weights in the dynamic parameter layer is given by
$$
{\frac{\partial\mathcal{L}}{\partial w^{d}_{mn}}}=f^{i}_{n}{\delta}^{o}_{m}.
$$

## Gradient over Parameter Hashing

As a single output value of the parameter prediction network is shared by multiple connections in the dynamic parameter layer, the derivatives with respect to all shared weights need to be accumulated to compute the derivative with respect to an element in the output of the parameter prediction network as follows:
$$
\begin{cases}
{\frac{\partial\mathcal{L}}{\partial p_{k}}}
&= \sum _{m=1}^{\scriptscriptstyle M}{\sum _{n=1}^{\scriptscriptstyle N}{   {\frac{\partial\mathcal{L}}{\partial w^d_{mn}}} {\frac{\partial w^d_{mn}}{\partial p_{k}}}}}  \nonumber \\
&= \sum _{m=1}^{\scriptscriptstyle M}{\sum _{n=1}^{\scriptscriptstyle N} {  {\frac{\partial\mathcal{L}}{\partial w^d_{mn}}}   {\xi(m,n)}  {{\mathbb{I}} [ \psi(m,n)=k ]}      }},
\end{cases}
$$
where ${\mathbb{I}} [ \cdot ]$ denotes the indicator function.
The gradients of all the preceding layers in the classification and parameter prediction networks are computed by the standard back-propagation algorithm.

## Using Pre-trained GRU

### necessity of employing pre-trained GRU model

Although encoders based on recurrent neural networks (RNNs) such as LSTM and GRU demonstrate impressive performance on sentence embedding([Recurrent neural network based language model](),[Sequence to sequence learning with neural networks]()), their benefits in the ImageQA task are marginal in comparison to bag-of-words model ["Exploring models and data for image question answering"]().

One of the reasons for this fact is the lack of language data in ImageQA dataset.
Contrary to the tasks that have large-scale training corpora, even the largest ImageQA dataset contains relatively small amount of language data; for example, VQA contains 750K questions in total.

Note that the model in ["Sequence to sequence learning with neural networks"]() is trained using a corpus with more than 12M sentences.

## finetuning skip-thought vector

To deal with the deficiency of linguistic information in ImageQA problem, we transfer the information acquired from a large language corpus by fine-tuning the pre-trained embedding network.
We initialize the GRU with the skip-thought vector model trained on a book-collection corpus containing more than 74M sentences ["Skip-thought vectors"]().
Note that the GRU of the skip-thought vector model is trained in an unsupervised manner by predicting the surrounding sentences from the embedded sentences.
As this task requires to understand context, the pre-trained model produces a generic sentence embedding, which is difficult to be trained with a limited number of training examples.
By fine-tuning our GRU initialized with a generic sentence embedding model for ImageQA, we obtain the representations for questions that are generalized better.

## Fine-tuning CNN

### Fine-tuning CNN with GRU is not easy

It is very common to transfer CNNs for new tasks in classification problems, but it is not trivial to fine-tune the CNN in our problem.

jointly with RNNs and achieve noticeable performance improvement~\cite{Baiduqa}.

since gradient vectors from RNNs tend to be noisy.

We observe that the gradients below the dynamic parameter layer in the CNN are noisy since the weights are predicted by the parameter prediction network.
Hence, a straightforward approach to fine-tune the CNN typically fails to improve performance, and
we employ a slightly different technique for CNN fine-tuning to sidestep the observed problem.
We update the parameters of the network using new datasets except the part transferred from VGG 16-layer net at the beginning, and start to update the weights in the subnetwork if the validation accuracy is saturated.

after the gradients through the dynamic parameter layer become less noisy.


## [How to determine "less noisy"?]

## proposed network training overview

The proposed network can be trained end-to-end by back-propagation with an objective of minimizing the error between the ground-truths and the estimated answers.
By chain-rule, the gradient over the training objective is propagated through both the classification network and the parameter prediction network and they are  jointly} trained with first-order optimization methods.

## Gradient over Network Components
### Notations

Let ${\mathcal{L}}$ denotes the objective function for training, then the gradient  with respect to} the output of dynamic parameter layer can be formally represented by $\frac{\partial\mathcal{L}}{\partial {\bf{f}}_{o}}$.
We use $\delta^{o}_{m}$ for the $m$-th element of the $\frac{\partial\mathcal{L}}{\partial {\bf{f}}_{o}}$.
Similarly, we use $\delta^{i}_{n}$ to denote the $n$-th element of $\frac{\partial\mathcal{L}}{\partial {\bf{f}}_{i}}$.

### Gradient over Dynamic Parameter Layer input
The gradient with respect to the output of dynamic parameter layer $\delta^{o}_{m}$ can be computed by {\color{blue}applying the} standard back-propagation  method to the upper layers of the classification network}.

The gradient  with respect to} the input of  the dynamic} parameter layer also can be computed by standard back-propagation as follows:
$$
{\delta}^{i}_n = \sum _{ m=1 }^{ M }{ w^d_{nm}\delta^o_{m} }
$$

 Gradient over Dynamic Parameter Layer weight}
Likewise, the gradient over the assigned weights of the dynamic parameter layer can be computed as follows:
$$
{\frac{\partial\mathcal{L}}{\partial w^{d}_{nm}}}=f^{i}_{n}{\delta}^{o}_{m}
$$

## Gradient over Parameter Hashing

As single output value of the parameter prediction layer is shared by several connections of the dynamic parameter layer, the gradient from each connection need to be accumulated to construct the gradient over the output value of the parameter prediction layer .

$$
{\frac{\partial\mathcal{L}}{\partial p_{k}}}
= \sum _{n=1}^{\scriptscriptstyle N}{\sum _{m=1}^{\scriptscriptstyle M}{   {\frac{\partial\mathcal{L}}{\partial w^d_{nm}}} {\frac{\partial w^d_{nm}}{\partial p_{k}}}}} \\
= \sum _{n=1}^{\scriptscriptstyle N}{\sum _{m=1}^{\scriptscriptstyle M} {  {\frac{\partial\mathcal{L}}{\partial w^d_{nm}}}   {\xi(n,m)}  {{\mathbb{I}}\{ \psi(n,m)=k \}}      }}
$$
where ${\mathbb{I}}\{ \cdot \}$ denotes the indicator function.

## Gradient over Parameter Prediction Layer

The gradients over  the parameters of the parameter prediction  network can  then be computed by  the standard back-propagation from the propagated gradient over the output of this layer.

## Pre-training GRU

necessity of employing pre-trained GRU model

Although RNN based encoders (e.g., LSTM , GRU ) shows impressive performance  on  embedding sentences , its benefit in the ImageQA task is marginal in comparison to the  continuous bag-of-words model ["Exploring models and data for image question answering"]().

One of the possible reason for the marginal performance gain is the lack of language data in ImageQA dataset.

Contrary to the other tasks  for which there are large-scale training corpora} (e.g.,["Sequence to sequence learning with neural networks"]() train the model on a corpus with more than 12M sentences), even the largest ImageQA dataset contains relatively small number of language data (e.g., VQA contains 750K questions in total).

## finetuning skip-thought vector

To deal with the deficiency  of liguistic information in ImageQA datasets, we transfer the information acquired from a large language corpus  by finetuning the pre-trained embedding network}.
 We initialize the GRU with the skip-thought vector model["Skip-thought vectors"]() trained on a book-collection corpus containing more than 74M sentences.

The  GRU of the skip-thought vector model is trained in an unsupervised manner  by predicting the surrounding sentences from the embedded sentence}.

As this task requires to understand the context, the  pre-trained model could produce generic sentence  embedding}, which is hard to be trained with a limited number of examples.

By finetuning  our GRU initialized by the generic sentence embedding model for  ImageQA, we can  obtain a more} generalized question feature extractor.

## Training Details

### data pre-processing

Before training, question sentences are normalized to lower cases and preprocessed by a simple tokenization technique as in ["Show, attend and tell: Neural image caption generation with visual  attention"]().
We normalize the answers to lower cases and regard a whole answer in a single or multiple words as a separate class.

### optimization

The network is trained end-to-end by back-propagation.
["Adam"]() is used for optimization with initial learning rate 0.01.
We clip the gradient to 0.1 to handle the gradient explosion from the recurrent structure of GRU.

Training is terminated when there is no progress on validation accuracy for 5 epochs.

## optmization of dynamic parameter layer

Optimizing the dynamic parameter layer is not straightforward since the distribution of the outputs in the dynamic parameter layer is likely to change significantly in each batch.
Therefore, we apply batch-normalization to the output activations of the layer to alleviate this problem.

## early stopping in GRU finetuning and word-embedding finetuning

In addition, we observe that GRU tends to converge fast and overfit data easily if training continues without any restriction.
We stop fine-tuning GRU when the network start to overfit and continue to train the other parts of the network; this strategy improves performance in practice.
%We stop fine-tuning GRU earlier than the other parts of the network, which helps performance improvement in practice.

# Experiments

We now describe the details of our implementation and evaluate the proposed method in various aspects.

## Datasets

To demonstrate the performance of proposed method on a wide variety of image and question pairs, we evaluate the proposed network on all the available ImageQA benchmarks such as DAQUAR~\cite{Multiworld}, COCO-QA~\cite{mren2015} and VQA~\cite{VQA}.

We evaluate the proposed network on all public ImageQA benchmark datasets such as DAQUAR~\cite{Multiworld}, COCO-QA~\cite{mren2015} and VQA~\cite{VQA}.

They collected question-answer pairs from existing image datasets and most of the answers are single words or short phrases.

### DAQUAR

DAQUAR is based on NYUDv2 ["Indoor segmentation and support inference from rgbd images"]() dataset, which is originally designed for indoor segmentation using RGBD images.

Therefore, most of images are indoor scene and questions are usually about recognizing objects their attributes appearing in indoor scene and their spatial relations.

DAQUAR provides two benchmarks, which are distinguished by the number of classes and the amount of data; DAQUAR-all consists of 6,795 and 5,673 questions for training and testing respectively, and includes 894 categories in answer.

DAQUAR-reduced includes only 37 answer categories for 3,876 training and 297 testing questions.
Some questions in this dataset are associated with a set of multiple answers instead of a single one.

### COCO-QA

The questions in COCO-QA are automatically generated from the image descriptions in MS COCO dataset ["{Microsoft COCO:} common objects in context"]() using the constituency parser with simple question-answer generation rules.

As automatically generated from the image descriptions,

The questions in this dataset are typically long and explicitly classified into 4 types depending on the generation rules: object questions, number questions, color questions and location questions.

All answers are with one-words and there are 78,736 questions for training and 38,948 questions for testing.

### VQA

Similar to COCO-QA, VQA is also constructed on MS COCO but each question is associated with multiple answers annotated by different people.

This dataset contains the largest number of questions: 248,349 for training, 121,512 for validation, and 244,302 for testing, where the testing data is splited into test-dev, test-standard, test-challenge and test-reserve as in Mscoco.

Each question is provided with 10 answers to take the consensus of annotators into account.
About 90\% of answers have single words and 98\% of answers do not exceed three words.


## Evaluation Metrics

As most answers are single word, plain accuracy is used as standard evaluation metric with slight variations for each dataset.

###  WUPS

DAQUAR and COCO-QA employ both classification accuracy and its relaxed version based on word similarity, WUPS ["A multi-world approach to question answering about real-world scenes based on uncertain input"]().

It uses thresholded Wu-Palmer similarity (["Verbs semantics and lexical selection"]()) based on WordNet (["Wordnet: An electronic database"]()) taxonomy to compute the similarity between words.
For predicted answer set $\mathcal{A}^{i}$ and ground-truth answer set $\mathcal{T}^{i}$ of the $i^{\rm th}$ example, WUPS is given by
$$
\begin{cases}
&{\text{WUPS}} = \nonumber \\
&\frac { 1 }{ N } \sum _{ i=1 }^{ N }{ \min { \left\{ \prod _{ a\in \mathcal{A}^{ i } }^{  }{ \max _{ t\in \mathcal{T}^{ i } }{ \mu \left( a,t \right)  }  } , \prod _{ t\in \mathcal{T}^{ i } }^{  }{ \max _{ a\in \mathcal{A}^i }{ \mu \left( a,t \right)  }  }  \right\}  }  } ,
\end{cases}
$$
where $\mu \left(\cdot, \cdot \right)$ denotes the thresholded Wu-Palmer similarity between prediction and ground-truth. where WUP is down-weighted if WUP is less than the threshold.
We use two threshold values ($0.9$ and $0.0$) in our evaluation.

## VQA evaluation (OpenEnded, MultipleChoice)

VQA dataset provides open-ended task and multiple-choice task for evaluation.
For open-ended task, the answer can be any word or phrase while an answer should be chosen out of 18 candidate answers in the multiple-choice task.
In both cases, answers are evaluated by accuracy reflecting human consensus.
For predicted answer $a_i$ and target answer set $\mathcal{T}^{i}$ of the $i^{\rm th}$ example, the accuracy is given by
$$
\text{Acc}_\textrm{VQA} = \frac{1}{N} \sum _{i=1}^{N}{\min { \left\{ \frac{\sum _{t\in \mathcal{T}^{i}}{  {\mathbb{I}} \left[a_i=t\right]}}{3} , 1\right\} }  }
%\min {\left( \frac{\# \text{humans that provided that answer}}{3}, 1 \right)}
$$
where ${\mathbb{I}}\left[ \cdot \right]$ denotes an indicator function.
In other words, a predicted answer is regarded as a correct one if at least three annotators agree, and the score depends on the number of agreements if the predicted answer is not correct.



# Results

We test three independent datasets, VQA, COCO-QA, and DAQUAR.

The proposed Dynamic Parameter Prediction network (DPPnet) outperforms all existing methods nontrivially.

We performed controlled experiments to analyze the contribution of individual components in the proposed algorithm---dynamic parameter prediction, use of pre-trained GRU and CNN fine-tuning, and trained 3 additional models, CONCAT, RAND-GRU, and CNN-FIXED.
CNN-FIXED is useful to see the impact of CNN fine-tuning since it is identical to DPPnet except that the weights in CNN are fixed.
RAND-GRU is the model without GRU pre-training, where the weights of GRU and word embedding model are initialized randomly.
It does not fine-tune CNN either.
CONCAT is the most basic model, which predicts answers using the two fully-connected layers for a combination of CNN and GRU features.
Obviously, it does not employ any of new components such as parameter prediction, pre-trained GRU and CNN fine-tuning.


## General descriptions of controlled models

We train 4 different models: CONCAT, RAND-GRU, CNN-FIXED and DPPnet.
RAND-GRU, CNN-FIXED and DPPnet all use the proposed network with equal parameterization but trained differently.
DPPnet is the model trained with all the components described in this paper: dynamic parameter prediction, usage of pre-trained GRU and CNN finetuning.
To see the effect of fine-tuning CNN, we use CNN-FIXED which is equal to DPPnet before the CNN finetuning.
RAND-GRU trained equally with CNN-FIXED, but weights of GRU and word embeddings are initialized randomly.
CONCAT uses different network to see the effect of dynamic parameter prediction and pre-trained GRU and CNN finetuning is not applied to this model for comparison with RAND-GRU.

##  Detailed configuration of CONCAT

CONCAT combines CNN and GRU features by concatenation and use two-layer neural network as classifier.
We control the number of parameters of CONCAT as similar as possible to RAND-GRU; the dimension of CNN feature is same as the input of parameter prediction layer and the dimension of GRU feature is same as the output of parameter prediction network.
Concatenated features are embedded by a fully-connected layer to have equal dimensionality as the output of the dynamic parameter layer and the embedded feature is fed into the final classifier.
To reduce the effect of batch-normalization, we apply batch-normalization before the final classification layer as well.

## Conclusion

We proposed a novel architecture for image question answering based on two subnetworks---classification network and parameter prediction network.
The classification network has a dynamic parameter layer, which enables the classification network to adaptively determine its weights through the parameter prediction network.
While predicting all entries of the weight matrix is infeasible due to its large dimensionality, we relieved this limitation using parameter hashing and weight sharing.
% With the qualitative results and analysis on the question embedding, we showed that the parameter prediction network successfully determine the weights of dynamic parameter layer based on the specific task the network need to solve.}
The effectiveness of the proposed architecture is supported by experimental results showing the state-of-the-art performances on three different datasets.
Note that the proposed method achieved outstanding performance even without more complex recognition processes such as referencing objects.
We believe that the proposed algorithm can be extended further by integrating attention model ["Show, attend and tell: Neural image caption generation with visual attention"]() to solve such difficult problems.
