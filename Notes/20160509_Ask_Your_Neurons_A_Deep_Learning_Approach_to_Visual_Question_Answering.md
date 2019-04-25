## [Ask Your Neurons: A Deep Learning Approach to VQA](https://arxiv.org/abs/1605.02697)

- [Author homepage](https://github.com/mateuszmalinowski)/[Github.io](http://mateuszmalinowski.github.io/)

<iframe width="560" height="315" src="https://www.youtube.com/embed/QZEwDcN8ehs" frameborder="0" allowfullscreen></iframe>

History: Very recently these two trends of employing neural architectures have been combined fruitfully with methods that can generate image (["Deep visual-semantic alignments for generating image descriptions"]()) and video descriptions (["Sequence to sequence -- video to text"]()). Both are conditioning on the visual features that stem from deep learning architectures and employ recurrent neural network approaches to produce descriptions.

![](https://www.dropbox.com/s/lxwiownhvw7azqn/Ask%20Your%20Neurons-%20A%20Deep%20Learning%20Approach%20to%20Visual%20Question%20Answering.png?dl=1)

### Address textual question answering tasks based on semantic parsing

While there is a rich body of work on natural language understanding that has addressed textual question answering tasks based on semantic parsing, symbolic representation and deduction systems, which also has seen applications to question answering about images (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()), there is evidence that deep architectures can indeed achieve a similar goal (["Memory networks"]()).



An overview is given in \autoref{fig:teaser}. The image is analyzed via a CNN and the question together with the visual representation is fed into a LSTM  network.
The system is trained to produce the correct answer to the question about the image. CNN and LSTM are trained jointly and end-to-end starting from words and pixels.

## Contributions

1. The approach combines a CNN with an LSTM into an end-to-end architecture that predicts answers conditioning on a question and an image.
2. Collect additional data to study human consensus on this task
3. Propose two new metrics sensitive to these effects
4. Provide a new baseline, by asking humans to answer the questions without observing the image.

## Related Work

1. <span style="color:red">CNNs for visual recognition</span>
    <span style="color:red">One component to answer questions about images is to extract information from visual content.</span>
    They evaluate AlexNet(["Imagenet classification with deep CNNs"]()), VGG(["Very deep convolutional networks for large-scale image recognition"]()), GoogleNet(["Going deeper with convolutions"]()), and ResNet(["Deep residual learning for image recognition"]()).
2. <span style="color:red">Encodings for text sequence understanding</span>
    1. The first approach is to encode all words of the question as a Bag-Of-Words(["Foundations of statistical natural language processing"]()), and hence ignoring an order in the sequence of words.
    2. Another option is to use, similar to the image encoding, a CNN with pooling to handle variable length input(["CNNs for sentence classification,A CNN for modelling sentences"]()).
    3. Finally, RNNs are methods developed to directly handle sequences, and have shown recent success on natural language tasks such as machine translation (["Learning phrase representations using rnn encoder-decoder for statistical machine translation,Sequence to sequence learning with neural networks"]()).
3. <span style="color:red">Combining RNNs and CNNs for description of visual content</span>.
    The task of describing visual content like still images as well as videos has been successfully addressed with a combination of encoding the image with CNNs and decoding, \ie predicting the sentence description with an RNN (["Long-term recurrent convolutional networks for visual recognition and description,Deep visual-semantic alignments for generating image descriptions,Translating videos to natural language using deep RNNs,Show and tell: A neural image caption generator,Learning the visual interpretation of sentences"]()). This is achieved by using the RNN model that first gets to observe the visual content and is trained to afterwards predict a sequence of words that is a description of the visual content.

## Grounding of natural language and visual concepts.

Dealing with natural language input does involve the association of words with meaning. This is often referred to as the grounding problem - in particular if the "meaning" is associated with a sensory input. While such problems have been historically addressed by symbolic semantic parsing techniques (["Jointly learning to parse and perceive: Connecting natural language to the physical world,A joint model of language and perception for grounded attribute learning"]())

There is a recent trend of machine learning-based approaches (["Deep visual-semantic alignments for generating image descriptions,Deep fragment embeddings for bidirectional image sentence mapping,Multi-cue zero-shot learning with strong supervision,What are you talking about? text-to-image coreference,Visual7W: Grounded Question Answering in Images,Grounding of textual phrases in images by reconstruction,Generation and comprehension of unambiguous object descriptions"]()) to find the associations.

Answering questions about images can be interpreted as first grounding the question in the image and then predicting an answer. Their approach thus is similar to the latter approaches in that we do not enforce or evaluate any particular representation of "meaning" on the language or image modality. They treat this as latent and leave it to the joint training approach to establish an appropriate hidden representation to link the visual and textual representations.


## Textual question answering.

Answering on purely textual questions has been studied in the NLP community (["Semantic parsing via paraphrasing,Learning dependency-based compositional semantics"]()) and state of the art techniques typically employ semantic parsing to arrive at a logical form capturing the intended meaning and infer relevant answers. Only recently, the success of the previously mentioned neural sequence models, namely RNNs, has carried over to this task (["A neural network for factoid question answering over paragraphs,Memory networks"]()).
More specifically (["A neural network for factoid question answering over paragraphs"]()) use dependency-tree Recursive NN instead of LSTM, and reduce the question-answering problem to a classification task. (["Memory networks"]()) propose different kind of network - memory networks - that is used to answer questions about short stories. In their work, all the parts of the story are embedded into different "memory cells", and next a network is trained to attend to relevant cells based on the question and decode an answer from that. A similar idea has also been applied to question answering about images, for instance by (["Stacked attention networks for image question answering"]()).


## Visual Turing Test

Recently, a large number architectures have been proposed to approach the Visual Turing Test (["Towards a visual turing challenge"]()), frequently also referred to as "VQA". They range from symbolic to neural based approaches. There are also architectures that combine both symbolic and neural paradigms together. Some approaches use explicit visual representation in the form of bounding boxes surrounding objects of interest, while other use global full frame image representation, or soft attention mechanism. Yet others use an external knowledge base that helps in answering questions.

## Symbolic based approaches

In their first work on Visual Turing Test ([" A multi-world approach to question answering about real-world scenes based on uncertain input"]()), they present a question answering system based on a semantic parser on a varied set of human question-answer pairs. Although it is the first attempt to handle question answering on DAQUAR, and despite its introspective benefits, it is a rule-based approach that requires a careful schema crafting, is not that scalable, and finally it strongly depends on the output of visual analysis methods as joint training in this model is not yet possible.
Due to such limitations, the community has rather shifted towards either neural based or combined approaches.

## Deep Neural Approaches with full frame CNN.

Most contemporary approaches use a global image representation, i.e. they encode the whole image with a CNN. Questions are then encoded with an RNN (["Ask your neurons: A neural-based approach to answering questions about images,Image question answering: A visual semantic embedding model and a new dataset,Are you talking to a machine? dataset and methods for multilingual image question answering"]()) or a CNN (["Learning to answer questions from image using CNN"]()).

In contrast to symbolic based approaches, neural based architectures offer scalable and joint end-to-end training that liberates  them from  ontological commitment that would otherwise be introduced by a semantic parser. Moreover, such approaches are not `hard' conditioned on the visual input and therefore can naturally take advantage of different language biases in question answer pairs, which can be interpret as learning common sense knowledge.

## Attention-based Approaches

Following (["Show, attend and tell: Neural image caption generation with visual attention"]()), who proposed to use spatial attention for image description, (["Stacked attention networks for image question answering,Ask, attend and answer: Exploring question-guided spatial attention for VQA,Visual7W: Grounded Question Answering in Images,Abc-cnn: An attention based CNN for VQA,Where to look: Focus regions for VQA"]()) predict a latent weighting (attention) of spatially localized images features (typically a convolutional layer of the CNN) based on the question. The weighted image representation rather than the full frame feature representation is then  used as a basis for answering the question.

In contrast to the previous models using attention, Dynamic Memory Networks (DMN) (["Ask me anything: Dynamic memory networks for natural language processing,Dynamic memory networks for visual and textual question answering"]()) first pass all spatial image features through a bi-directional GRU that captures spatial information from the neighboring image patches, and next retrieve an answer from a recurrent attention based neural network that allows to focus only on a subset of the visual features extracted in the first pass.

Another interesting direction has been taken by (["A focused dynamic attention model for VQA"]()) who run state-of-the-art object detector of the classes extracted from the key words in the question. In contrast to other attention mechanisms, such approach offers a focused, question dependent, "hard" attention.

## Answering with an external knowledge base

(["Ask Me Anything: Free-form VQA Based on Knowledge from External Sources"]()) argue for an approach that first represents an image as an intermediate semantic attribute representation, and next query external knowledge sources based on the most prominent attributes and relate them to the question. With the help of such external knowledge base, such approach captures richer semantic representation of the world, beyond what is directly contained in images.

## Compositional approaches

A different direction is taken by (["Learning to compose neural networks for question answering,Neural module networks"]()) who predict the most important components to answer the question with a natural language parser. The components are then mapped to neural modules, which are composed to a deep neural network based on the parse tree. While each question induces a different network, the modules are trained jointly across questions. This work compares to (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) by exploiting explicit assumptions about the compositionality of natural language sentences.

Related to the Visual Turing Test, ([" A pooling approach to modelling spatial relations for image retrieval and annotation"]()) have also combined a neural based representation with the compositionality of the language for the text-to-image retrieval task.

## Dynamic parameters

(["Image question answering using CNN with dynamic parameter prediction"]()) have an image recognition network and a Recurrent Neural Network (GRU) that dynamically change the parameters (weights) of visual representation based on the question. More precisely, the parameters of its second last layer are dynamically predicted from the question encoder network and in this way changing for each question. While question encoding and image encoding is pre-trained, the network learns parameter prediction only from image-question-answer triples.

## Datasets for VQA

Datasets are a driving force for the recent progress in VQA.

### DAQUAR

We evaluate our approach on this dataset and discuss several consensus evaluation metrics that take the extended annotations into account.

In parallel to our Visual Turing Test, (["Visual turing test for computer vision systems"]()) developed another Visual Turing Test. Their work, however, focuses on yes/no type of questions, and provide detailed object-scene annotations.

### MS-COCO

Shortly after the introduction of DAQUAR, three other large-scale datasets have been proposed. All are based on MS-COCO (["Microsoft coco: Common objects in context"]()).
(["Are you talking to a machine? dataset and methods for multilingual image question answering"]()) have annotated about $158k$ images with $316k$  Chinese question answer pairs together with their corresponding English translations. (["Image question answering: A visual semantic embedding model and a new dataset"]()) have taken advantage of the existing annotations for the purpose of image description generation task and transform them into question answer pairs with the help of a set of hand-designed rules and a syntactic parser (["Accurate unlexicalized parsing"]()). This procedure has approximately generated $118k$  question answer pairs.

### VQA

Finally, arguably nowadays the most popular, large scale dataset on question answering about images is (["VQA"]()). It has approximately $614k$  questions about the visual content of about $205k$  real-world images. Similarly to our Consensus idea, VQA provides $10$ answers per each image. For the purpose of the challenge the test answers are not publicly available. We perform one part of the experimental analysis in this paper on the VQA dataset, examining different variants of our proposed approach.

Although simple, automatic  performance evaluation metrics have been a part of building first VQA datasets (["A multi-world approach to question answering about real-world scenes based on uncertain input,Towards a visual turing challenge,Hard to cheat: A turing test based on answering questions about images"]()), (["Visual madlibs: Fill in the blank description generation and question answering"]()) have simplified the evaluation even further by introducing Visual Madlibs - a multiple choice question answering by filling the blanks task. In this task, a question answering architecture has to choose one out of four provided answers for a given image and the prompt. Formulating question answering task in this way has wiped out ambiguities in answers, and just a simple accuracy metric can be used to evaluate different architectures on this task. Yet, the task requires holistic reasoning about the images, and despite of simple evaluation, it remains challenging for machines.

### Visual7W

The Visual7W (["Visual7W: Grounded Question Answering in Images"]()) extends canonical question and answer pairs with additional groundings of all objects appearing in the questions and answers to the image by annotating the correspondences. It contains natural language answers, but also answers which require to locate the object, which is then similar to the task of explicit grounding discussed above. Visual7W builds part of the $1.7$ million question answer pairs in the Visual Genome dataset (["Visual genome: Connecting language and vision using crowdsourced dense image annotations"]()). While not all question are grounded in the Visual Genome datasets it forms the largest dataset with respect to the number of questions.

In contrast to others such as (["VQA"]()) or DAQUAR (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) that has collected unconstrained question answer pairs, the Visual Genome focuses on the six, so called, Ws: $\mathbf{what}$, $\mathbf{where}$, $\mathbf{when}$, $\mathbf{who}$, $\mathbf{why}$, and $\mathbf{how}$, which can be answered with a natural language answer. An additional 7th question --~$\mathbf{which}$~-- requires a bounding box location as answer.
Similarly to Visual Madlibs (["Visual madlibs: Fill in the blank description generation and question answering"]()), Visual7W also contains multiple-choice answers.

Related to Visual Turing Test, (["Contextual media retrieval using natural language queries"]()) have proposed collective memories and Xplore-M-Ego - a dataset of images with natural language queries, and a media retrieval system. This work focuses on a user centric, dynamic scenario, where the provided answers are conditioned not only on questions but also on the geographical position of the questioner.

Moving from asking questions about images to questions about video enhances typical questions with temporal structure. (["Uncovering temporal context for video question and answering"]()) propose a task which requires to fill in blanks the captions associated with videos. The task requires inferring the past, describing the present and predicting the future in a diverse set of video description data ranging from cooking videos (["Grounding Action Descriptions in Videos"]()) over web videos (["Trecvid"]()) to movies (["A dataset for movie description"]()). (["tapaswi16cvpr"]()) propose MovieQA, which requires to understand long term connections in the plot of the movie.

Given the difficulty of the data, both works provide multiple-choice answers.

## Relations to our work

The original version of this work (["Ask your neurons: A neural-based approach to answering questions about images"]()) belongs to the category of "Deep Neural Approaches with full frame CNN", and is among the very first methods of this kind (\autoref{sec:iccvArch}). We extend (["Ask your neurons: A neural-based approach to answering questions about images"]()) by introducing a more general and modular encoder-decoder perspective (\autoref{sec:alternative_approaches}) that encapsulates a few different neural approaches. Next, we broaden our original analysis done on DAQUAR (\autoref{sec:results}) to the analysis of different neural based approaches on VQA showing the importance of getting a few details right together with benefits of a stronger visual encoder (\autoref{section:analysis_on_vqa}). Finally, we transfer lessons learnt from ["VQA"]()) to DAQUAR (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()), showing a significant improvement on this challenging task (\autoref{section:analysis_on_vqa}).

# method

Answering questions about images can be formulated as the problem of predicting an answer, given an image $\bs{x}$ and a question $\bs{q}$ according to a parametric probability measure:

$$
\label{eq:problem}
\bs{\hat{a}}=\argmax_{\bs{a}\in \mathcal{A}}p(\bs{a}|\bs{x},\bs{q};\bs{\theta})
$$
where $\bs{\theta}$ represent a vector of all parameters to learn and $\mathcal{A}$ is a set of all answers. %
The question $\bs{q}$ is a sequence of words, i.e., $\bs{q}=\left[\bs{q}_1,\ldots,\bs{q}_{n}\right]$, where each $\bs{q}_t$ is the $t$-th word question  with $\bs{q}_n= "?"$ encoding the question mark - the end of the question.
In the following we describe how we represent $\bs{x}$, $\bs{a}$, $\bs{q}$, and $p(\cdot|\bs{x},\bs{q};\bs{\theta})$ in more details.

In a scenario of \textbf{multiple word answers}, we consequently decompose the problem to predicting a set of answer words $\bs{a}_{\bs{q},\bs{x}} = \left\{\bs{a}_1, \bs{a}_2, ..., \bs{a}_{\mathcal{N}(q,x)}\right\}$, where $\bs{a}_t$ are words from a finite vocabulary $\mathcal{V'}$, and $\mathcal{N}(q,x)$ is the number of answer words for the given question and image.

In our approach, named \AproachName, we propose to tackle the problem as follows.  To predict multiple words we formulate the problem as predicting a sequence of words from the vocabulary $\mathcal{V}:=\mathcal{V'}\cup\left\{\$\right\}$ where the extra token $\$$ indicates the end of the answer sequence, and points out that the question has been fully answered. %
We thus formulate the prediction procedure recursively:
$$
\label{eq:recursivePred}
\bs{\hat{a}}_t=\argmax_{\bs{a}\in \mathcal{V}}p(\bs{a}|\bs{x},\bs{q},\hat{A}_{t-1};\bs{\theta})
$$
where $\hat{A}_{t-1}=\left\{\bs{\hat{a}}_1,\ldots,\bs{\hat{a}_{t-1}}\right\}$ is the set of previous words, with $\hat{A}_{0}=\left\{\right\}$ at the beginning, when our approach has not given any answer word so far. The approach is terminated when $\hat{a}_t=\$$.
We evaluate the method solely based on the predicted answer words ignoring the extra token $\$$.

To ensure uniqueness of the predicted answer words, as we want to predict the $\mathbf{set} of answer words, the prediction procedure can be trivially changed by maximizing over $\mathcal{V}\setminus\hat{A}_{t-1}$. However, in practice, our algorithm learns to not predict any previously predicted words.

If we only have \textbf{single word answers}, or if we model each multi-word answer as a different answer (\ie vocabulary entry), we directly use \autoref{eq:problem}.

In the following we first present a \iccvArch that models multi-word answers with a single recurrent network for question and image encoding and answer prediction (\autoref{sec:iccvArch}) and then present a more general and modular framework with question and image encoders, as well as answer decoder as  modules.

## Method

As shown in \autoref{fig:teaser} and \autoref{fig:lstm-approach}, we  feed our approach \AproachName with a question as a sequence of words.
Since our problem is formulated as a variable-length input output sequence,
we decide to model the parametric distribution $p(\cdot|\bs{x},\bs{q};\bs{\theta})$ of  \AproachName with a recurrent neural network and a  softmax prediction layer. More precisely, \AproachName is a deep network built of CNN (["Gradient-based learning applied to document recognition"]()) and Long-Short Term Memory (LSTM) (["Long short-term memory"]()). We decide on LSTM as it has been recently shown to be effective in learning a variable-length sequence-to-sequence mapping (["Long-term recurrent convolutional networks for visual recognition and description,Sequence to sequence learning with neural networks"]()).

Both question and answer words are represented with one-hot vector encoding (a binary vector with exactly one non-zero entry at the position indicating the index of the word in the vocabulary) and embedded in a lower dimensional space, using a jointly learnt latent linear embedding.

In the training phase, we augment the question words sequence $\bs{q}$ with the corresponding ground truth answer words sequence $\bs{a}$, \ie $\bs{\hat{q}} := \left[\bs{q}, \bs{a}\right]$. During the test time, in the prediction phase, at time step $t$, we augment $\bs{q}$ with previously predicted answer words $\hat{\bs{a}}_{1..t} := \left[\bs{\hat{a}}_1,\ldots,\bs{\hat{a}}_{t-1}\right]$, \ie $\bs{\hat{q}}_{t} := \left[\bs{q},\bs{\hat{a}}_{1..t}\right]$.
This means the question $\bs{q}$ and the previous answer words are encoded implicitly in the hidden states of the LSTM, while the latent hidden representation is learnt. We encode the image $\bs{x}$ using a CNN and provide it at every time step as input to the LSTM.
We set the input $\bs{v}_t$ as a concatenation of $\left[\Phi(\bs{x}), \bs{\hat{q}}_t\right]$, where $\Phi(\cdot)$ is the CNN encoding.


## ## Long-Short Term Memory ()}

As visualized in detail in \autoref{fig:lstm}, the LSTM unit takes an input vector $\bs{v}_t$ at each time step $t$ and predicts an output word  $\bs{z_t}$ which is equal to its latent hidden state $\bs{h}_t$. As discussed above $\bs{z_t}$ is a linear embedding of the corresponding answer word $\bs{a_t}$.

In contrast to a simple RNN unit the LSTM unit additionally maintains a memory cell $\bs{c}$. This allows to learn long-term dynamics more easily and significantly reduces the vanishing and exploding gradients problem~(["Long short-term memory"]()). More precisely, we use the LSTM unit as described in (["Learning to execute"]()).

With the  sigmoid} nonlinearity $\sigma:\mathbb{R} \mapsto [0, 1]$, $\sigma(v) = \left(1 + e^{-v}\right)^{-1}$ and the hyperbolic tangent} nonlinearity $\phi:\mathbb{R} \mapsto [-1, 1]$, $\phi(v) = \frac{e^v - e^{-v}}{e^v + e^{-v}} = 2\sigma(2v) - 1$, the LSTM updates for time step $t$ given inputs $\bs{v}_t$, $\bs{h}_{t-1}$, and the memory cell $\bs{c}_{t-1}$ as follows:
$$
\begin{cases}
\mathbf{i}_t &= \sigma(W_{vi}\bs{v}_t + W_{hi}\bs{h}_{t-1} + \bs{b}_i)\label{eq:i}\\
\mathbf{f}_t &= \sigma(W_{vf}\bs{v}_t + W_{hf}\bs{h}_{t-1} + \bs{b}_f)\label{eq:f} \\
\mathbf{o}_t &= \sigma(W_{vo}\bs{v}_t + W_{ho}\bs{h}_{t-1} + \bs{b}_o) \label{eq:o}\\
\mathbf{g}_t &=   \phi(W_{vg}\bs{v}_t + W_{hg}\bs{h}_{t-1} + \bs{b}_g)\label{eq:g} \\
\mathbf{c}_t &= \bs{f}_t \odot \bs{c}_{t-1} + \bs{i}_t \odot \bs{g}_t \label{eq:c}\\
\mathbf{h}_t &= \bs{o}_t \odot \phi(\bs{c}_t)\label{eq:h}
\end{cases}
$$

where $\odot$ denotes element-wise multiplication.
All the weights $W$ and biases $b$ of the network are learnt jointly with the cross-entropy loss. Conceptually, as shown in \autoref{fig:lstm},  \autoref{eq:i} corresponds to the input gate, \autoref{eq:g} the input modulation gate, and \autoref{eq:f} the forget gate, which determines how much to keep from the previous memory $c_{t-1}$ state.
As \Figsref{fig:teaser} and \ref{fig:lstm-approach} suggest, all the output predictions that occur before the question mark are excluded from the loss computation, so that the model is penalized solely based on the predicted answer words.

## Question encoders

The main goal of a question encoder is to capture a meaning of the question, which we write here as $\qenc{}$.
Such an encoder can range from a very structured one like Semantic Parser used in (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) and (["Learning dependency-based compositional semantics"]()) that explicitly model compositional nature of the question, to structureless Bag-Of-Word (BOW) approaches that temporarily sum up the input question words (\autoref{fig:bow-approach}).

In this work, we investigate a few encoders within such spectrum. Two recurrent question encoders, LSTM (["Long short-term memory"]()) (see \autoref{sec:LSTM}) and GRU (["cho2014learning"]()), that assume a temporal ordering in questions, as well as the aforementioned BOW.

### Gated Recurrent Unit (GRU)

GRU is a simpler variant of LSTM that also uses gates (a reset gate $\bs{r}$ and an update gate $\bs{u}$) in order to keep long term dependencies. GRU is expressed by the following set of equations:
$$
\begin{cases}
\mathbf{r}_t &= \sigma(W_{vr}\bs{v}_t + W_{hr}\bs{h}_{t-1} + \bs{b}_r)\\
\mathbf{u}_t &= \sigma(W_{vu}\bs{v}_t + W_{hu}\bs{h}_{t-1} + \bs{b}_u) \\
\mathbf{c}_t &= W_{vc}\bs{v}_t + W_{hc}\bs{h}_{t-1} + \bs{b}_c \\
\mathbf{h}_t &= \bs{u}_t \odot \bs{h}_{t-1} + (\bs{1} - \bs{u}_t) \odot \phi(\bs{c}_t)
\end{cases}
$$
where $\sigma$ is the sigmoid function, $\phi$ is the hyperbolic tangent, and $\bs{v}_t$, $\bs{h}_t$ are input and hidden state at time $t$. The representation of the question $\bs{q}$ is the hidden vector at last time step, \ie $ \qenc{RNN} := \bs{h}_T$.


### CNN
CNN that models language (["CNNs for sentence classification,A CNN for modelling sentences,Learning to answer questions from image using CNN,Stacked attention networks for image question answering"]()) is gaining popularity due to its speed and good accuracy for the language-oriented tasks. Since it considers a larger context, it arguably maintains more structure than BOW but does not model such long term dependencies as RNNs. \autoref{fig:cnn_lang-approach} depicts our CNN architecture, which is very similar to (["Learning to answer questions from image using CNN"]()) and (["Stacked attention networks for image question answering"]()), that convolves word embeddings (we either learn it jointly with the whole model or use GLOVE (["Glove: Global vectors for word representation"]()) in our experiments) with three convolutional kernels of length $1$, $2$ and $3$ (for the sake of clarity, we only show two kernels in the Figure). We call such architecture with $1$, ..., $n$ kernel lengths $n$ views CNN. At the end, the kernel's outputs are temporarily aggregated for the final question's representation. We use either sum pooling or a recurrent neural network (CNN-RNN) to accomplish this step.

## Visual encoders

The second important component of the encoder-decoder architectures for Visual Turing Test is visual representation. Nowadays, CNNs (CNNs) become the state-of-the-art framework that provide features from images. The typical protocol of using the visual models is to first pre-train them on the ImageNet dataset (["Imagenet large scale visual recognition challenge"]()), a large scale recognition dataset, and next use them as an input for the rest of the architecture. Fine-tuning the weights of the encoder to the task at hand is also possible.

In our experiments, we use chronologically the oldest CNN architecture fully trained on ImageNet -- a Caffe implementation of AlexNet (["Caffe: Convolutional architecture for fast feature embedding,Imagenet classification with deep CNNs"]()) -- as well as the recently introduced deeper networks -- Caffe implementations of GoogLeNet and VGG (["Going deeper with convolutions, Very deep convolutional networks for large-scale image recognition"]()) -- to the most recent extremely deep architectures -- a Facebook implementation of $152$ layered ResidualNet (["Deep residual learning for image recognition"]()). As can be seen from our experiments in \autoref{section:analysis_on_vqa}, a strong visual encoder plays an important role in Visual Turing Test.

## Multimodal embedding

The presented neural question encoders transform linguistic question into a vector space. Similarly visual encoders encode images as vectors. A multimodal fusion module combines both vector spaces into another vector space that decoding of answers is feasible.

Let $\qenc{}$ be a question representation (BOW, CNN, LSTM, GRU), and $\ienc{}$ be a representation of an image. Then $C(\qenc{} ,\ienc{})$ is a function which embeds both vectors.
In this work, we investigate three multimodal embedding techniques: Concatenation, piecewise multiplication, and summation. Since the last two techniques require compatibility in the number of feature components, we use additional visual embedding matrix $W_{ve} \in \mathbb{R}^{|\qenc{}| \times |\ienc{}|}$.

Let $W$ be weights of an answer decoder. Then we have $W C(\qenc{}, \ienc{})$, which is
$$
\begin{cases}
  W_{q} & \qenc{} + W_{v} \ienc{} \label{eq:concat_fusion}\\
  W (&\qenc{} \odot W_{ve} \ienc{}) \label{eq:piecewise_mult_fusion} \\
  W &\qenc{} + W W_{ve} \ienc{} \label{eq:summation_fusion}
\end{cases}
$$
in concatenation, piecewise multiplication, and summation fusion techniques respectively. In \autoref{eq:concat_fusion}, we decompose $W$ into two matrices $W_q$ and $W_v$, that is $W = \left[W_q; W_v\right]$. In \autoref{eq:piecewise_mult_fusion}, $\odot$ is a piecewise multiplication. Similarity between \autoref{eq:concat_fusion} and \autoref{eq:summation_fusion} is interesting as the latter is the former with weight sharing and additional decomposition into $W W_{ve}$.

## Answer decoders

### Answer words generation

The last component of the encoder-decoder architecture for Visual Turing Test (\autoref{fig:vqa_encoder_decoder}) is an answer decoder. (["Ask your neurons: A neural-based approach to answering questions about images"]()), inspired by the work on the image description task (["Long-term recurrent convolutional networks for visual recognition and description"]()), uses an LSTM as decoder that shares the parameters with the encoder.

### Classification

An alternative approach that cast answering problem as a classification task, with answers as different classes, has recently gained popularity, especially in VQA task (["Vqa:VQA"]()). Thorough this work, we investigate both approaches.

# Analysis on DAQUAR

In this section we benchmark our method on a task of answering questions about images. We compare different variants of our proposed model to prior work in Section \ref{sec:experiments:eval}. In addition, in Section \ref{sec:experiments:evalNoImg}, we analyze how well questions can be answered without using the image in order to gain an understanding of biases in form of prior knowledge and common sense.

We provide a new human baseline for this task.
In Section \ref{sec:experiments:humanConsensus} we discuss ambiguities in the question answering tasks and analyze them further by introducing metrics that are sensitive to these phenomena. In particular, the WUPS score (["A multi-world approach to question answering about real-world scenes based on uncertain input} is extended to a consensus metric that considers multiple human answers. All the material is available on the project webpage \footnote{\url{http://mpii.de/visual_turing_test}}.

### Experimental protocol

We evaluate our approach from \autoref{sec:method} on the DAQUAR dataset (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) which provides $12,468$ human question answer pairs on images of indoor scenes (["Indoor segmentation and support inference from rgbd images"]()) and follow the same evaluation protocol by providing results on  accuracy and the WUPS score at $\left\{0.9,0.0\right\}$.
We run experiments for the full dataset as well as their proposed reduced set that restricts the output space to only $37$ object categories and uses $25$ test images. In addition, we also evaluate the methods on different subsets of DAQUAR where only $1$, $2$, $3$ or $4$ word answers are present.

We use default hyper-parameters of LSTM (["Long-term recurrent convolutional networks for visual recognition and description"]()) and CNN (["Caffe: Convolutional architecture for fast feature embedding"]()). All CNN models are first pre-trained on the ImageNet dataset (["Imagenet large scale visual recognition challenge"]()), and next we randomly initialize and train the last layer together with the LSTM network on the task. We find this step crucial in obtaining good results.
We have explored the use of a 2 layered LSTM model, but have consistently obtained worse performance.
In a pilot study, we have found that GoogleNet} architecture (["Caffe: Convolutional architecture for fast feature embedding,Going deeper with convolutions"]()) consistently outperforms the AlexNet} architecture (["Caffe: Convolutional architecture for fast feature embedding,Imagenet classification with deep CNNs"]()) as a CNN model for our task and model.

### WUPS scores

We base our experiments and the consensus metrics on WUPS scores (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()). The metric is a generalization of the accuracy measure that accounts for word-level ambiguities in the answer words. For instance `carton' and `box' can be associated with a similar concept, and hence models should not be strongly penalized for this type of mistakes. Formally:

$$
\textrm{WUPS}(A,T) = \frac{1}{N} \sum_{i=1}^N\min\{ \prod_{a \in A^i} \max_{t\in T^i} \mu(a, t) ,\; \\ \prod_{t \in T^i} \max_{a \in A^i} \mu(a, t)\}
$$

To embrace the aforementioned ambiguities, (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) suggest using a thresholded taxonomy-based Wu-Palmer similarity (["Verbs semantics and lexical selection"]()) for $\mu$. Smaller thresholds yield more forgiving metrics. As in (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()), we report WUPS at two extremes, $0.0$ and $0.9$.


## Evaluation

We start with the evaluation of our \AproachName on the full DAQUAR dataset in order to study different variants and training conditions. Afterwards we evaluate on the reduced DAQUAR for additional points of comparison to prior work.

### Results on full DAQUAR

\autoref{table:full_daquar} shows the results of our \AproachName method on the full set ("multiple words") with $653$ images and $5673$ question-answer pairs available at test time. In addition, we evaluate a variant that is trained to predict only a single word ("single word") as well as a variant that does not use visual features ("Question-only"). Note, however, that "single word" refers to a training procedure. All the methods in \autoref{table:full_daquar} are evaluated on the full DAQUAR dataset at test time.
In comparison to the prior work (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) (shown in the first row in \autoref{table:full_daquar}), we observe strong improvements of over $9\%$ points in accuracy and over $11\%$ in the WUPS scores [second row in \autoref{table:full_daquar} that corresponds to "multiple words"]. Note that, we achieve this improvement despite the fact that the only published number available for the comparison on the full set uses ground truth object annotations (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) -- which puts our method at a disadvantage.
Further improvements are observed when we train only on a single word answer, which doubles the accuracy obtained in prior work.
We attribute this to a joint training of the language and visual representations and the dataset bias, where about $90\%$ of the answers contain only a single word.

We further analyze this effect in \autoref{fig:exactly_n_words}, where we show performance of our approach ("multiple words") in dependence on the number of words in the answer (truncated at 4 words due to the diminishing performance). The performance of the "single word" variants on the one-word subset are shown as horizontal lines. Although accuracy drops rapidly for longer answers, our model is capable of producing a significant number of correct two words answers. The "single word" variants have an edge on the single answers and benefit from the dataset bias towards these type of answers. Quantitative results of the "single word" model on the one-word answers subset of DAQUAR are shown in  \autoref{table:subset_single_word}.
While we have made substantial progress compared to prior work, there is still a $30\%$ points margin to human accuracy and $25$ in WUPS score ["Human answers" in \autoref{table:full_daquar}].

Later on, in \autoref{sec:soa}, we will show improved results on DAQUAR with a stronger visual model and a pre-trained word embedding, with ADAM (["Adam: A method for stochastic optimization"]()) as the chosen optimization technique. We also put the method in a broader context, and compare with other approaches.

### Results on reduced DAQUAR

In order to provide performance numbers that are comparable to the proposed Multi-World approach in (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()), we also run our method on the reduced set with $37$ object classes and only $25$ images with $297$ question-answer pairs at test time.

\autoref{table:reduced_daquar} shows that \AproachName also improves on the reduced DAQUAR set, achieving $34.68\%$ Accuracy and $40.76\%$ WUPS at 0.9 substantially outperforming (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) by  $21.95\%$ Accuracy and $22.6$ WUPS. Similarly to previous experiments, we achieve the best performance using the "single word" variant.

## Answering questions without looking at images}

In order to study how much information is already contained in questions, we train a version of our model that ignores the visual input.

The results are shown in \autoref{table:full_daquar} and \autoref{table:reduced_daquar} under "Question-only (ours)".
The best "Question-only" models with $17.15\%$ and $32.32\%$ compare very well in terms of accuracy to the best models that include vision. The latter achieve $19.43\%$ and $34.68\%$ on the full and reduced set respectively.

In order to further analyze this finding, we have collected a new human baseline "Human answer, no image", where we have asked participants to answer on the DAQUAR questions without looking at the images. It turns out that humans can guess the correct answer in $7.86\%$ of the cases by exploiting prior knowledge and common sense. Interestingly, our best "Question-only" model outperforms the human baseline by over $9\%$.

A substantial number of answers are plausible and resemble a form of common sense knowledge employed by humans to infer answers without having seen the image.

## Human Consensus

### resultTableReduced

We observe that in many cases there is an inter human agreement in the answers for a given image and question and this is also reflected by the human baseline performance on the question answering task of $50.20\%$ ["Human answers" in \autoref{table:full_daquar}].

We study and analyze this effect further by extending our dataset to multiple human reference answers in \autoref{sec:extended_consensus_annotation}, and proposing a new measure -- inspired by the work in psychology (["cohen1960coefficient,fleiss1973equivalence,nakashole2013fine"]()) -- that handles disagreement in \autoref{sec:consensus_measure}, as well as conducting additional experiments in \autoref{sec:consensus_results}.

In order to study the effects of consensus in the question answering task, we have asked multiple participants to answer the same question of the DAQUAR dataset given the respective image.

We follow the same scheme as in the original data collection effort, where the answer is a set of words or numbers. We do not impose any further restrictions on the answers.

This extends the original data (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) to an average of $5$ test answers per image and question. We refer to this dataset as \daquarNew.

## Consensus Measures

While we have to acknowledge inherent ambiguities in our task, we seek a metric that prefers an answer that is commonly seen as preferred. We make two proposals:

### Average Consensus

We use our new annotation set that contains multiple answers per question in order to compute an expected score in the evaluation:
$$
\frac{1}{N K} \sum_{i=1}^N \sum_{k=1}^K \min\{ \prod_{a \in A^i} \max_{t\in T^i_k} \mu(a, t) ,\; \prod_{t \in T^i_k} \max_{a \in A^i} \mu(a, t)\}
$$
where for the $i$-th question $A^i$ is the answer generated by the architecture and $T^i_k$ is the $k$-th possible human answer corresponding to the $k$-th interpretation of the question.
Both answers $A^i$ and $T^i_k$ are sets of the words, and $\mu$ is a membership measure, for instance WUP (["Verbs semantics and lexical selection"]()). We call this metric "Average Consensus Metric (ACM)" since, in the limits, as $K$ approaches the total number of humans, we truly measure the inter human agreement of every question.

## Consensus results

Using the multiple reference answers in \daquarNew we can show a more detailed analysis of inter human agreement. \autoref{fig:consensus_quality} shows the fraction of the data where the answers agree between all available questions ("100"), at least $50\%$ of the available questions and do not agree at all (no agreement - "0"). We observe that for the majority of the data, there is a partial agreement, but even full disagreement is possible. We split the dataset into three parts according to the above criteria "No agreement", "$\ge 50\%$ agreement" and "Full agreement" and evaluate our models on these splits (\autoref{table:results_agreement_daquar} summarizes the results).
On subsets with stronger agreement, we achieve substantial gains of up to $10\%$ and $20\%$ points in accuracy over the full set (\autoref{table:full_daquar}) and the \textbf{Subset: No agreement} (\autoref{table:results_agreement_daquar}), respectively.

These splits can be seen as curated versions of DAQUAR, which allows studies with factored out ambiguities.

The aforementioned "Average Consensus Metric" generalizes the notion of the agreement, and encourages predictions of the most agreeable answers. On the other hand "Min Consensus Metric" has a desired effect of providing a more optimistic evaluation.  \autoref{table:aconsensus_daquar} shows the application of both measures to our data and models.

Moreover,  \autoref{table:consensus_human_baseline} shows that "MCM" applied to human answers at test time captures ambiguities in interpreting questions by improving the score of the human baseline from (["A multi-world approach to question answering about real-world scenes based on uncertain input"]()) (here, as opposed to \autoref{table:aconsensus_daquar}, we exclude the original human answers from the measure). It  cooperates well with WUPS at $0.9$, which takes word ambiguities into account, gaining  an $18\%$ higher score.%

## Qualitative results
We show predicted answers of different architecture variants in Tables \ref{fig:vision_vs_language}, \ref{fig:multiple_answers}, and \ref{fig:mix_predictions}.
We chose the examples to highlight differences between \AproachName and the "Question-only".

We use a "multiple words" approach only in \autoref{fig:multiple_answers}, otherwise the "single word" model is shown. Despite some failure cases, "Question-only" makes "reasonable guesses" like predicting that the largest object could be table or an object that could be found on the bed is a pillow or doll.

## Failure cases

While our method answers correctly on a large part of the challenge (e.g. $\approx 35 $ WUPS at $0.9$ on "what color" and "how many" question subsets),  spatial relations ($\approx 21$ WUPS at $0.9$) which account for a substantial part of DAQUAR remain challenging.  Other errors involve questions with small objects, negations, and shapes (below $12$ WUPS at $0.9$). Too few training data points for the aforementioned cases may contribute to these mistakes.

\autoref{fig:mix_predictions} shows examples of failure cases that include (in order) strong occlusion, a possible answer not captured by our ground truth answers, and unusual instances (red toaster).

# Analysis on VQA

While \autoref{sec:results} analyses our original architecture (["Ask your neurons: A neural-based approach to answering questions about images"]()) on the DAQUAR dataset, in this section, we analyze different variants and design choices for neural question answering on the large-scale VQA (VQA) dataset (["Vqa:VQA"]()). It is currently one of the largest and most popular VQA dataset with human question answer pairs.
In the following, after describing the experimental setup (\autoref{sec:vqa:setup}), we first describe several experiments which examine the different variants of question encoding, only looking at language input to predict the answer (\autoref{sec:vqa:setup}), and then, we examine the full model (\autoref{sec:vqa:vision_language}).

## Experimental setup

We evaluate on the VQA dataset (["Vqa:VQA"]()), which is built on top of the MS-COCO dataset (["Microsoft coco: Common objects in context"]()). Although VQA offers a different challenge tasks, we focus our efforts on the Real Open-Ended VQA challenge. The challenge consists of $10$ answers per question with about $248k$  training questions, about $122k$  validation questions, and about $244k$  test questions.

As VQA consist mostly of single word answers (over 89\%), we treat the question answering problem as a classification problem of the most frequent answers in the training set. For the evaluation of the different model variants and design choices, we train on the training set and test on the validation set. Only the final evaluations (\autoref{table:vqa_test}) are evaluated on the test set of the VQA challenge, we evaluate on both parts test-dev and test-standard, where for the latter the answers are not publicly available. As a performance measure we use a Consensus variant of Accuracy introduced in (["Vqa:VQA"]()) where the predicted answer gets score between $0$ and $1$, with $1$ if it matches with at least three human answers.

We use ADAM (["Adam: A method for stochastic optimization"]()) throughout our experiments as we found out it performs better than SGD with momentum.

We keep default hyper-parameters for ADAM. Employed RNNs maps input question into $500$ dimensional vector representation.
All the CNNs for text are using $500$ feature maps in our experiments, but the output dimensionality also depends on the number of views.
In preliminary experiments we found that removing question mark '?' in the questions slightly improves the results, and we report the numbers only with this setting. Since VQA has $10$ answers associated with each question, we need to consider a suitable training strategy that takes this into account. We have examined the following strategies: picking an answer  randomly,  randomly but if possible annotated as confidently answered, all answers, or choosing the most frequent answer. In the following, we only report the results using the last strategy as we have found out little difference in accuracy between the strategies.
To allow training and evaluating many different models with limited time and computational power, we %
do not fine-tune the visual representations in these experiments, although our model would allow us to do so. All the models, which are publicly available under \url{https://github.com/mateuszmalinowski/Kraino}, are implemented in Keras (["Chollet F (2015) keras"]()) and Theano (["Bastien-Theano-2012"]()).


## CNN questions encoder

We first examine different hyper-parameters for CNNs to encode the question. We first  consider the filter's length of the convolutional kernel. We run the model over different kernel lengths ranging from $1$ to $4$ (\autoref{table:vqa_cnn_filter_length}, left column). We notice that increasing the kernel lengths improves performance up to length $3$ were the performance levels out, we thus use kernel length $3$ in the following experiments for, such CNN can be interpreted as a trigram model.

We also tried to run simultaneously a few kernels with different lengths. In \autoref{table:vqa_cnn_filter_length} (right column) one view corresponds to a kernel length $1$, two views correspond to two kernels with length $1$ and $2$, three views correspond to length $1$, $2$ and $3$, etc. However, we find that the best performance still achieve with a single view and kernel length $3$ or $4$.

## BOW questions encoder
Alternatively to neural network encoders, we consider Bag-Of-Words (BOW) approach where one-hot representations of the question words are first mapped to a shared embedding space, and subsequently summed over. ie $\Psi(\text{question}) := \sum_{\text{word}} W_e(word)$. Surprisingly, such a simple approach gives very competitive results (first row in \autoref{table:question_encoders}) compared to the CNN encoding discussed in the previous section  (second row). %

## Recurrent questions encoder

We examine two recurrent questions encoders, LSTM (["Long short-term memory"]()) and a simpler GRU (["Learning phrase representations using rnn encoder-decoder for statistical machine translation"]()).
The last two rows of \autoref{table:question_encoders} show a slight advantage of using LSTM.%

## Pre-trained words embedding
In all the previous experiments, we jointly learn the embedding transformation $\bs{W}_e$ together with the whole architecture only on the VQA dataset. This means we do not have any means for dealing with unknown words in questions at test time apart from using a special token $\left<\text{UNK}\right>$ to indicate such class.
To address such shortcoming, we investigate the pre-trained word embedding transformation GLOVE (["Glove: Global vectors for word representation"]()) that encodes question words (technically it maps one-hot vector into a $300$ dimensional real vector). This choice naturally extends the vocabulary of the question words to about $2$ million words extracted a large corpus of web data -- Common Crawl (["Glove: Global vectors for word representation"]()) --that is used to train the GLOVE embedding.

Since the BOW architecture in this scenario becomes shallow (only classification weights are learnt),  we add an extra hidden layer between pooling and classification (without this embedding, accuracy drops by $5\%$). %

\autoref{table:question_encoders} (right column) summarizes our experiments with GLOVE. For all question encoders, the word embedding consistently improves performance which confirms that using a word embedding model learnt from a larger corpus helps. LSTM benefits most from GLOVE embedding, archiving the overall best performance with 48.58\% accuracy.


## Top most frequent answers
Our experiments reported in \autoref{table:top_frequent_words} investigate predictions using different number of answer classes. We experiment with a truncation of $1000$, $2000$, or $4000$ most frequent classes. For all question encoders (and always using GLOVE word embedding), we find that a truncation at $2000$ words is best, being apparently a good compromise between answer frequency and missing recall.

## Summary Question-only
We achieve the best "Question-only" accuracy with GLOVE word embedding, LSTM sentence encoding, and using the top $2000$ most frequent answers. This achieves an performance of 48.86\% accuracy.  In the remaining experiments, we use these settings for language and answer encoding.

## Vision and Language

Although Question-only models can answer on a substantial number of questions as they arguably capture common sense knowledge, for further development we also need images.

## Multimodal fusion
\autoref{table:multimodal_fusion} investigates different techniques that combine visual and language representations. To speed up training, we combine the last unit of the question encoder with the visual encoder, as it is explicitly shown in \autoref{fig:vqa_encoder_decoder}. In the experiments we use Concatenation, Summation, and Piece-wise multiplication on the BOW language encoder with GLOVE word embedding and features extracted from the VGG-19 net. In addition, we also investigate using L2 normalization of the visual features, which divides every feature vector by its L2 norm. The experiments show that the normalization is crucial in obtaining good performance, especially for Concatenation and Summation. In the remaining experiments, we use Summation.

## Questions encoders
\autoref{table:multimodal_methods} shows how well different questions encoders combine with the visual features. We can see that LSTM slightly outperforms two other encoders GRU and CNN, while BOW remains the worst, confirming our findings in our language-only experiments with GLOVE and 2000 answers (\autoref{table:top_frequent_words}, second column). %

## Visual encoders
Next we fix the question encoder to LSTM and vary different visual encoders: Caffe variant of AlexNet} (["Imagenet classification with deep CNNs"]()), GoogLeNet} (["Going deeper with convolutions"]()), VGG-19} (["simonyan2014very}, and recently introduced 152 layered ResNet} (we use the Facebook implementation of (["Deep residual learning for image recognition"]())). \autoref{table:visual_encoders} confirms our hypothesis that stronger visual models perform better.

## Qualitative results
We show predicted answers using our best model on VQA test set in Tables \ref{fig:vqa-image_qa-yes_no}, \ref{fig:vqa-image_qa-counting} ,\ref{fig:vqa-image_qa-what}, \ref{fig:vqa-image_qa-compound}. We show chosen examples with 'yes/no', 'counting', and 'what'  questions, where our model, according to our opinion, makes valid predictions. Moreover, \autoref{fig:vqa-image_qa-compound} shows predicted compound answers.

## Summary VQA results
\autoref{table:vqa_summary} summarises our findings on the validation set. We can see that on one hand methods that use contextual language information such as CNN and LSTM are performing better, on the other hand adding strong vision becomes crucial.

Furthermore, we use the best found models to run experiments on the VQA test sets: test-dev2015 and test-standard. To prevent overfitting, the latter restricts the number of submissions to 1 per day and 5 submissions in total. Here, we also study the effect of larger datasets where first we train only on the training set, and next we train for $20$ epochs on a joint, training and validation, set. When we train on the join set, we consider question answer pairs with answers among $2000$ the most frequent answer classes from the training and validation sets.  Training on the joint set have gained us about $0.9\%$.  This implies that on one hand having more data indeed helps, but arguably we also need better models that exploit the current training datasets more effectively. Our findings are summarized in \autoref{table:vqa_test}.
