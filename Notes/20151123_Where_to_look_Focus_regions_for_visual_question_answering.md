## [Where to look: Focus regions for visual question answering](http://arxiv.org/abs/1511.07394)


# abstract

We present a method that learns to answer visual questions by
selecting image regions relevant to the text-based query. Our method
maps textual queries and visual features from various regions into a shared space where they are
compared for relevance with an inner product. Our method
exhibits significant improvements in answering questions such as "what
color," where it is necessary to evaluate a specific location,
and "what room," where it selectively identifies informative image
regions. Our model is tested on the recently released VQA (["VQA"]())
dataset, which features free-form human-annotated questions and
answers.

# Introduction

## Intuition: why do you think this approach will work?

Visual question answering (VQA) is the task of answering a natural language
question about an image. VQA includes many challenges
in language representation and grounding, recognition, common sense
reasoning, and specialized tasks like counting and reading.  In this
paper, we focus on a key problem for VQA and other visual reasoning
tasks: knowing where to look. Consider Figure.

<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/175e9bb50cc062c6c1742a5d90c8dfe31d2e4e22/0-Figure1-1.png" width="500" ></p>
<font size="2">Figure 1. Our goal is to identify the correct answer for a natural language question, such as “What color is the walk light?” or “Is it raining?” We particularly focus on the problem of learning where to look. This is a challenging problem as it requires grounding language with vision and learning to recognize objects, use relations, and determine relevance. For example, whether it is raining may be determined by detecting the presence of puddles gray skies, or umbrellas in the scene, whereas the color of the walk light requires focused attention on the light alone. The above figure shows example attention regions produced by our proposed model.</font>


It's easy to answer "What color is the walk light?" if the light bulb is localized,
while answering whether it's raining may be dealt with by identifying
umbrellas, puddles, or cloudy skies. We want to learn where to look to
answer questions supervised by only images and question/answer pairs.
For example, if we have several training examples for "What time of
day is it?" or similar questions, the system should learn what kind
of answer is expected and where in the image it should base its
response.


Learning where to look from question-image pairs has many
challenges. Questions such as "What sport is this?" might be
best answered using the full image.  Other questions such as "What
is on the sofa?" or "What color is the woman's shirt?" require
focusing on particular regions.  Still others such as "What does the
sign say?" or "Are the man and woman dating?" require specialized
knowledge or reasoning that we do not expect to achieve. The system
needs to learn to recognize objects, infer spatial
relations, determine relevance, and find correspondence between
natural language and visual features.  Our key idea is to learn a
non-linear mapping of language and visual region features into a
common latent space to determine relevance. The relevant regions are
then used to score a specific question-answer pairing. The latent embedding and
the scoring function are learned jointly using a margin-based loss
supervised solely by question-answer pairings. We perform
experiments on the VQA dataset (["VQA"]()) because it features
open-ended language, with a wide variety of questions.  Specifically, we focus on its
multiple-choice format because its evaluation is much less ambiguous
than open-ended answer verification.

We focus on learning where to look but also provide useful baselines and analysis for the task as a whole.  Our contributions are as follows:

- We present an image-region selection mechanism that learns to identify image regions relevant to questions.
- We present a learning framework for solving multiple-choice visual QA with a margin-based loss that significantly outperforms provided baselines from (["VQA"]()).
- We compare with baselines that answer questions without the image, use the whole image, and use all image regions with uniform weighting, providing a detailed analysis for when selective regions improve VQA performance.

# Related Works
Many recent works in tying text to images have explored the task of automated image captioning (["Improving image-sentence embeddings using large weakly annotated photo collections"],["Show, attend and tell: Neural image caption generation with visual attention"](),["Deep visual-semantic alignments for generating image descriptions"](),["Deep captioning with multimodal recurrent neural networks"](),["Explain images with multimodal recurrent neural networks"](),["Long-term recurrent convolutional networks for visual recognition and description"](),["Mind's eye: A recurrent visual representation for image caption generation"](),["Show and tell: A neural image caption generator"]()). While VQA can be considered as a type of directed
captioning task, our work relates to some (["Show, attend and tell: Neural image caption generation with visual attention"](), ["From captions to visual concepts and back"]()) in that we learn to employ an attention mechanism for region focus, though our formulation makes determining region relevance a more explicit part of the learning process.

In Fang et al. (["From captions to visual concepts and bac"]()), words
are detected in various portions of the image and combined together
with a language model to generate captions. Similarly, Xu et al. (["Show, attend and tell: Neural image caption generation with visual attention"]())
uses a recurrent network model to detect salient objects and generate
caption words one by one.  Our model works in
the opposite direction of these caption models at test time by determining the relevant image
region given a textual query as input. This allows our model to
determine whether a question-answer pair is a good match given evidence from
the image.

Partly due to the difficulty of evaluating image captioning, several
visual question answering datasets have been proposed along with
applied approaches.  We choose to experiment on VQA (["VQA"]()) due to
the open ended nature of its question and answer
annotations. Questions are collected by asking annotators to pose a difficult question for a smart robot, and multiple
answers are collected for each question.  We experiment on the
multiple-choice setting as its evaluation is less ambiguous than that
of open-ended response evaluation.  Most other visual question answering
datasets (["Exploring models and data for image question answering,Visual madlibs: Fill in the blank image generation and question answering"]()) are based on reformulating
existing object annotations into questions, which provides an
interesting visual task but limits the scope of visual and abstract
knowledge required.

Our model is inspired by End-to-End Memory Networks (["Weakly supervised memory networks"]()) proposed
for answering questions based on a series of sentences.  The regions
in our model are analogous to the sentences in theirs, and, similarly
to them, we learn an embedding to project question and potential
features into a shared subspace to determine relevance with an inner product.  Our method
differs in many details such as the language model and more broadly in
that we are answering questions based on an image, rather than a text
document. Ba et al. (["Predicting deep zero-shot convolutional neural networks using textual  descriptions"]()) also uses a similar architecture, but in a zero-shot
learning framework to predict classifiers for novel categories. They project language and vision features into a shared subspace to perform similarity computations with inner products like us, though the score is used to guide the generation of object classifiers rather than to rank image regions.

Existing approaches in VQA tend to use recurrent
networks to model language and predict
answers (["Exploring models and data for image question answering,VQA,Visual madlibs: Fill in the blank image generation and question answering,Ask your neurons: A neural-based approach to answering questions about images"]()), though simpler Bag-Of-Words (BOW) and averaging models have been
shown to perform roughly as well if not better than sequence-based
LSTM (["Exploring models and data for image question answering, VQA"]()). Yu et al. (["Visual madlibs: Fill in the blank image generation and question answering"]()), which
proposes a Visual Madlibs dataset for fill-in-the-blank and question
answering, focuses their approach on learning latent embeddings and
finds normalized CCA on averaged word2vec representations (["Improving image-sentence embeddings using large weakly annotated photo collections,Efficient estimation of word representations in vector space"]())  to outperform recurrent networks for embedding. Similarly in our work, we find a fixed-length averaged representation of word2vec vectors for language to be
highly effective and much simpler to train, and our approach differs at a high level in our
focus on learning where to look.


# Approach


<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/175e9bb50cc062c6c1742a5d90c8dfe31d2e4e22/2-Figure3-1.png" width="800" ></p>
Table 2. Accuracy comparison on Test-dev (top) and Test-standard (bottom). Our model outperforms the best performing image and text models from [1].

Our method learns to embed the textual question and the
set of visual image regions into a latent space where the inner product
yields a relevance weighting for each region.

The input is a question,
potential answer, and image features from a set of automatically selected candidate regions.  We encode the parsed question and answer using word2vec (["Efficient estimation of word representations in vector space"]()) and a two-layer network.  Visual features for each region are encoded using the top two layers (including the output layer) of a CNN trained on ImageNet (["ImageNet Large Scale Visual Recognition Challenge"]()).  The language and vision features are then embedded and compared with a dot product, which is soft-maxed to produce a per-region relevance weighting.  Using these weights, a weighted average of concatenated vision and language features is the input to a 2-layer network that outputs a score for whether the answer is correct.


## QA Objective}
Our model is trained for the multiple choice task of the VQA
dateset. For a given question and its corresponding choices, the
objective of our network aims to maximize a margin between correct and
incorrect choices in a structured-learning fashion.  We achieve this
by using a hinge loss over predicted confidences $y$.

In our setting, multiple answers could be acceptable to varying
degrees, as correctness is determined by the consensus of 10
annotators.  For example, most may say that the color of a scarf is
"blue" while a few others say "purple".  To take this into
account, we scale the margin by the gap in number of annotators returning the specific answer:

$$
  {\mathcal L}(y) = \max_{\forall n \ne p} (0,y_n + (a_p - a_n) - y_p).
$$

The above objective requires that the score of the correct answer ($y_p$) is at
least some margin above the score of the highest-scoring incorrect answer
($y_n$) selected from among the set of incorrect choices ($n \ne p$).
For example, if 6/10 of the annotators answer $p$ ($a_p = 0.6$) and
2 annotators answer $n$ ($a_n = 0.2$), then $y_p$ should outscore $y_n$ by a margin of at least 0.4.

## Region Selection Layer

Our region selection layer selectively combines
incoming text features with image features from relevant regions of
the image.  To determine relevance, the layer first projects the image
features and the text features into a shared
N-dimensional space, after which an inner product is computed for each
question-answer pair and all available regions.


Let $G_r$ be the projection of all region features in column vectors
of $X_r$, and $\vec{g}_l$ be the projection of a single embedded question-answer
pair. The feedforward pass to compute the relevance weightings is
computed as follows:
$$
\begin{align}
  G_r =& AX_r+\vec{b}_r\\
  \vec{g}_l =& B\vec{x}_l + \vec{b}_l\\
  \vec{s}_{l,r} =& \sigma(G_r^T\vec{g}_l)\\
  \sigma(\vec{z}) = &\frac{e^{z_j}}{\sum_{k=1}^{K}e^{z_k}}\mbox{ for }
  j = 1,...K
\end{align}
$$
Here, the output $\vec{s}_{l,r}$ is the softmax normalized weighting
($\sigma$) of the inner products of $\vec{g}_l$ with each projected
region feature in $G_r$. Vectors $\vec{b}$ represent biases. The
purpose of the inner product is to force the model to determine region
relevance in a vector similarity fashion.

Using 100 regions per image, this gives us 100 region weights for a question-answer pair. Next, the text features are concatenated directly with image features for each region to produce
100 different feature vectors.  This is shown in the horizontal
stacking of $X_r$ and repetitions of $\vec{x}_l$ below. Each feature vector is linearly projected with $W$, and the weighted average is computed using $\vec{s}_r$ to attain feature vector $\vec{a}_l$ for each question and answer pair, which is then fed through relu and batch-normalization layers.
$$
\begin{align}
  P_{l,r} =& W\left[ {\begin{array}{c} X_r\\\begin{array}{ccc}- &\vec{x}_l &-\end{array} \end{array}} \right]+\vec{b}_o\\
  \vec{a}_l =& P\vec{s}_{l,r}
\end{align}
$$
We also tried learning to predict a relevance score directly from concatenated vision and language features, rather than computing the dot product of the features in a latent embedded space.  However,
the resulting model appeared to learn a salient region weighting
scheme that varied little with the language component.  The
inner-product based relevance was the only formulation we tried that successfully takes account of both the query and the region information.


## Language Representation

We represent our words with 300-dimensional word2vec
vectors (["Efficient estimation of word representations in vector space"]()) for their simplicity
and compact representation.  We are also motivated by the ability of vector-based language representations to encode similar words with similar vectors, which may aid answering open-ended questions.
Using averages across word2vec vectors, we construct fixed-length
vectors for each question-answer pair, which our model then learns to
score. In our results section, we show that our vector-averaging language
model noticeably outperforms a more complex LSTM-based model from
 (["VQA"]()), demonstrating that BOW-like models provide very effective
and simple language representations for VQA tasks.

We first tried separately averaging vectors for each word with the
question and answer, concatenating them to yield a 600-dimensional
vector, but since the word2vec representation is not sparse, averaging
several words may muddle the representation.  We improve the
representation using the Stanford Parser (["Generating typed dependency parses from phrase structure parse"]()) to bin the question into
additional separate semantic bins. The bins are defined as follows:


$\mathbf{Bin 1}$ captures the type of question by averaging the word2vec representation of the first two words.  For example, "How many" tends to require a numerical answer,  while "Is there" requires a yes or no answer.

$\mathbf{Bin 2}$ contains the nominal subject to encode subject of question.

$\mathbf{Bin 3}$ contains the average of all other noun words.

$\mathbf{Bin 4}$ contains the average of all remaining words, excluding determiners such as "a," "the," and "few."

Each bin then contains a 300-dimensional representation, which are
concatenated with a bin for the words in the candidate answer to yield
a 1500-dimensional question/answer representation.   This representation separates out important components of a variable-length question while maintaining a fixed-length representation that simplifies the network architecture.


<p align="center"><img src="https://ai2-s2-public.s3.amazonaws.com/figures/2016-03-25/175e9bb50cc062c6c1742a5d90c8dfe31d2e4e22/3-Figure4-1.png" width="400" ></p>
Figure 4. Example parse-based binning of questions. Each bin is represented with the average of the word2vec vectors of its members. Empty bins are represented with a zero-vector.

## Image Features}

The image features are fed directly into the region-selection
layer from a pre-trained network.  We first select candidate regions
by extracting the top-ranked 99 Edge Boxes (["Edge boxes: Locating object proposals from edge"]()) from
the image after performing non-max suppression with a 0.2 intersection
over union overlap criterion. We found this aggressive non-max
suppression to be important for selecting smaller regions that may be
important for some questions, as the top-ranked regions tend to be
highly overlapping large regions. Finally, a whole-image region is also added to
ensure that the model at least has the spatial support of the full
frame if necessary, bringing the total number of candidate regions to
100 per image. While we have not experimented with the number of
regions, it is possible that the improved recall from additional regions
may improve performance.

We extract features using the VGG-s network (["Return of the devil in the details: Delving deep into convolutional nets"]()), concatenating the output
from the last fully connected layer (4096 dimensions) and the
pre-softmax layer (1000 dimensions) to get a 5096 dimensional feature
per region.  The pre-softmax classification layer was included to
provide a more direct signal for objects from the Imagenet (["ILSVRC15"]()) classification task.


## Training}

Our overall network architecture is implemented in
MatConvNet(["Matconvnet -- convolutional neural networks for matlab"]()). Our fully connected layers are
initialized with Xavier initialization ($\frac{1}{\sqrt{n_{in}}}$) (["Understanding the difficulty of training deep feedforward neural networks"]()) and
separated with a batch-normalization (["Batch normalization: Accelerating deep network training by reducing internal covariate shift"]()) and relu
  layer (["Deep sparse rectifier neural networks"]()) between each. The word2vec text features
  are fed into the network's input layer, whereas the image region features
feed in through the region selection layer.

Our network sizes are set as follows. The 1500 dimensional language
features first pass through 3 fully connected layers with output
dimensions 2048, 1500, and 1024 respectively. The embedded language features are then passed through the region selection layer to be combined with the vision
features. Inside the region selection layer, projections $A$ and $B$
project both vision and language representations down to 900
dimensions before computing their inner product. The exiting feature
representation passes through $W$ with an output dimension of
2048. then finally through two more fully connected layers with output
dimensions of 900 and 1 where the output scalar is the
question-answer pair score.

It is necessary to pay extra attention to the initialization of the
region-selection layer. The magnitude of the projection matrices $A$,
$B$ and $W$ are initialized to $0.001$ times the standard normal
distribution.  We found that low initial values were important to
prevent the softmax in selection from spiking too early and to prevent
the higher-dimensional vision component from dominating early in the training.

# Experiments}
We evaluate the effects of our region-selection layer on the
multiple-choice format of the MS COCO Visual Question Answering (VQA)
dataset (["VQA"]()). This dataset contains 82,783 images for training,
40,504 for validation, and 81,434 for testing.  Each image has 3
corresponding questions with recorded free-response answers from
10 annotators. Any response that comes from at least 3 annotators is
considered correct. We use the 18-way multiple choice task because its
evaluation is much less ambiguous than the open-ended response task,
though our method could be applied to the latter by treating the most
common or likely K responses as a large K-way multiple
choice task.  We trained using only the training set, with 10\% set
aside for model selection and parameter tuning.  We perform detailed
evaluation on the validation set and further comparison on the test
set using the provided submission tools.

We evaluate and analyze how much our region-weighting improves accuracy  compared to using the whole image or only language (Tables~\ref{tbl:Overall_acc_val},~\ref{tbl:Overall_acc_testdev},~\ref{tbl:breakdown}) and show examples in Figure~\ref{fig:vqa_qual_comp}.  We also perform a simple evaluation on a subset of images showing that relevant regions tend to have higher than average weights (Fig.~\ref{fig:weight_anno}).  We also show the advantage of our language model over other schemes (Table~\ref{tbl:language}).

## Comparisons between region, image, and language-only models}

We compare our region selection model with several baseline methods, described below.

$\mathbf{Language-only}$ We train a network to score each answer purely from the language representation.  This provides a baseline to demonstrate improvement due to image features, rather than just good guesses.

$\mathbf{Word+Whole image}$ We concatenate CNN features computed over the entire image with the language features and score them using a 3-layer neural network, essentially replacing the region-selection layer with features computed over the whole image.

$\mathbf{Word+Uniform averaged region features}$ To test that region weighting is important, we also try uniformly averaging features across all regions as the image representation and train as above.

Our proposed region-selection model outperforms all other models.  Also, we can see that uniform weighting of regions is not helpful.  We also include the
best-performing LSTM question+image model from the authors of the VQA
dataset (["VQA"]()). This model significantly underperforms even our much simpler baselines, which could be partly because the model was designed for open-ended answering and adapted for multiple choice.
