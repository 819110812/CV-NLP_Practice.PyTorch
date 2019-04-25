## [Ask me anything: Free-form visual question answering based on knowledge from external sources](https://arxiv.org/abs/1511.06973)


# Introduction

Visual question answering (VQA) is distinct from many problems in Computer Vision because the question to be answered is not determined until run time (["VQA: Visual Question Answering"]()).  In more traditional problems such as segmentation or detection, the single question to be answered by an algorithm is predetermined, and only the image changes.  In visual question answering, in contrast, the form that the question will take is unknown, as is the set of operations required to answer it.  In this sense it more closely reflects the challenge of general image interpretation.

VQA typically requires processing both visual information (the image) and textual information (the question and answer).  One approach to Vision-to-Language problems, such as VQA and image captioning, which interrelate visual and textual information is based on a direct method pioneered in machine language translation (["Learning phrase representations using rnn encoder-decoder for  statistical machine translation"](),["Sequence to sequence learning with neural networks"]()). This direct approach develops an encoding of the input text using a Recurrent Neural Network (RNN) and passes it to another RNN for decoding. %


The Vision-to-Language version of this direct translation approach uses a Convolutional Neural Network (CNN) to encode the image, and an RNN to encode and generate text.

This approach has been well studied.  The image captioning methods developed in~(["donahue2014long,karpathy2014deep,mao2014deep,vinyals2014show,yao2015describing}, for example, learn a mapping from images to text

using CNNs trained on object recognition, and word embeddings trained on large scale text corpora.

Visual question answering is a significantly more complex problem than image captioning, not least because it requires accessing information not present in the image.  This may be common sense, or specific knowledge about the image subject.

For example, given an image, showing "a group of people enjoying a sunny day at the beach with umbrellas", if one asks a question "why do they have umbrellas?", to answer this question, the machine must not only detect the scene "beach", but must know that "umbrellas are often used as points of shade on a sunny beach". Recently, Antol et.al ~(["VQA: Visual Question Answering"]()) also have suggested that VQA is a more ""AI-complete" task since it requires multimodal knowledge beyond a single sub-domain. Our proposed system finally gives the right answer "shade' for the above real example.

Large-scale Knowledge Bases (KBs), such as Freebase~(["Freebase: a collaboratively created graph database for structuring human knowledge"]()) and DBpedia(["Dbpedia: A nucleus for a web of open data"]()), have been used successfully in several natural language Question Answering (QA) systems~(["Semantic Parsing on Freebase from Question-Answer Pairs"](),["Building Watson: An overview of the DeepQA project"]()}.

However, VQA systems exploiting KBs are still relatively rare.

Gao et.al (["Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering"]()) and Malinowski et.al (["Ask Your Neurons: A Neural-based Approach to Answering Questions about Images"]()) do not use a KB at all.

Zhu et.al.(["Building a Large-scale Multimodal Knowledge Base for Visual Question Answering"]()) do use a KB, but it is created specifically for the purpose, and consequently contains a small amount of very specific, largely image-focused, information.  This in turn means that only a specific set of questions may be asked of the system, and the query generation approach taken mandates a very particular question form.  The method that we propose here, in contrast, is applicable to general (even publicly created)  KBs, and admits general questions.


In this work, we fuse an automatically generated description of an image with information extracted from an external KB to provide an answer to a general question about the image. The image description takes the form of a set of captions, and the external knowledge is text-based information mined from a Knowledge Base. Given an image-question pair, a CNN is first employed to predict a set of attributes of the image. The attributes cover a wide range of high-level concepts, including objects, scenes, actions, modifiers and so on.  A state-of-the-art image captioning model (["What Value Do Explicit High Level Concepts Have in Vision to Language Problems?"]()) is applied to generate a series of captions based on the attributes.

We then use the detected attributes to extract relevant information from the KB.  Specifically, for each of the top-5 attributes detected in the image we generate a query which may be applied to a Resource Description Framework (RDF) KB, such as DBpedia.  RDF is the standard format for large KBs, of which there are many.  The queries are specified using Semantic Protocol And RDF Query Language (SPARQL).

We encode the paragraphs extracted from the KB using Doc2Vec (["Distributed representations of sentences and documents"]()), which maps paragraphs into a fixed-length feature representation.

The encoded attributes, captions, and KB information are then input to an LSTM which is trained so as to maximise the likelihood of the ground truth answers in a training set.

The approach we propose here combines the generality of information that using a KB allows with the generality of question that the LSTM allows.

In addition, it achieves an accuracy of 69.73\% on the Toronto COCO-QA, while the latest state-of-the-art is 55.92\%. %
We also produce the best results on the VQA evaluation server (which does not publish ground truth answers for its test set), which is 59.44\%.

# Related Work

Malinowski et.al.(["A multi-world approach to question answering about real-world scenes based on uncertain input"]() were among the first to study the VQA problem. They proposed a method that combines semantic parsing and image segmentation with a Bayesian approach to sample from nearest neighbors in the training set. This approach requires human defined predicates, which are inevitably dataset-specific. This approach is also very dependent on the accuracy of the image segmentation algorithm and on the estimated image depth information. Tu et.al (["Joint video and text parsing for understanding events and answering queries"]() built a query answering system based on a joint parse graph from text and videos. Geman et.al (["Visual Turing test for computer vision systems"]()) proposed an automatic "query generator' that was trained on annotated images and produced a sequence of binary questions from any given test image.

Each of these approaches places significant limitations on the form of question that can be answered.

Most recently, inspired by the significant progress achieved using deep neural network models in both computer vision and natural language processing, an architecture which combines a CNN and RNN to learn the mapping from images to sentences has become the dominant trend. Both Gao et.al (["Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering"]()) and Malinowski et.al.(["Ask Your Neurons: A Neural-based Approach to Answering Questions about Images"]()) used RNNs to encode the question and output the answer.  Whereas Gao et.al ~(["Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering"]()) used two networks, a separate encoder and decoder, Malinowski et.al.(["Ask Your Neurons: A Neural-based Approach to Answering Questions about Images"]()) used a single network for both encoding and decoding. Ren et.al.(["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()) focused on questions with a single-word answer and formulated the task as a classification problem using an LSTM. A single-word answer dataset COCO-QA was published with (["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()). Ma et.al.(["Learning to Answer Questions From Image using Convolutional Neural Network"]()) used CNNs to both extract image features and sentence features, and fused the features together with another multimodal CNN. Antol et.al.(["VQA: Visual Question Answering"]()) proposed a large-scale open-ended VQA dataset based on COCO, which is called VQA. They also provided several baseline methods which combined both image features (CNN extracted) and question features (LSTM extracted) to obtain a single embedding and further built a MLP (Multi-Layer Perceptron) to obtain a distribution over answers. Our framework also exploits both CNNs and RNNs, but in contrast to preceding approaches which use only image features extracted from a CNN in answering a question, we employ multiple sources, including image content, generated image captions and mined external knowledge, to feed to an RNN to answer questions.

The quality of the information in the KB is one of the primary issues in this approach to VQA. The problem is that KBs constructed by analysing Wikipedia and similar are patchy and inconsistent at best, and hand-curated KBs are inevitably very topic specific. Using visually-sourced information is a promising approach to solving this problem (["Don't Just Listen, Use Your Imagination: Leveraging Visual Common   Sense for Non-Visual Tasks"]()), ["VisKE: Visual Knowledge Extraction and Question Answering by Visual Verification of Relation Phrases"]()), but has a way to go before it might be usefully applied within our approach.  Thus, although our SPARQL and RDF driven approach can incorporate any information that might be extracted from a KB, the limitations of the existing available KBs mean that the text descriptions of the detected attributes is all that can be usefully extracted.

Zhu et.al.(["Building a Large-scale Multimodal Knowledge Base for Visual Question Answering"]()), in contrast used a hand-crafted KB primarily containing image-related information such as category labels, attribute labels and affordance labels, but also some quantities relating to their specific question format such as GPS coordinates and similar.

The questions in that system are phrased in the DBMS query language, and are thus tightly coupled to the nature of the hand-crafted KB.  This represents a significant restriction on the form of question that might be asked, but has the significant advantage that the DBMS is able to respond decisively as to whether it has the information required to answer the question. Instead of building a problem-specific KB, we use a pre-built large-scale KB (DBpedia (["Dbpedia: A nucleus for a web of open data"]()) from which we extract information using a standard RDF query language. DBpedia has been created by extracting structured information from Wikipedia, and is thus significantly larger and more general than a hand-crafted KB. Rather than having a user to pose their question in a formal query language, our VQA system is able to encode questions written in natural language automatically.  This is achieved without manually specified formalization, but rather depends on processing a suitable training set. The result is a model which is very general in the forms of question that it will accept.

# Extracting, Encoding, and Merging
## Feature_Extract

The key differentiator of our approach is that it is able to usefully combine image information with that extracted from a KB, within the LSTM framework.  The novelty lies in the fact that this is achieved by representing both of these disparate forms of information as text before combining them.

We now describe each step in more details.

## Attribute-based Image Representation
Our first task is to describe the image content in terms of a set of attributes.
Each attribute in our vocabulary is extracted from captions from MS COCO (["Microsoft COCO captions: Data collection and evaluation server"]()), a large-scale image-captioning dataset. An attribute can be any part of speech, including object names (nouns), motions (verbs) or properties (adjectives). Because MS COCO captions are created by humans, attributes derived from captions are likely to represent image features that are of particular significance to humans, and are thus likely to be the subject of image-based questions.

We formalize attribute prediction as a multi-label classification problem. To address the issue that some attributes may only apply to image sub-regions, we follow Wei et.al ~(["CNN: Single-label to multi-label"]()) to design a region-based multi-label classification framework that takes an arbitrary number of sub-region proposals as input. A shared CNN is connected with each proposal, and the CNN outputs from different proposals are aggregated with max pooling to produce the final prediction over the attribute vocabulary.

To initialize the attribute prediction model, we use the powerful VggNet-16 (["Very deep convolutional networks for large-scale image recognition"]()) pre-trained on the ImageNet (["Imagenet: A large-scale hierarchical image database"]()). The shared CNN is then fine-tuned on the multi-label dataset, the MS COCO image-attribute training data (["What Value Do Explicit High Level Concepts Have in Vision to Language Problems?"]()). The output of the last fully-connected layer is fed into a $c$-way softmax which produces a probability distribution over the $c$ class labels. The $c$ represents the attribute vocabulary size, which here is 256. The fine-tuning learning rates of the last two fully connect layers are initialized to 0.001 and the prediction layer is initialized to 0.01. All the other layers are fixed. We execute 40 epochs in total and decrease the learning rate to one tenth of the current rate for each layer after 10 epochs. The momentum is set to 0.9. The dropout rate is set to 0.5. Then, we use Multiscale Combinatorial Grouping (MCG)~(["Multiscale Combinatorial Grouping for Image Segmentation and Object Proposal Generation"]())} for the proposal generation.

Finally, a cross hypothesis max-pooling is applied to integrate the outputs into a single prediction vector $\text{Att}(I)$.

## Caption-based Image Representation
### Story

Currently the most successful approach to image captioning (["Learning a Recurrent Visual Representation for Image Caption Generation"](),["Long-term recurrent convolutional networks for visual recognition  and description"](),["Deep fragment embeddings for bidirectional image sentence mapping"](),["Deep Captioning with Multimodal Recurrent Neural Networks"](),["Show and tell: A neural image caption generator"]()) is to attach a CNN to an RNN to learn the mapping from images to sentences directly. Wu et.al (["What Value Do Explicit High Level Concepts Have in Vision to Language Problems?"]()) proposed to feed a high-level attribute-based representation to an LSTM to generate captions, instead of directly using  CNN-extracted features. This method produces promising results on the major public captioning challenge~(["chen2015microsoft} and accepts our attributes prediction vector $\text{Att}(I)$ as the input. We thus use this approach to generate 5 different captions (using beam search) that constitute the internal textual representation for a given image.

The hidden state vector of the caption-LSTM after it has generated the last word in each caption is used to represent its content.

Average-pooling is applied over the 5 hidden-state vectors, to obtain a 512-d vector $\Cap(I)$ for the image~$I$. The caption-LSTM is trained on the human-generated captions from the MS COCO training set, which means that the resulting model is focused on the types of image content that humans are most interested in describing.

### Relating to the Knowledge Base

The external data source that we use here is DBpedia~(["Dbpedia: A nucleus for a web of open data"]()). As a source of general background information, although any such KB could equally be applied, DBpedia is a structured database of information extracted from Wikipedia. The whole DBpedia dataset describes $4.58$ million entities, of which $4.22$ million are classified in a consistent ontology. The data can be accessed using an SQL-like query language for RDF called SPARQL. Given an image and its predicted attributes, we use the top-5 most strongly predicted attributes, {We only use the top-5 attributes to query the KB because, based on the observation of training data, an image typically contains 5-8 attributes. We also tested with top-10, but no improvements were observed.} to generate DBpedia queries.

Inspecting the database shows that the "comment' field is the most generally informative about an attribute, as it contains a general text description of it.  We therefore retrieve the comment text for each query term. The KB+SPARQL combination is very general, however, and could be applied to problem specific KBs, or a database of common sense information, and can even perform basic inference over RDF.


Since the text returned by the SPARQL query is typically much longer than the captions generated above, we turn to Doc2Vec (["Distributed representations of sentences and documents"]()) to extract the semantic meanings.
Doc2Vec, also known as Paragraph Vector, is an unsupervised algorithm that learns fixed-length feature representations from variable-length pieces of texts, such as sentences, paragraphs, and documents. Le et.al.(["Distributed representations of sentences and documents"]()) proved that it can capture the semantics of paragraphs. A Doc2Vec model is trained to predict words in the document given the context words. We collect 100,000 documents from  DBpedia to train a model with vector size 500. To obtain the knowledge vector $\text{Know}(I)$ for image $I$, we combine the 5 returned paragraphs in to a single large paragraph, before extracting semantic features using our Doc2Vec model.


# A VQA Model with Multiple Inputs}
## Answer_Generation

We propose to train a VQA model by maximizing the probability of the correct answer given the image and question.

We want our VQA model to be able to generate multiple word answers, so we formulate the answering process as a word sequence generation procedure. Let $Q=\{q_1,...,q_n\}$ represents the sequence of words in a question, and $A=\{a_1,...,a_l\}$ the answer sequence, where $n$ and $l$ are the length of question and answer, respectively. The log-likelihood of the generated answer can be written as:
$$
    \log p(A|I,Q)=\sum_{t=1}^l \log p(a_{t}|a_{1:t-1},I,Q)
$$
where $p(a_t|a_{1:t-1},I,Q)$ is the probability of generating $a_t$ given image information $I$, question $Q$ and previous words $a_{1:t-1}$. We employ an encoder LSTM to take the semantic information from image $I$ and the question $Q$, while using a decoder LSTM to generate the answer. Weights are shared between the encoder and decoder LSTM.

In the training phase, the question $Q$ and answer $A$ are concatenated as $\{q_1,...,q_n,a_1,...,a_l,a_{l+1}\}$, where $a_{l+1}$ is a special END token. Each word is represented as a one-hot vector of dimension equal to the size of the word dictionary. The training procedure is as follows: at time step $t=0$, we set the LSTM input:
$$
   x_{initial}=[ W_{ea}\text{Att}(I), W_{ec}\Cap(I), W_{ek}\text{Know}(I) ]
$$
where $W_{ea}$, $W_{ec}$, $W_{ek}$ are learnable embedding weights for the vector representation of attributes, captions and external knowledge, respectively. In practice, all these embedding weights are learned jointly. Given the randomly initialized hidden state, the encoder LSTM feeds forward to produce  hidden state $h_{0}$ which encodes all of the input information. From $t=1$ to $t=n$, we set $x_t=W_{es}q_t$ and the hidden state $h_{t-1}$ is given by the previous step, where $W_{es}$ is the learnable word embedding weights. The decoder LSTM runs from time step $n+1$ to $l+1$. Specifically, at time step $t=n+1$, the LSTM layer takes the input $x_{n+1}=W_{es}a_1$ and the hidden state $h_{n}$ corresponding to the last word of the question, where $a_1$ is the start word of the answer. The hidden state $h_{n}$ thus encodes all available information about the image and the question. The probability distribution $p_{t+1}$ over all answer words in the vocabulary is then computed by the LSTM feed-forward process. Finally, for the final step, when $a_{l+1}$ represents the last word of the answer, the target label is set to the END token.

Our training objective is to learn parameters $W_{ea}$, $W_{ec}$, $W_{ek}$, $W_{es}$ and all the parameters in the LSTM by minimizing the following cost function:

$$
\mathcal{C}=-\frac{1}{N}\sum_{i=1}^N\log p(A^{(i)}|I,Q)+\lambda_{\text{ModParms}} \cdot||\text{ModParms}||_2^2 =-\frac{1}{N}\sum_{i=1}^N\sum_{j=1}^{l^{(i)}+1}\log p_j(a_j^{(i)})+\lambda_{\text{ModParms}}\cdot||\text{ModParms}||_2^2
$$

where $N$ is the number of training examples, and $n^{(i)}$ and $l^{(i)}$ are the length of question and answer respectively for the $i$-th training example. Let $p_t(a_t^{(i)})$ correspond to the activation of the Softmax layer in the LSTM model for the $i$-th input and $\text{ModParms}$ represent the model parameters. Note that $\lambda_\text{ModParms}\cdot||\text{ModParms}||_2^2$ is a regularization term. %


# Experiments

We evaluate our model on two recent publicly available visual question answering datasets, both based on MS COCO images. The Toronto COCO-QA Dataset (["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()} contains 78,736 training and 38,948 testing examples, which are generated from 117,684 images. There are four types of questions, relating to the object, number, color and location, all constructed so as to have a single-word answer. All of the question-answer pairs in this dataset are automatically converted from human-sourced image descriptions. Another benchmarked dataset is VQA (["VQA: Visual Question Answering"]()), which is a much larger dataset and contains 614,163 questions and 6,141,630 answers based on 204,721 MS COCO images. This dataset provides a surprising variety of question types, including ""What is...', ""How Many" and even ""Why...". The ground truth answers were generated by 10 human subjects and can be single word or sentences. The data train/val split follows the COCO official split, which contains 82,783 training images and 40,504 validation images, each has 3 questions and 10 answers. We randomly choose 5000 images from the validation set as our val set, with the remainder testing. The human ground truth answers for the actual VQA test split are not available publicly and only can be evaluated via the VQA evaluation server. Hence, we also apply our final model on a test split and report the overall accuracy.

We did not test on the DAQUAR dataset (["Towards a Visual Turing Challenge"]()) as it is an order of magnitude smaller than the datasets mentioned above, and thus too small to train our system, and to test its generality.

## Implementation Details

To train the VQA model with multiple inputs in the Section above, we use Stochastic gradient Descent (SGD) with mini-batches of 100 image-QA pairs. The attributes, internal textual representation, external knowledge embedding size, word embedding size and hidden state size are all 256 in all experiments. The learning rate is set to 0.001 and clip gradients is 5. The dropout rate is set to 0.5.

## Results on Toronto COCO-QA
### Metrics

Following (["Learning to Answer Questions From Image using Convolutional Neural Network"]()), ["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()), the accuracy value (the proportion of correctly answered test questions),

and the Wu-Palmer similarity (WUPS) (["Verbs semantics and lexical selection"]()) are used to measure performance. The WUPS calculates the similarity between two words based on the similarity between their common subsequence in the taxonomy tree. If the similarity between two words is greater than a threshold then the candidate answer is considered to be right.

We report on thresholds 0.9 and 0.0, following (["Learning to Answer Questions From Image using Convolutional Neural Network"]()), ["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()).


### Evaluations

To illustrate the effectiveness of our model, we provide a baseline and several state-of-the-art results on the Toronto COCO-QA dataset. The $\textbf{Baseline}$ method is implemented simply by connecting a CNN to an LSTM. The CNN is a pre-trained (on ImageNet) VggNet model from which we extract the coefficients of the last fully connected layer.

We also implement a baseline model $\textbf{VggNet+ft-LSTM}$, which applies a VggNet that has been fine-tuned on the COCO dataset, based on the task of image-attributes classification.

We also present results from a series of cut down versions of our approach for comparison.

$\textbf{Att-LSTM}$ uses only the semantic level attribute representation $\text{Att}$ as the LSTM input. To evaluate the contribution of the internal textual representation and external knowledge for the question answering, we feed the image caption representation $\Cap$ and knowledge representation $\text{Know}$ with the $\text{Att}$ separately, producing two models, $\textbf{Att+Cap-LSTM}$ and $\textbf{Att+Know-LSTM}$. We also tested the $\textbf{Cap+Know-LSTM}$, for the experiment completeness. Our final model is the $\textbf{Att+Cap+Know-LSTM}$, which combines all the available information.

$\textbf{GUESS}$ (["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()) simply selects the modal answer from the training set for each of 4 question types (the modal answers are "cat', "two', "white', and "room').

$\textbf{VIS+BOW}$ (["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()) performs multinomial logistic regression based on image features and a BOW vector obtained by summing all the word vectors of the question. $\textbf{VIS+LSTM}$ (["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()) has one LSTM to encode the image and question, while $\textbf{2-VIS+BLSTM}$~(["Image Question Answering: A Visual Semantic Embedding Model and a New Dataset"]()) has two image feature inputs, at the start and the end. Ma et al. (["Learning to Answer Questions From Image using Convolutional Neural Network"]()) encode both images and questions with a CNN.

## Results on the VQA

Antol et.al.in (["antol2015vqa} provide the VQA dataset which is intended to support ""free-form and open-ended Visual Question Answering". They also provide a metric for measuring performance:
$\min\{\frac{\text{# humans that said answer}}{3},1\}$, thus $100\%$ means that at least 3 of the 10 humans who answered the question  gave the same answer.

Inspecting Table~\ref{tab4}, results on the VQA validation set, we see that the attribute-based \textbf{Att-LSTM} is a significant improvement over our $\textbf{VggNet+LSTM}$ baseline. We also evaluate another baseline, the $\textbf{VggNet+ft+LSTM}$, which uses the penultimate layer of the attributes prediction CNN as the input to the LSTM. Its overall accuracy on the VQA is 50.01, which is still lower than our proposed models. Adding either image captions or external knowledge further improves the result. The model $\textbf{Cap+Know}$ produces overall accuracy 52.31, slightly lower than Att+Know (53.79). This suggests that the attributes representation plays a more important role here.

Our final model $\textbf{A+C+K-LSTM}$ produces the best results, outperforming the baseline $\textbf{VggNet-LSTM}$ by 11\% overall. %

Antol et.al.(["VQA: Visual Question Answering"]()) provide several results for this dataset. In each case they encode the image with the final hidden layer from VggNet, and questions are encoded using a BOW representation. A softmax neural network classifier with 2 hidden layers and 1000 hidden units (dropout 0.5) in each layer with tanh non-linearity is then trained, the output space of which is the 1000 most frequent answers in the training set. They also provide an LSTM model followed by a softmax layer to generate the answer. Two versions of this approach are used, one which is given both the question and the image, and one which is given only the question (see~(["VQA: Visual Question Answering"]()) for details).

Our final model outperforms the listed approaches according to the overall accuracy and all answer types.
