## [A Diagram Is Worth A Dozen Images](https://github.com/seominjoon/dqa-net)/[Paper](http://arxiv.org/abs/1603.07396)

## abstract

Diagrams are common tools for representing complex concepts, relationships and events, often when it would be difficult to portray the same information with natural images. Understanding natural images has been extensively studied in computer vision, while diagram understanding has received little attention. In this paper, we study the problem of diagram interpretation and reasoning, the challenging task of identifying the structure of a diagram and the semantics of its constituents and their relationships. We introduce Diagram Parse Graphs (DPG) as our representation to model the structure of diagrams. We define syntactic parsing of diagrams as learning to infer DPGs for diagrams and study semantic interpretation and reasoning of diagrams in the context of diagram question answering. We devise an LSTM-based method  for syntactic parsing of diagrams and  introduce a DPG-based attention model for diagram question answering. We compile a new dataset of diagrams with exhaustive annotations of constituents and relationships for over 5,000 diagrams and 15,000 questions and answers. Our results show the significance of our models for syntactic parsing and question answering in diagrams using DPGs.

## Introduction

For thousands of years visual illustrations have been used to depict the lives of people, animals,  their environment, and major events. Archaeological discoveries have unearthed cave paintings showing lucid representations of hunting, religious rites, communal dancing, burial, etc.
From ancient rock carvings and maps, to modern info-graphics and 3-D visualizations, to diagrams in science textbooks, the set of visual illustrations  is very large, diverse and ever growing, constituting a considerable portion of visual data. These  illustrations  often represent complex concepts, such as events or systems, that are otherwise difficult to portray in a few sentences of text or a natural image.

While understanding natural images has been a major area of research in computer vision, understanding rich visual illustrations has received scant attention.

From a computer vision perspective, these illustrations are inherently different from natural images and offer a unique and interesting set of problems. Since they are purposefully designed to express information, they typically suppress irrelevant signals such as background clutter, intricate textures and shading nuances. This often makes the detection and recognition of individual elements inherently different than their counterparts, objects, in natural images. On the other hand, visual illustrations may depict complex phenomena and higher-order relations between objects (such as temporal transitions, phase transformations and inter object dependencies) that go well beyond what a single natural image can convey. For instance, one might struggle to find natural images that compactly represent the phenomena seen in some grade school science diagrams. In this paper, we define the problem of understanding visual illustrations by identifying visual entities and their relations as well as establishing semantic correspondences to real-world concepts.

The characteristics of visual illustrations also afford opportunities for deeper reasoning than provided by natural images. One can further reason about higher order relations between entities such as the effect on the population of foxes caused by a reduction in the population of plants. Some of these phenomena are shown to occur on the surface of the earth while others occur either above or below the surface. The main components of the cycle (e.g., evaporation and condensation) are labeled and the flow of water is displayed using arrows. Reasoning about these objects and their interactions in such rich scenes provides many exciting research challenges.


In this paper, we address the problem of diagram} interpretation and reasoning in the context of science diagrams, defined as the two tasks of <span style="color:red">Syntactic parsing</span> and <span style="color:red">Semantic interpretation</span>. <span style="color:red">Syntactic parsing</span> involves detecting and recognizing constituents and their syntactic relationships in a diagram. This is most analogous to the problem of scene parsing in natural images. The wide variety of diagrams as well as large intra-class variation shows several varied images depicting a water cycle) make this step very challenging.  <span style="color:red">Semantic interpretation</span> is the task of  mapping constituents and their relationships  to semantic entities and events (real-world concepts). This is a challenging task given the inherent ambiguities in the mapping functions. For example, an arrow in a food chain diagram typically corresponds to the concept of <span style="color:red">consumption</span>, arrows in water cycles typically refer to phase changes, and arrows in a planetary diagram often refers to rotatory motion.

We introduce a  representation to encode diagram constituents and their relationships in a graph, called diagram parse graphs (DPG).

The problem of syntactic parsing of diagrams is formulated as the task of learning to infer the DPG that best explains a diagram.  We introduce a <span style="color:red">Deep Sequential Diagram Parser Network</span> that learns to sequentially add relationships and their constituents to form DPGs, using LSTM networks. The problem of semantically interpreting a diagram and reasoning about the constituents and their relationships is studied in the context of diagram question answering.  We present a neural network architecture (called \questionnet) that learns to attend to useful relations in a DPG given a question about the diagram.

We compile a dataset named AI2 Diagrams of over 5000 grade school science diagrams with over 150000 rich annotations, their ground truth syntactic parses, and more than 15000 corresponding multiple choice questions. Our experimental results show that the proposed LSTM for syntactic parsing outperforms several baseline methods. Moreover, we show that the proposed approach of incorporating diagram relations into question answering outperforms standard visual question answering methods.

Our contributions include:
1. We present two new tasks of diagram interpretation and reasoning,
2. we introduce the DPG representation to encode diagram parses and introduce a model that learns to map diagrams into DPGs,
3. we introduce a model for diagram question answering that learns the attention of questions into DPGs and
4. we present a new dataset to evaluate the above models with baselines.


## Background

<span style="color:red">Understanding diagrams}$

The problem of understanding diagrams received a fair amount of interest in the 80's and 90's.
1. [Computational models for integrating linguistic and visual information: A survey]()
2. [Telling juxtapositions: Using repetition and alignable difference in diagram understanding]()
3. [Diagram understanding using integration of layout information and textual information]()
4. [Document Image Analysis]()

However, many of these techniques either used hand written rules, assumed that the visual primitives were manually identified or worked on a specific set of diagrams.

More recently, Futrelle et al.(["Extraction, layout analysis and classification of diagrams in pdf documents"]()) proposed methods to analyze graphs and finite automata sketches but only worked with vector representations of these diagrams. Recently, Seo et al. (["Diagram understanding in geometry questions"]()) proposed a method for understanding diagrams in geometry questions that identifies visual elements in a diagram while maximizing agreement between textual and visual data. In contrast to these past approaches, we propose a unified approach to diagram understanding that builds upon the representational language of graphic representations proposed by Engelhardt (["The language of graphics: A framework for the analysis of syntax and meaning in maps, charts and diagrams"]()) and works on a diverse set of diagrams.

The domain of abstract images has also received a considerable amount of interest over the past couple of years (["Bringing semantics into focus using visual abstraction,Yin and yang: Balancing and answering binary visual questions,Learning common sense through visual abstraction,Zero-shot learning via visual abstraction"]()). While abstract images significantly reduce the noise introduced by low level computer vision modules, thus bringing the semantics of the scene into focus, they still depict real world scenes, and hence differ significantly from diagrams which may depict more complex phenomena.


$\mathbf{Parsing natural images}$

Several approaches to building bottom-up and top-down parsers have been proposed to syntactically parse natural images and videos. These include Bayesian approaches (["Image parsing: Unifying segmentation, detection, and recognition"]()), And-Or graph structures (["A stochastic grammar of images"]()), stochastic context free grammars (["Bayesian grammar learning for inverse procedural modeling"]()), regular grammars (["Parsing videos of actions with segmental grammars"]()), 3D Geometric Phrases (["Understanding indoor scenes using 3d geometric phrases"]()) and a max margin structured prediction framework based on recursive neural networks (["Parsing natural scenes and natural language with recursive neural networks"]()). Inspired by these methods, we adopt a graph based representation for diagrams.


$\mathbf{Answering questions}$

One of relevant tasks in NLP is machine comprehension, which is to answer questions about a reading passage (["Towards ai-complete question answering: {A} set of prerequisite toy  tasks,{A} challenge dataset for the open-domain machine comprehension of text,Teaching machines to read and comprehend"]()).
Our QA system is similar to memory networks (["End-to-end memory networks"]()) in that we use attention mechanism to focus on the best supporting fact among possible candidates.
While these candidates are trivially obtained from the given passage in (["End-to-end memory networks"]()), in DQA we obtain them from diagram via its parse graph.
Recently, answering questions about real images has drawn much attention in both NLP and vision communities (["Vqa: Visual question answering,Exploring models and data for image question answering,Visual7w: Grounded question answering in images"]()).
However, diagram images are vastly different from real images, and so are the corresponding questions. Hence, most QA systems built for real images (["Learning to compose neural networks for question answering,Image question answering using convolutional neural network with dynamic parameter prediction"]()) cannot be directly used for diagram QA tasks.

# The Language of Diagrams

Much of the existing literature on graphic representations (["Visual language: Global communication for the 21st century,Readings in information visualization: using vision to think,A schema for the study of graphic language"]()) covers only specific types of graphics or specific aspects of their syntactic structure. More recently, Engelhardt (["The language of graphics: A framework for the analysis of syntax and meaning in maps, charts and diagrams"]()) proposed a coherent framework integrating various structural, semiotic and classification aspects that can be applied to the complete spectrum of graphic representations including diagrams, maps and more complex computer visualizations.
We briefly describe some of his proposed principles below, as they apply to our space of diagrams, but refer the reader to (["The language of graphics: A framework for the analysis of syntax and meaning in maps, charts and diagrams"]()) for a more thorough understanding.

A diagram is a composite graphic that consists of a graphic space, a set of constituents,  and a set of relationships involving these constituents. A graphic space may be a metric space, distorted metric space (e.g., a solar system diagram) or a non-meaningful space (e.g., a food web). Constituents in a diagram include illustrative elements (e.g., drawings of animals and plants), textual elements, diagrammatic elements (e.g., arrows and grid lines), informative elements (e.g., legends and captions) and decorative elements. Relationships include spatial relations between constituents and their positions in the diagram space, and spatial and attribute-based relations between constituents (e.g., linkage, lineup, variation in color, shape).
An individual constituent may itself be a composite graphic, rendering this formulation recursive.


We build upon Engelhardt's representation by introducing the concept of <span style="color:red">Diagrammatic Objects</span> in our diagrams, defined as the primary entities being described in the diagram. Examples of objects include animals in the food web, the human body in an anatomy diagram,  and the sun in water cycle. The relationships within and between objects include intra-object, inter-object, and constituent-space relationships.
We represent a diagram with a  Diagram Parse Graph} (DPG), in which nodes correspond to constituents} and edges  correspond to relationships} between the constituents. We model four types of constituents:  Blobs (Illustrations), Text Boxes, Arrows, and Arrow Heads.

#  Syntactic Diagram Parsing

Syntactic diagram parsing is the problem of learning to map diagrams into DPGs. Specifically, the goal is to detect and recognize constituents and their syntactic relationships in a diagram and find the DPG that best explains the diagram.
In order to form candidate DPGs, we first generate proposals for nodes in the DPG using object detectors built for each constituent category. We also generate proposals for edges in the DPG by combining proposal nodes using relationship classifiers. Given sets of noisy node and edge proposals, our method then selects a subset of these to form a DPG by exploiting several local and global cues.

The constituent and relationship proposal generators result in several hundred constituent proposals and several thousand relationship proposals per diagram. These large sets of proposals, the relatively smaller number of true nodes and edges in the truth DPG and the rich nature of the structure of our DPGs, makes the search space for possible parse graphs incredibly large.
We observe that forming a DPG amounts to choosing a subset of relationships among the proposals. Therefore, we propose a sequential formulation to this task that adds a relationship and its constituents at every step, exploiting local cues  as well as long range global contextual cues using a memory-based model.


### Model

We introduce a Deep Sequential Diagram Parser. Central to this is a stacked Long Short Term Memory (LSTM) recurrent neural network (["Long short-term memory"]()) with fully connected layers used prior to, and after the LSTM. Proposal relationships are then sequentially fed into the network, one at every time step, and the network predicts if this relationship (and its constituents) should be added to the DPG or not. Each relationship in our large candidate set is represented by a feature vector, capturing the spatial layout of its constituents in image space and their detection scores.


### Training

LSTM networks typically require large amounts of training data. We provide training data for the lstmnetwork in the form of sequences of relationships by sampling from training diagrams. For each training diagram, we sample a large number of relationship sequences using sampling without replacement from  thousands of relationship candidates, utilizing the relationship proposal scores as sampling weights. For each sampled sequence, we sequentially label the relationship at every time step by comparing the generated DPG to the groundtruth DPG.

A relationship labeled with a positive label in one sampled sequence may be labeled with a negative label in another sequence due to the presence of overlapping constituents and relationships in our candidate sets.

The lstmnetwork  is able to model dependencies between nodes and edges selected at different time steps in the sequence. It chooses relationships with large proposal scores but also learns to reject relationships that may lead to a high level of spatial redundancy or an incorrect structure in the layout. It also works well with a variable number of candidate relationships per diagram. Finally, it  learns to stop adding relationships once the entire image space has been covered by the nodes and edges already present in the graph.

### Test

At test time, relationships in the candidate set are sorted by their proposal scores and presented to the network. Selected relationships are then sequentially added to form the final DPG.

# Semantic Interpretation

DPGs represent the syntactic relationships between constituents of a diagram. They, however, do not encode the semantics of constituents and relationships.
Constituents and relationships with a similar visual representation may have different semantic meanings in different diagrams. For example, the Inter-Object Linkage relationship can be interpreted as consuming} in food webs and as evaporation} in water cycles. Moreover, diagrams typically depict complex phenomena and reasoning about these phenomena goes beyond the tasks of matching and interpretation.

In order to evaluate the task of reasoning about the semantics of diagrams, we study semantic interpretation of diagrams in the context of diagram question answering. This is inspired by evaluation paradigms in school education systems and the recent progress in visual and textual question answering. Studying semantic interpretation of diagrams in the context of question answering also provides a well-defined problem definition, evaluation criteria, and metrics.


### Diagram Question Answering

A diagram question consists of a diagram $d$ in raster graphics, a question sentence $q$, and multiple choices $\{c_1\ldots c_4\}$. The goal of question answering is to select a single correct choice $c_k$ given $d$ and $q$.

We design a neural network architecture (called \questionnet) to answer diagrammatic questions. The main intuition of the network is to encode the DPG into a set of facts and learn an attention model to find the closest fact to the question.

A question embedding module that takes the question $q$ and a choice $c_k, k\in\{1\ldots4\}$ to build a statement $s_k$ and uses an LSTM to learn a $d$-dimensional embedding of the statement $s_k\in \mathbb{R}^d$; (b) a diagram embedding module that takes the DPG, extracts $M$ relations $m_i, i\in\{1\ldots M\}$ from DPG, and uses an LSTM to learn a $d$-dimensional embedding of diagram relations $m_i\in \mathbb{R}^d$; (c) an attention module that learns to attend to the relevant diagram relations by selecting the best statement choice that has a high similarity with  the relevant diagram relations. For every statement $s_k$, our model computes a probability distribution over statement choices by feeding the best similarity scores between statements and diagram relations through a softmax layer.
$$
    \gamma_k=\max_i{s_k^T\cdot m_i}, \ \ \ \  \hat{y}=\text{softmax}_k(\gamma_k)=\frac{\exp(\gamma_k)}{\sum_{k'}\exp(\gamma_{k'})}
$$
We use cross entropy loss to train our model: $  L(\theta)=H(y,\hat{y})=-\sum_ky_k\log \hat{y}_k$. More details about the parameters can be found in Section~\ref{sec:dqaexperiment}.


# Dataset

We build a new dataset (named AI2 Diagrams), to evaluate the task of diagram interpretation. datasetname is comprised of more than 5,000 diagrams representing topics from grade school science, each annotated with constituent segmentations, their relationships to each other and their relationships to the diagram canvas. In total, datasetname contains annotations for more than 118K constituents and 53K relationships. The dataset is also comprised of more than 15000 multiple choice questions associated to the diagrams. We divide the datasetname dataset into a train set with 4000 images and a blind test set with 1000 images and report our numbers on this blind test set.


The images are collected by scraping Google Image Search with seed terms derived from the chapter titles in Grade 1 - 6 science textbooks. Each image is annotated using Amazon Mechanical Turk (AMT). Annotating each image with rich annotations such as ours, is a rather complicated task and must be broken down
into several phases to maximize the level of agreement obtained from turkers. Also, these phases need to be carried out sequentially to avoid conflicts in the annotations.

The phases involve :
1. annotating the four low-level constituents,
2. categorizing the text boxes into one of four categories: relationship with the canvas, relationship with a diagrammatic element, intra-object relationship and inter-object relationship,
3. categorizing the arrows into one of three categories: intra-object relationship, inter-object relationship or neither,
4. labelling intra-object linkage and inter-object linkage relationships. For this step, we display arrows to turkers and have them choose the origin and destination constituents in the diagram,
5. labelling intra-object label, intra-object region label and arrow descriptor relationships. For this purpose, we display text boxes to turkers and have them choose the constituents related to it,
6. and finally multiple choice questions with answers, representing grade school science questions are then obtained for each image using AMT.

# Experiments

We describe methods used to generate constituent and relationship proposals and show evaluations of our methods in generating proposals versus several baselines. We also evaluate our introduced model lstmnetwork\  for syntactic parsing of diagrams that forms DPGs and compare it to several baseline approaches. Finally, we evaluate the proposed diagram question answering model questionnet and compare with standard visual question answering approaches. In each section, we also describe the hyperparameters, features, and the baselines.

## Generating Constituent Proposals

### Diagram Canvas

A diagram consists of multiple constituents overlaid onto a canvas, which may be uniformly colored, textured or have a blended image. We classify every pixel in the diagram into canvas vs constituent. We build non-parametric kernel density estimates (KDE) in RGB, texture and entropy spaces to generate features for a Random Forest (RF) classifier with 100 trees to obtain an Average Precision (AP) of 0.9142.


$\textbf{Detecting blobs:}$ Blobs exhibit a large degree of variability in their size, shape and appearance in diagrams, making them challenging to model. We combine segments at multiple levels of a segmentation hierarchy, obtained using Multiscale Combinatorial Grouping (MCG) (["Multiscale combinatorial grouping"]())
with segments produced using the canvas probability map to produce a set of candidates. Features capturing the location, size, central and Hu moments, etc. are provided to an RF classifier with 100 trees.

<span style="color:red">Baselines</span>.

We evaluated several object proposal approaches including Edge Boxes (["Edge boxes: Locating object proposals from edges"]()), Objectness (["Measuring the objectness of image windows"]()) and Selective Search (["Selective search for object recognition"]()). Since these are designed to work on natural images, they do not provide good results on diagrams. We compare the RF approach to Edge Boxes, the most suitable of these methods, since it uses edge maps to propose objects and relies less on colors and gradients observed in natural images.

$\textbf{Detecting arrow tails: }$

Arrow tails are challenging to model since they are easily confused with other line segments present in the diagram and do not always have a corresponding arrow head to provide context. We generate proposal segments using a three pronged approach. We obtain candidates using the boundary detection method in (["Highly accurate boundary detection and grouping"]()),
Hough transforms and by detecting parallel curved edge segments in a canny edge map; and recursively merge proximal segments that exhibit a low residual when fit to a 2\textsuperscript{nd} degree polynomial. We then train a binary class Convolutional Neural Network (CNN) resembling the architecture of the VGG-16 model by (["Very deep convolutional networks for large-scale image recognition"]()), with a fourth channel appended to the standard three channel RGB input. This fourth channel specifies the location of the arrow tail candidate smoothed with a Gaussian kernel of width 5. All filters except the ones for the fourth input channel at layer 1 are initialized from a publicly available VGG-16 model. The remaining filters are initialized with random values drawn from a Gaussian distribution. We use a batch size of 32 and a starting learning rate (LR) of 0.001. {\bf Results.} Figure~\ref{fig:arrowsPR} shows the PR curve for our model with an AP of 0.6748. We tend to miss arrows that overlap significantly with more than three other arrows in the image as well as very thick arrows that are confused for blobs.

$\textbf{Detecting arrow heads: }$

Arrow head proposals are obtained by a scanning window approach over 6 scales and 16 orientations. RGB pixels in each window undergo PCA followed by a 250 tree RF classifier. We then train a binary class CNN resembling the standard architecture of AlexNet (["Imagenet classification with deep convolutional neural networks"]()) and initialize using a publicly available model. We use a batch size of 128 and a starting LR of 0.001. {\bf Results.} Figure~\ref{fig:arrowHeadsPR} shows the PR curves for our CNN model as well as the first pass RF model. We miss arrow heads which are extremely small and some which are present in poor quality images.


$\textbf{Detecting text: }$ We use an Optical Character Recognition (OCR) service (["Project oxford"]()) provided by Microsoft's Project Oxford to localize and recognize text in our diagrams. To improve the performance on single characters, we train a single character localizer using a CNN having the same architecture as AlexNet (["Imagenet classification with deep convolutional neural networks"]()). We use three training resources: (1) Chars74K (a character recognition dataset for natural images (["Character recognition in natural images"]()), (2) a character dataset obtained from vector PDFs of scientific publications and (3) a set of synthetic renderings of single characters. The localized bounding boxes are then recognized using Tesseract (["Open source ocr engine"]()). {\bf Results.} Using Tesseract end-to-end provides poor text localization results for diagrams with a 0.2 precision and a 0.46 recall. Our method improves the precision to 0.89 and recall to 0.75. Our false negatives comprise of vertically oriented and curved text, cursive fonts and unmerged multi-line blocks.

## Generating Relationship Proposals

Categories $\mathbb{R}_1$ through $\mathbb{R}_6$ relate two or more constituents with one another. We compute features capturing the spatial layout of the constituents with respect to one another as well as the diagram space and combine them with detection probabilities provided by the low level constituent models. A 100 trees RF classifier is trained for each category. At test time, we generate proposal relationships from the large combinatorial set of candidate constituents using a proximity based pruning scheme, which get classified by the RF model. Categories $\mathbb{R}_7$ through $\mathbb{R}_{10}$ relate a single constituent with the entire image. We model each category using a non parametric Kernel Density Estimate (KDE) in X,Y space. At test time, every candidate text detection is passed through the KDE models to obtain a relationship probability. {\bf Results.} Figure~\ref{fig:relPR} shows the PR curves for the relationships built using the RF classifier. The AP for several of relationships is low, owing to the inherent ambiguity in classifying relationships using local spatial decisions.

## Syntactic Parsing: DPG Inference

$\textbf{Our model DSDP-Net: }$
The introduced lstmnetwork~(depicted in Figure~\ref{fig:lstmNetwork}) consists of a 2 layer stacked LSTM with each layer having a hidden state of dimensionality 512. The LSTM is preceded by two fully connected layers with an output dimensionality of 64 and a Rectified Linear Unit (ReLu) (["Rectified linear units improve restricted boltzmann machines"]()) activation function each. The LSTM is proceeded by a fully connected layer with a softmax activation function. This network is trained using RMSProp (["Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude"]()) to optimize the cross-entropy loss function. The initial learning rate is set to 0.0002.

Each candidate relationship is represented as a 92 dimensional feature vector that includes features for each constituent in the relationship (normalized x,y coordinates, detection score, overlap ratio with higher scoring candidates and the presence of this constituent in relationships presented to the network at prior time-steps) and features describing the relationship itself (relationship score and the presence of tuples of candidates in relationships presented to the network at previous time steps). We sample 100 relationship sequences per training image to generate roughly 400000 training samples. At test time, relationships are presented to the network in sorted order, based on their detection scores.

$\textbf{Baselines: }$
<span style="color:red">Greedy Search:</span>
The first baseline is a greedy algorithm whereby nodes and edges are greedily added to the DPG using their proposal scores. It uses an exit} model as a stopping condition. The exit model is trained to score DPGs based on their distance to the desired completed DPG. To train the exit model, we use features capturing the quality, coverage, redundancy and structure of the nodes and edges in the DPGs and use 100 tree RF models.

<span style="color:red">A* Search:</span>
The second baseline is an A* search, which starts from an empty DPG and sequentially adds nodes and edges according to a cost. We improve upon the greedy algorithm by training a RF model that utilizes local and contextual cues to rank available constituents and relationships. The cost function for each potential step is a linear combination of this RF model's score and the distance of the resultant DPG to the desired complete DPG. In order to model the distance function, we use the same exit model as before to approximate the distance from the goal.

<span style="color:red">Direct Regression:</span>
We also trained a CNN to directly regress the DPG, akin to YOLO (["You only look once: Unified, real-time object detection"]()). This generated no meaningful results on our dataset.

$\mathbf{Evaluation.}$ To evaluate these methods, we compute the Jaccard Index between the sets of nodes and edges in our proposed DPG and and the ground truth DPG. We refer to this metric by the Jaccard Index for Graphs (JIG) score. The Jaccard Index, which measures similarity between finite sample sets, is defined as the size of the intersection divided by the size of the union of the sample sets.

## Diagram Question Answering

$\mathbf{ Our model DQA-Net:}$ questionnet uses GloVe (["Glove: Global vectors for word representation"]()) model pre-trained on 6B tokens (Wikipedia 2014) to map each word to a 300D vector.
The LSTM units have a single layer, $50$ hidden units, and forget bias of $2.5$.
We place a single 50-by-300 FC layer between the word vectors and the LSTM units.
The LSTM variables in all sentence embeddings (relation and statement) are shared.
The loss function is optimized with stochastic gradient descent with the batch size of $100$.
Learning rate starts at $0.01$ and decays by $0.5$ in every $25$ epochs, for $100$ epochs in total.

$\mathbf{ Baselines.}$ We use the best model (LSTM Q+I) from (["Vqa: Visual question answering"]()) as the baseline, which consists of an LSTM module for statement embedding and a CNN module for diagram image embedding.
In the LSTM module, we use the same setup as \questionnet, translating question-answer pairs to statements and obtaining 50D vector for each statement.
In the CNN module, we obtain 4096D vector from the last layer of pre-trained VGG-19 model (["Very deep convolutional networks for large-scale image recognition"]()) for each diagram image. Each image vector is transformed to a 50D vector by a single 50-by-4096 FC layer.
We then compute the dot product between each statement embedding and the transformed 50D image vector,
followed by a softmax layer. We use cross entropy loss and the same optimization techniques as in \questionnet.

## Libraries
We use Keras (["chollet2015keras"]()) to build our constituent CNN models and the lstmnetwork\ network, TensorFlow (["tensorflow2015-whitepaper"]()) to build our questionnet network and Scikit-learn (["scikit-learn"]()) to build our Random Forest models.

# Conclusion

We introduced the task of diagram interpretation and reasoning.  We proposed lstmnetwork\ to parse diagrams and reason about the global context of a diagram using our proposed representation called DPG. DPGs encode most useful syntactic  information depicted in the diagram. Moreover, we introduced questionnet that learns to answer diagram questions by attending to diagram relations encoded with DPGs. The task of diagram question answering is a well-defined task to evaluate capabilities of different systems in semantic interpretation of diagrams and reasoning. Our experimental results show improvements of lstmnetwork\ in parsing diagrams compared to strong baselines. Moreover, we show that questionnet outperforms standard VQA techniques in diagram question answering.

Diagram interpretation and reasoning raises new research questions that goes beyond natural image understanding. We release datasetname and our baselines to facilitate further research in diagram understanding and reasoning. Future work involves incorporating diagrammatic and commonsense knowledge in DQA.
