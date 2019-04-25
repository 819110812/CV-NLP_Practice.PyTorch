# [MovieQA: Understanding Stories in Movies through Question-Answering](http://arxiv.org/pdf/1512.02902v1.pdf)

# abstract

We introduce the MovieQA dataset which aims to evaluate automatic story comprehension from both video and text.
The dataset consists of 7702 questions about 294 movies with high semantic diversity. The questions range from simpler "Who" did "What" to "Whom", to "Why" and "How" certain events occurred. Each question comes with a set of five possible answers; a correct one and four deceiving answers provided by human annotators.
Our dataset is unique in that it contains multiple sources of information -- full-length movies, plots, subtitles, scripts and for a subset DVS from (["A Dataset for Movie Description"]()). We analyze our data through various statistics and intelligent baselines. We further extend existing QA techniques to show that question-answering with such open-ended semantics is hard.
We plan to create a benchmark with an active leader board, to encourage inspiring work in this challenging domain.

# Introduction

Fast progress in Deep Learning as well as a large amount of available labeled data has significantly pushed forward the performance in many visual tasks such as image tagging, object detection and segmentation, action recognition, and image/video captioning.
We are steps closer to applications such as assistive solutions for the visually impaired, or cognitive robotics, which require a holistic understanding of the visual world by reasoning about all these tasks in a common framework.
However, a truly intelligent machine would ideally also infer high-level semantics underlying human actions such as motivation, intent and emotion, in order to react and, possibly, communicate appropriately.
These topics have only begun to be explored in the literature (["Inferring the Why in Images,Aligning Books and Movies: Towards Story-like Visual Explanations by  Watching Movies and Reading Books"]()), as well as have the ability to communicate  with the humans in natural language. %, and to react appropriately in a variety of situations.

A great way of showing one's understanding about the scene is to be able to answer any question about it (["A Multi-World Approach to Question Answering about Real-World Scenes based on Uncertain Input1"]()).
This idea gave rise to several question-answering datasets which provide a set of questions for each image along with multi-choice answers.
These datasets are either based on RGB-D images (["A Multi-World Approach to Question Answering about Real-World Scenes based on Uncertain Input"]()) or a large collection of static photos such as Microsoft COCO (["VQA,Visual Madlibs: Fill in the blank Image Generation and Question  Answering"]()).
The types of questions typically asked are  $\mathbf{what}$ is there and $\mathbf{where}$ is it, what attributes an object has, what is its relation to other objects in the scene, and $\mathbf{how many}$ objects of certain type are present.

While these questions verify the holistic nature of our vision algorithms, there is an inherent limitation in what can be asked about a static image.
High-level semantics about actions and their intent is mostly lost and can typically only be inferred from temporal, possibly life-long visual observations.

In this paper, we argue that question-answering about movies ...
Movies provide us with snapshots from people's lives that link into stories, allowing an experienced human viewer to get a high-level understanding of the characters, their actions, and the motivations behind them. % as well as the emotions they are feeling by taking them. %We believe that a machine able to answer a diverse set of questions about stories of such complexity, demonstrates both deep understanding

Our goal here is to create a question-answering dataset that will push such automatic semantic understanding to the next level, required to truly understand stories of such complexity.
Our goal is to create a question-answering dataset to evaluate machine comprehension of both, complex videos such as movies and their accompanying text.  We believe that this data will help push automatic semantic understanding to the next level, required to truly understand stories of such complexity.



This paper introduces MovieQA, a large-scale question-answering dataset about movies.
Our dataset consists of 7702 questions about 294 movies with high semantic diversity.
For 34 of these movies, we have timestamp annotations indicating the location of the question in the video.
The questions range from simpler Who did What to Whom that can be solved by vision alone, to Why and How something happened, questions that can only be solved by exploiting both the visual information and dialogs. Each question has a set of five possible answers; one correct and four deceiving answers provided by the human annotators.
Our dataset is unique in that it contains multiple sources of information: full-length movies, subtitles, scripts, and plots.
For a subset of our movies, DVS (described video for the blind) is also available from (["A Dataset for Movie Description"]()).
We analyze our data through various statistics and intelligent baselines that mimic how different "students" would approach the quiz.
We further extend existing QA techniques to work with our data and show that question-answering with such open-ended semantics is hard.

We plan to create an $\mathbf{online benchmark}$, encouraging inspiring work in this challenging domain. We expect this benchmark to be online in early 2016 . It will have 15,000 questions and 75,000 answers, with the test set ground-truth for 5,000 questions held-out. Various sub challenges will evaluate performance with different sources of information (visual and various forms of text).

# Related work

Integration of language and vision is a natural step towards improved understanding and is receiving increasing attention from the research community.
This is in large part due to efforts in large-scale data collection such as Microsoft's COCO (["Microsoft COCO: Common Objects in Contex"]()), Flickr30K (["Flickr30k"]()) and Abstract Scenes (["Adopting abstract images for semantic scene understanding"]()) providing tens to hundreds of thousand images with natural language captions.
Another way of conveying semantic understanding of both vision and text is by retrieving semantically meaningful images given a natural language query (["Deep Visual-Semantic Alignments for Generating Image Descriptions"]()).
An interesting direction, particularly for the goals of our paper, is also the task of learning common sense knowledge from captioned  images (["Learning Common Sense Through Visual Abstraction"]()).
This has so far been demonstrated only on  synthetic clip-art scenes which enable perfect visual parsing.

$\mathbf{Video understanding via language}$

In the video domain, there are fewer works on integrating vision and language, likely due to less available labeled data.
In (["Long-term Recurrent Convolutional Networks for Visual Recognition and Description,Translating Videos to Natural Language Using Deep Recurrent Neural Networks"]()), the authors caption video clips using LSTMs,  (["Translating Video Content to Natural Language Descriptions"]()) formulates description as a machine translation model, while older work uses templates (["Video-In-sentences Out,A Thousand Frames in Just a Few Words: Lingual Description of Videos through Latent Topics and Sparse Object Stitching,Generating Natural-Language Video Descriptions Using Text-Mined Knowledge"]()).
In (["Visual Semantic Search: Retrieving Videos via Complex Textual Queries"]()), the authors retrieve relevant video clips for natural language queries, while (["Video Event Understanding using Natural Language Descriptions"]()) exploits captioned clips to  learn action and role models.
For TV series in particular, the majority of work aims at recognizing and tracking characters in the videos (["Semi-supervised Learning with Constraints for Person Identification in Multimedia Dat,Finding Actors and Actions in Movies,Linking People in Videos with "Their" Names Using Coreference Resolution,Who are you?- Learning person specific classifiers from vide"]()).
In (["Movie/Script: Alignment and Parsing of Video and Text Transcription,Subtitle-free Movie to Script Alignment"]()), the authors aligned videos with movie scripts in order to improve scene prediction.
 (["Aligning Plot Synopses to Videos for Story-based Retrieval"]()) aligns movies with their plot synopses with the aim to allow semantic browsing of large video content via textual queries.
Just recently, (["Book2Movie: Aligning Video scenes with Book chapter,Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books"]()) aligned movies to books with the aim to ground temporal visual data with verbose and detailed descriptions available in books.

$\mathbf{Question-answering}$
QA is a popular task in NLP with significant advances made recently with neural models such as memory networks (["End-To-End Memory Networks"]()), deep LSTMs (["Teaching Machines to Read and Comprehend"]()), and structured prediction (["Machine Comprehension with Syntax, Frames, and Semantics"]()).
In computer vision, (["A Multi-World Approach to Question Answering about Real-World Scenes based on Uncertain Input"]()) proposed a Bayesian approach on top of a logic-based QA system (["Learning dependency-based compositional semantics"]()), while (["Ask Your Neurons: A Neural-based Approach to Answering Questions about Images,Exploring Models and Data for Image Question Answering"]()) encoded both an image and the question using an LSTM and decoded an answer.
We are not aware of QA methods addressing the temporal domain.

$\mathbf{QA Datasets}$
Most available datasets focus on image (["What are you talking about? Text-to-Image Coreference,Microsoft COCO: Common Objects in Context,Flickr30k,Adopting abstract images for semantic scene understanding"]()) or video description (["Collecting highly parallel data for paraphrase evaluation,A Dataset for Movie Description,A thousand frames in just a few words: Lingual description of videos through latent topics and sparse object stitching"]()).  Particularly relevant to our work is the MovieDescription dataset (["A Dataset for Movie Description"]()) which transcribed text from the Described Video Service (DVS), a narration service for the visually impaired, for a collection of 100 movies.

For QA,  (["A Multi-World Approach to Question Answering about Real-World Scenes based on Uncertain Inpu"]()) provides questions and answers (mainly lists of objects, colors, \etc) for the NYUv2 RGB-D dataset, while (["VQA,Visual Madlibs: Fill in the blank Image Generation and Question  Answering"]()) do so for MS-COCO with a dataset of a million QAs.
While these datasets are unique in testing the vision algorithms in performing various tasks such as recognition, attribute induction and counting, they are inherently limited to static images.
In our work, we collect a large QA dataset of about 300 movies with challenging questions that require semantic reasoning over a long temporal domain.

Our dataset is also related to purely text-datasets such as MCTest (["Mctest: A challenge dataset for the open-domain machine comprehension  of text"]()) which contains 660 short stories with multi-choice QAs, and (["Teaching Machines to Read and Comprehen"]()) which converted 300K news summaries into Cloze-style questions.
We go beyond these datasets by having significantly longer text, as well as multiple sources of available information (video clips, plots, subtitles, scripts and DVS).
This makes our data one of a kind.

# MovieQA dataset

The goal of our paper is to create a challenging benchmark that evaluates semantic understanding over long temporal data.
We collect a dataset with very diverse sources of information that can be exploited in this challenging domain.
Our data consists of quizzes about movies that the automatic systems will have to answer.
For each movie, a quiz comprises of a set of questions, each with 5 multiple-choice answers, only one of which is correct.
The system has access to various sources of textual and visual information, which we describe in detail below.

We collected 294 movies with subtitles, and obtained their extended summaries in the form of plot synopses from $\mathbf{Wikipedia}$.
Additionally, we crawled \emph{imsdb} for scripts, which were available for 40\% (117) of our movies.
A fraction of our movies (45) come from the MovieDescription dataset (["A Dataset for Movie Description"]()) which contains movies with DVS transcripts.

$\mathbf{Plot synopses}$
Plot synopses are extended movie summaries that fans write after watching the movie.
This makes them faithful to the story that takes place in the movie.
Synopses widely vary in detail, and range from one to 20 paragraphs, but focus on describing content that is directly relevant to the story.
They rarely contain detailed visual information such as how a character looks or dresses, and mainly focus on describing the movie events, character interactions and at times emotions.
Plots are thus in many ways what a perfect automatic algorithm should get from "watching" the movie.
We exploit such plots to gather our quizzes.

$\mathbf{Videos and subtitles}$
An average movie is about 2 hours in length and has over 198K frames and almost 2000 shots.
Note that on its own, video contains information about e.g., who did what to whom, but does not contain sufficient information to explain why something happened.
Dialogs play an important role, and only both modalities together allow us to fully understand the story.
Note that subtitles do not contain speaker information.

$\mathbf{DVS}$ is a service that narrates movie scenes to the visually impaired by inserting relevant descriptions in between dialogs. These descriptions contain sufficient "visual" information about the scene that allows the visually impaired audience to follow the movie.
DVS is thus a proxy for a perfect vision system, and potentially allows quizzes to be answered without needing to process the videos.

$\mathbf{Scripts}$
The scripts that we collected are written by screenwriters and serve as a guideline for movie making.
They typically contain detailed descriptions of scenes, and, unlike subtitles, contain both dialogs and speaker information.
Scripts are thus similar, if not richer in content to DVS+subtitles, however are not always entirely faithful to the movie as the director may aspire to artistic freedom.


## QA Collection method}


Since videos are difficult and expensive to provide to annotators, we used plot synopses as a proxy for the movie.
Thus, while creating quizzes, our annotators were only looking at text and were "forced" to ask questions that are at a higher semantic level and more story-like.
In particular, we split our annotation efforts into two parts to ensure high quality of the collected data.

$\mathbf{Q and correct A}$
Our annotators were first asked to select a movie from a provided list, and were then shown its plot synopsis one paragraph at a time.
For each paragraph, the annotator had the freedom of forming any number and type of questions. On average they formed 5.4 questions per paragraph.
Each annotator was also asked to provide the correct answer, and was additionally required to mark a minimal set of sentences within the plot synopsis paragraph which are needed to both frame the question and answer it.
This acts as ground-truth for localizing the QA in the plot.

In our instructions, we asked the annotators to provide context to each question, such that the person taking their quiz would be able to answer it by watching the movie alone (without having access to the synopsis).
The purpose of this was to ensure questions that are localizable in the video and story as opposed to generic questions such as "What are they talking about?".
We trained our annotators for about one to two hours and gave them the option to re-visit and correct their data.
We paid them by the hour, a strategy that allowed us to collect more thoughtful and complex QAs, rather than short questions and single-word answers.

$\mathbf{Multi-choice}$
In the second step of data collection, we collected multiple-choice answers for each question.
Our annotators were shown a paragraph and a question at a time, but not the correct answer.
They were then asked to answer the question correctly as well as to provide 4 wrong answers.
These answers were either deceiving facts from the same paragraph or common-sense answers.
The annotator was also allowed to re-formulate or correct the question.
We used this to sanity check  all the questions received in the first step, and was one of the main reasons as to why we split our data collection into two phases.

$\mathbf{Time-stamp to video}$
Parallel to our movie QA collection, we asked in-house annotators to align each sentence in the plot synopsis to video, by marking the beginning and end (in seconds) in the video that the sentence describes.
Long and complicated sentences were often aligned to multiple, non-consecutive video clips.
Annotation took roughly 2 hours per movie.
Since we have each QA aligned to a sentence (or multiple ones) in the plot synopsis, this alignment provides QA time-stamped with corresponding video clips.
We will provide these clips as part of our benchmark.

## Dataset Statistics

In the following, we present some statistics about the questions and answers  in our MovieQA dataset.
Table~\ref{tab:dataset-comparison} presents an overview of popular and recent Question-Answering datasets in the field.
Most datasets (except MCTest) use very short answers and are thus limited to covering simpler visual / textual forms of understanding.
To the best of our knowledge, our dataset is also the first to use videos in the form of movies.

$\mathbf{Multi-choice QA}$
So far, we have collected a total of 7702 QAs about 294 movies%
\footnote{We are currently working on increasing the number of QAs to 15k. We will update the tables and make the data publicly available soon.}.
Each question comes with one correct and four deceiving answers.
Table~\ref{tab:qa_stats} presents an overview of the dataset along with the information about the train/test splits, which will be used to train and evaluate the automatic QA models.
Unlike most previous datasets, our questions and answers are fairly long and have on average about 9 and 5 words, respectively.
We create a video-based answering split for our dataset, currently based on the number of movies we have collected with plot synopses alignment.
Note that the QA methods needs to look at a long video clip ($\sim$150 seconds) to answer the question.

Fig.~\ref{fig:stats:qword_calength} presents the number of questions (as area) depending on the first word of the question.
We see the diversity among questions and the number of words used to answer them.
"Why" questions require verbose answers which is justified by having the largest average number of words in the correct answer.
On the other hand, Does answers are very short ("Yes", or "No, he killed John"), while the question itself needs to describe a lot of things to pinpoint to a particular part of the story.

A different way to look at QAs is to decide their type based on the answer.
For example, especially, "What" questions can cover a large variety in types of answers ("What happens ...", "What did X do?", "What is the name ...", "What is X's purpose?", \etc).
In Fig.~\ref{fig:stats:answer_type} we show the questions from our dataset in a variety of answers.
In particular, reasoning based questions (\cf~top half of the pie) are a large part of our data.
In the bottom left quadrant we see typical question types which can likely be answered using vision alone. Note however, that even the reasoning questions typically require vision, as the question provides context which is typically a visual description of a scene (\eg, "When John runs after Marry...").

$\mathbf{Text sources for answering}$
Finally, Table~\ref{tab:textsrc_stats} presents different statistics of the various text sources.
For plot synopses, we see that the average number of words per sentence stands above all other forms of text which speaks for the richness of the descriptions.



# Multi-choice Question-Answering

We investigate a number of intelligent baselines for question-answering ranging from very simple ones to more complex architectures, building on the recent work on automatic QA.
We also study inherent biases in the data and try to answer the quiz based simply on characteristics such as word length or within answer diversity.


% general formulation
Formally, let $S$ denote the story, which can take the form of any of the available sources of information -- \eg~plots, subtitles, or video shots.
Each story $S$ has a set of questions, and we assume that the (automatic) student reads one question $q^S$ at a time.
%Let $\mathcal{Q}^S = \{q_i^S\}$ be a set of questions obtained from a story $S$.
%Since we will assume that the questions are independent, we drop the subscript and only consider a question $q^S$ to be answered.
%Let $\mathcal{A}_i^S = \{a_{ij}^S\}_{j=1}^{M}$ be the set of multiple choice answers (only one of which is correct) corresponding to $q_i^S$.
Let $\{a_{j}^S\}_{j=1}^{M}$ be the set of multiple choice answers (only one of which is correct) corresponding to $q^S$, with $M=5$ in our dataset.

The general problem of multi-choice question answering can be formulated by a three-way scoring function $f(S,q^S,a^S)$. This function evaluates the "quality" of the answer given the story and the question.
Our goal is thus to pick the correct answer $a^S$ for $q^S$ that maximizes $f$:
\begin{equation}
j^* = \arg\max_{j=1\ldots M} f(S, q^S, a_{j}^S) \,
\end{equation}
%Here $f(\cdot, \cdot, \cdot)$ uses information from the story $S$ to select an answer among $a_{j}^S$ for the question $q^S$.
We next discuss different possibilities for $f$. We drop the superscript $(\cdot)^S$ for simplicity of notation.

## The Hasty Student

%We now discuss various formulations of $f$ representing different ways to answer the questions.
We first consider $f$ which ignores the story and attempts to answer the question directly based on  latent biases and similarities.
We call such a baseline as the "hasty student" since he/she does not care to read/watch the actual story.

The extreme case of a hasty student is to try and answer the question by only looking at the answers.
Here, $f(S, q, a_{j}) = g_{H1}(a_{j}|$\mathbf{a})$, where $g_{H1}(\cdot)$ captures some properties of the answers.


$\mathbf{Answer length}$
We use the number of words in the multiple choices to select the correct answer.
This idea explores the bias in the data where the number of words in the correct answer is slightly larger than the number of words in wrong answers.
We choose the correct answer by:
(i) selecting the longest answer;
(ii) selecting the shortest answer; or
(iii) selecting the answer with the most different length.

$\mathbf{Within answer similarity/difference}$
While still looking only at the answers, we compute a distance between all answers based on their representations (discussed in Sec.~\ref{sec:representation}).
We then select the correct answer as either the most similar or most distinct among all answers.

$\mathbf{Q and A similarity}$
We now consider a hasty student that looks at both the question and answer,  $f(S, q, a_j) = g_{H2}(q, a_{j})$.
We compute similarity between the question and each answer and pick the most similar answer.


## The Searching Student

While the hasty student ignores the story, we consider a student that tries to answer the question by trying to locate a subset of the story $S$ which is most similar to both the question and the answer.  The similarity of the question and the answer is ignored in this case.

The scoring function $f$  is thus factorized into two parts:
\begin{equation}
f(S, q, a_{j}) = g_I(S, q) + g_I(S, a_{j}) \, .
\end{equation}
%In particular, when $S$ is a story composed of $L$ different sentences (or shots, or dialogs), we can attempt to find the best set of sentences, observed $H$ at a time, as:
We use two possible similarity functions: a simple cosine similarity defined over a window, and one using a neural architecture. We describe these next.

$\mathbf{Cosine similarity with a sliding window.} We aim to find the best window of $H$ sentences (or shots) in $S$ that maximize similarity between the story and the question, and the story and the answer. We define our similarity function:% follows:
\begin{equation}
f(S, q, a_{j}) = \max_l \sum_{k = l}^{l+H} g_{ss}(s_k, q) + g_{ss}(s_k, a_{j}) \, ,
\end{equation}
where $s_k$ denotes a sentence (or shot) from the story $S$. We use $g_{ss}(s, q) = x(s)^T x(q)$ as a dot product between the (normalized) representations of the two sentences (shots). We discuss these representations in detail in Sec.~\ref{sec:representation}.

$\mathbf{Searching student with a convolutional brain (SSCB).}
Instead of factoring $f(S, q, a_{j})$ as a fixed (unweighted) sum of two similarity functions $g_{I}(S, q)$ and $g_{I}(S, a_{j})$, we build a neural network that learns such a function. Assuming the story $S$ is of length $n$, \eg~$n$ sentences in the plot, or $n$ shots in the video clip, $g_{I}(S, q)$ and $g_{I}(S, a_{j})$ can be seen as two vectors of length $n$. The $k$-th entry in e.g., the former vector is $g_{ss}(s_k, q)$.
We further combine all $[g_I(S, a_{j})]_j$ for the $5$ answers into a $n\times 5$ matrix. We then replicate the vector $g_{I}(S, q)$ $5$-times, and stack the question and answer matrix together to obtain a tensor of size $n \times 5 \times 2$.

Our neural similarity model is a convolutional neural net (CNN), shown in Fig.~\ref{fig:model:cnn}, that takes this tensor, and several layers of $1 \times 1$ convolutions to approximate a family of functions $\phi(g_I(S, q), g_I(S, a_{j}))$.
We also add a max pooling layer with kernel size $3$ to allow for scoring the similarity within a window in the story.
The last convolutional output is a matrix of size $ \frac{n}{3} \times 5$, and we apply both mean and max pooling across the storyline, add them, and make predictions using the softmax. We train our network on a randomized train/val split of our training set using cross-entropy loss and Adam optimizer (["Adam: A method for stochastic optimization"]()).

## Memory Network for Complex QA

Memory Networks were originally proposed specifically for QA tasks and model complex three-way relationships between the story, question and an answer.
We briefly describe MemN2N proposed by (["End-To-End Memory Networks"]()) and suggest simple extensions to make it suitable for our data and task.

The original MemN2N takes a story and a question related to it.
The answering is restricted to single words and is done by picking the most likely word from the vocabulary $\mathcal{V}$ of 20-40 words.
This is not directly applicable to our domain, as our data set does not have a fixed set of answers.

A question $q$ is encoded as a vector $u \in \mathbb{R}^d$ using a word embedding $B \in \mathbb{R}^{d \times |\mathcal{V}|}$.
Here, $d$ is the embedding dimension, and $u$ is obtained by mean-pooling  the representations of words in the question.
Simultaneously, the sentences of the story $s_l$ are encoded using word embeddings $A$ and $C$ to provide two different sentence representations $m_l$ and $c_l$, respectively.
Here, $m_l$, the representation of sentence $l$ in the story, is used in conjunction with $u$ to produce an attention-like mechanism which selects sentences in the story most similar to the question via a softmax function:
\begin{equation}
p_l = \mathrm{softmax}(u^T m_l) \, .
\end{equation}
The probability $p_l$ is used to weight the second sentence embedding $c_l$, and the story representation $o = \sum_l p_l c_l$ is obtained by pooling the weighted sentence representations across the story.
Finally, a linear projection $W \in \mathbb{R}^{|\mathcal{V}| \times d}$ decodes the question $u$ and the story representation $o$ to provide a soft score for each vocabulary word
\begin{equation}
a = \mathrm{softmax}(W (o + u)) \,,
\end{equation}
and finds the answer $\hat a$ as the top scoring word. The free parameters to train %in the MemN2N
are the $B$, $A$, $C$, and $W$ embeddings for different words which can be shared across different layers.

Due to its fixed set of output answers, the MemN2N in the current form is not designed for multi-choice answering with open, natural language answers.
We propose two key modifications to make the network suitable for our task.


$\mathbf{Memory Network for natural language answers}$
To allow the Memory Network to rank multiple answers written in natural language, we can add an additional embedding layer $F$ which maps each multi-choice answer $a_j$ to a vector $g_j$.
Note that $F$ is similar to previous word embeddings $B$, $A$ and $C$, but operates on answers instead of question and story respectively.
To predict the correct answer, we compute the similarity between the answers $g$, the embedding $u$ of the question and the story representation $o$:
\begin{equation}
\label{eq:memnet_multichoice_ans}
a = \mathrm{softmax}((o + u)^T g)
\end{equation}
and simply pick the most probably answer as correct. In our general QA formulation, this is equivalent to
\begin{equation}
f(S, q, a_{j}) = g_{M1}(S, q, a_{j}) + g_{M2}(q, a_{j}),
\end{equation}
that is, a function $g_{M1}$ that considers the story, question and answer, and a second function $g_{M2}$ that directly considers similarities between the question and the answer.


$\mathbf{Weight sharing and fixed word embeddings}$
The original MemN2N learns embeddings for each word based directly on the task of question-answering.
However, to scale this to large vocabulary data sets like ours, this requires unreasonable amounts of training data.
For example, training a model with vocabulary size 12000 (obtained from plot synopses) and $d = 100$ would entail learning 1.2M parameters for each embedding.
To prevent overfitting, we can share all word embeddings $B, A, C, F$ of the memory network.
Nevertheless, this is still a large amount of parameters.

We make the following crucial modification that allows us to use the Memory Network for our dataset. We drop $B$, $A$, $C$, $F$ and replace them by a fixed (pre-trained) word embedding $Z \in \mathbb{R}^{d_1 \times |\mathcal{V}|}$ obtained from the Word2Vec model and learn a shared linear projection layer $T \in \mathbb{R}^{d_2 \times d_1}$ to map all sentences (stories, questions and answers) into a common space.
Here, $d_1$ is the dimension of the Word2Vec embedding, and $d_2$ is the projection dimension.
Thus, the new encodings are
\begin{equation}
u = T \cdot Z q, \, m_l = T \cdot Z s_l, \, \mathrm{and} \, g_j = T \cdot Z a_j .
\end{equation}
Answer prediction is performed as before in Eq.~\ref{eq:memnet_multichoice_ans}.

We initialize the projections either using an identity matrix $d_1 \times d_1$ or using PCA to lower the dimension from $d_1 = 300$ to $d_2 = 100$.
Training is performed using stochastic gradient descent with a batch size of 32 for plots and DVS.
For subtitles and scripts we needed to use a batch size of 16 to ensure that the story data fits in our 6GB Titan Black GPU memory.


## Representations for Text and Video}


$\mathbf{TF-IDF}$ is a popular and successful feature in information retrieval.
In our case, we treat plots (or the other forms of text) of different movies as documents and compute a weighting for each word.
We set all words to lower case, use stemming and compute the vocabulary $\mathcal{V}$ which consists of all words $w$ in the documents. % that appear at least $c$ times.
We represent each sentence (or question or answer) as a bag-of-words  weighted with an TF-IDF score for each word.
%For a sentence $k$ coming from document $d$ we compute its representation $x(s_k) = \sum_{w \in s_k} d_w$ by summing over its words, where
%$d_w \in \mathbb{R}^{|\mathcal{V}|}$ is a vector of all zeros except at the location of the word $w$ and is weighted by the TF-IDF score for the word.
%We treat questions and answers in a similar way and represent them using the document to which they belong.

$\mathbf{Word2Vec}$
A disadvantage of TF-IDF is that it is unable to capture the similarities between words.
We use the skip-gram model proposed by (["Efficient estimation of word representations in vector space"]()) and train it on roughly 1200 movie plots to obtain domain-specific, $300$ dimensional word embeddings.
A sentence is then represented by mean-pooling its word embeddings.
%A story sentence, question, or answer is now represented as the sum over the embeddings for each word within it.
We normalize the resulting vector to have unit norm.

$\mathbf{SkipThoughts}$
While the sentence representation using mean pooled Word2Vec discards word order, SkipThoughts (["Skip-Thought Vectors"]()) use a Recurrent Neural Network to capture the underlying sentence semantics.
We use the pre-trained model by (["Skip-Thought Vectors"]()) to compute a $4800$ dimensional sentence representation. %, a question and answer.

To answer questions from the video, we learn an embedding between a shot and a sentence, which maps the two modalities in a common space. In this joint space, one can score the similarity between the two modalities via a simple dot product. This allows us to apply all of our proposed question-answering techniques in their original form.

To learn the joint embedding we follow (["Aligning Books and Movies: Towards Story-like Visual Explanations by  Watching Movies and Reading Books"]()) which extends (["Unifying Visual-Semantic Embeddings with Multimodal Neural Language  Models"]()) to video.
Specifically, we use the GoogLeNet architecture (["Going deeper with convolutions"]()) as well as hybrid-CNN (["Learning Deep Features for Scene Recognition using Places Database"]()) for extracting frame features, and mean-pool the representations over all frames of a shot.
The embedding is then a linear mapping of the shot representation and an LSTM on word embeddings on the sentence side.
The model evaluates the dot product of mapped vectors on both sides using the ranking loss.
We train the embedding on the MovieDescription Dataset (["A Dataset for Movie Description"]()) as in (["Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books"]()).

# QA Evaluation

We present results for question-answering with the proposed intelligent baselines on our MovieQA dataset.
We study how various sources of information influence the performance, and how different level of complexity encoded in $f$ affects the quality of automatic QA.

$\mathbf{Protocol}$
Note that we have two primary tasks for evaluation.
(i) $\mathbf{Text-based}: where the story is represented with plots, subtitles, scripts and/or DVS; and
(ii) $\mathbf{Video-based}: which uses video and dialogs (subtitles).
For each task,  the train and test split statistics are presented in Table~\ref{tab:qa_stats}.
We will provide more details on the project page with the release of our dataset.

$\mathbf{Metrics}$
Multiple choice QA leads to simple and objective evaluation.
We measure accuracy as the number of questions where an automatic model chooses the correct answer over the total number of questions.

In addition to accuracy, we propose to use another metric "Quiz Score" (QS) inspired by real-world multiple-choice examinations.
This metric penalizes students for choosing wrong answers and also (albeit by a smaller amount) for unanswered questions.
Similar to the concept of "refuse to predict" schemes, we want to stress that it might be better to leave answers blank (say "I don't know") than pick the wrong answer.
We plan to use this scoring scheme in the leader board rankings for the benchmark.

The score is computed as
\begin{equation}
\text{Quiz Score} = 100\cdot \frac{\text{\#CA} - 0.25 \cdot \text{\#WA} - 0.05 \cdot \text{\#UA}}{\text{Total no. of questions}} \, . \nonumber
\end{equation}
CA, WA and UA stand for \underline{c}orrect, \underline{w}rong and \underline{u}nanswered questions respectively.

% We discuss some extreme cases to analyze this metric:\\
% \noindent(i) No question is answered leads to $\text{QS} = -5$.\\
% \noindent(ii) All questions answered incorrectly, $\text{QS} = -25$.\\
% \noindent(iii) All questions answered at random, $\text{QS} = 0$, since each QA has 5 multiple choice options.\\
% \noindent(iv) All QAs answered correctly gives $\text{QS} = 100$.\\
% \noindent(v) Finally, the penalty for not attempting QAs requires that more than 20\% QAs are answered correctly to obtain an overall QS above 0.
% For example, in a test of 26 questions, the student needs 2 correct answers out of the attempted 6 (increasing the ratio to 1/3 instead of 1/5) to score 0.


$\mathbf{Answering to maximize Quiz Score}$
An easy way to decide which questions are not worth attempting (leave unanswered) is to learn a threshold on a subset of the training set.
We learn a threshold on the difference between the top 2 highest scoring options via grid search, by optimizing for the Quiz Score as the metric.
The difference in score between the top 2 options can be considered as our model confidence in answering questions correctly.
We then use the learned threshold on the test set to decide whether a question should be answered.


## Hasty Student

The first part of Table~\ref{tab:QS-results-cheating_baseline} shows performance of the three models when trying to answer questions based on the length of the answers.
Selecting the longest answer performs better (28.2\%) than random (20\%) while the answer with the most different length is only slightly better at 22.6\%.
The second part of Table~\ref{tab:QS-results-cheating_baseline} presents results when using feature-based similarity within answers.
We see that the most similar answer is likely to be correct when the representations are generic and try to capture the semantics of the sentence (Word2Vec, SkipThoughts).
On the other hand, when using TF-IDF, discriminating between different names is very easy and thus the most distinct answer is likely to be more correct.
Finally, in the last part of Table~\ref{tab:QS-results-cheating_baseline} we see that questions and answers are very different from each other.
Especially, TF-IDF performs worse than random since words in the question rarely appear in the answer.

Performance of the methods using our second metric "Quiz Score" is indicated by numbers in paranthesis in the Table~\ref{tab:QS-results-cheating_baseline}.
We see the bias towards longer answers results in the highest QS.
More interestingly, while the difference between accuracy for within-answer similarity and answer length is not high (27.0\% vs. 28.2\%), the large difference in QS (8.5 vs. 18.7) reveals that answer length is a more confident way to predict answers.
Most other methods result in a quiz score close to 0.

## Hasty Turker
To analyze the quality of our collected multi-choice answers and their deceiving nature, we tested humans (via AMT) on a subset of 200 QAs.
The turkers were not shown the story in any form and were asked to pick the best possible answer given the question and a set of options.
The purpose of this experiment is to analyze whether our multi-choice answers are difficult enough, so as to even deceive humans when provided with no context.
We asked each question to 10 turkers, and rewarded each with a bonus if their answer agreed with the majority.
%Note that some questions reveal the source of the movie due to names (e.g., "Darth Vader") or places in the questions and answers.

% Fig.~\ref{fig:hasty_turker} shows the results of this experiment.
% We see that without access to the story, humans are able to pick the correct answer with an accuracy of 27.6\%.
% This bias may likely be due to the fact that some of the QAs reveal the movie due to names (e.g., "Darth Vader") and the turker may have seen  this movie.
% Interestingly, we see that for 12.0\% of all questions, the correct answer was not picked by any of the 10 turkers.
% This shows the genuine difficulty of our task.

The results are presented in Fig.~\ref{fig:hasty_turker_detail}.
The \emph{overall accuracy} is computed as the number of all correct answers over all annotators.
We also compute \emph{accuracy of majority vote}, which is the number of times a correct answer was chosen by the majority of the turkers divided by the total number of questions.
Finally, \emph{Q with corr. answ. never picked} is the percentage of questions for which none of the turkers selected the correct answer.

In Fig.~\ref{fig:hasty_turker_detail}~(a) we see that 27.6\% of all answers were correct, and 37\% questions got a correct answer via the majority vote.
Since some of the questions and answers reveal the identity of the movie (\eg~a reference to "Darth Vader", "Indiana Jones", "Borat"), we decided to also select a subset of these questions for which the names did not necessarily indicate a movie.
This removed the possibility of an annotator actually remembering the movie while answering the question.
We present the results of this experiment (evaluated on 135 QAs) in Fig.~\ref{fig:hasty_turker_detail}~(b).
While the overall accuracy is closer to random, it is still slightly higher (24.7\% overall accuracy and 30.4\% by majority vote).
This may indicate that some of the wrong answers are somewhat correlated, making the test slightly easier for a human.
It also indicates that a machine which takes into account all answers should likely do better than looking at each answer in isolation.

The small bias of answer length in our dataset was not noticed by the turkers.
31.3\% of the annotators chose the longest answer as the correct one, and in fact 37.3\% of them picked the shortest answer.


## Searching Student

$\mathbf{Cosine similarity in window}$
The first three rows of Table~\ref{tab:QS-results-comprehensive_baseline} present results of the proposed method using different representations and input story types.
% For TF-IDF we consider two settings depending on the vocabulary size which is controlled by the number of times a word occurs in the documents seen together.
Using the plot to answer questions outperforms other information sources such as subtitles, scripts or DVS.
This is most likely due to the fact that the data was collected using plot synopses and while framing the QAs annotators often reproduce parts of the plot verbatim.

We show the results of using Word2Vec or SkipThought representations in the following rows of Table~\ref{tab:QS-results-comprehensive_baseline}.
Both perform significantly worse than the TF-IDF representation and Word2Vec is consistently better than SkipThoughts.
We suspect that while Word2Vec and SkipThoughts are good at capturing the overall semantic structure of the words and sentences respectively, but proper nouns -- names, places -- are often hard to distinguish.
This is more evident as we move from individual word representations (Word2Vec) towards the sentence representation (SkipThoughts) which is then likely to ignore the subtleties between different names.



Fig.~\ref{fig:results-simple_baselines_qfw} presents a breakup of the overall accuracy based on the first word of the questions.
The story here is the plot synopsis and answering method employed is the searching student with cosine similarity.
While TF-IDF works better on all question types, the difference between TF-IDF with respect to the semantic representations is extremely high when answering questions of type "Who" and "Where".
On "Why" and "How", we see a more gradual decay in performance.


$\mathbf{Influence of window}$
We notice that the window size $H$ significantly influences the results of using TF-IDF based representations on stories of subtitles and scripts.
We believe that this results from two factors:
(i) the questions are about the story and answering them by just looking at one dialog is a very hard task; and
(ii) the TF-IDF representation in particular sees more words which directly makes matching less sparse and easier.

We analyze the case of using subtitles as stories and show the variation in accuracy in Fig.~\ref{fig:subtt-window-fw-acc}.
Each subtitle, on average, corresponds to 4.74 seconds of video.
The figure shows that the performance improves strongly up to a window of size 100 -- which corresponds to about 8 minutes of video -- and then shows small improvement thereafter.

## Search student with convolutional brain

$\mathbf{SSCB}$. The middle rows of Table~\ref{tab:QS-results-comprehensive_baseline} show the result of our neural similarity model.
Here we also tried to combine all text features (\textit{SSCB fusion}) via our CNN.
We randomly split the training set into $80\%$ train / $20\%$ val, keeping all questions / answers of the same movie in the same split, and train our model on train and monitor performance on val.
During training, we also create several model replicas and pick the ones with the best validation performance.

Table~\ref{tab:QS-results-comprehensive_baseline} shows that the neural model outperforms the simple cosine similarity on most tasks, while the fusion method achieves the highest performance on two out of four story types.
%This is likely because the fusion CNN consists of 3 times as many parameters and does not generalize well  when trained with $45$ movies with DVS.
Overall, the accuracy is capped at $35\%$ for most modalities showing the difficulty of our dataset.


## Memory network}

The original MemN2N which allows to train the word embeddings overfits strongly on our dataset leading to a test error near random performance ($\sim$20\%).
However, our modifications help in restraining the learning.
Table~\ref{tab:QS-results-comprehensive_baseline} presents results for MemN2N with Word2Vec initialization and a linear projection layer.
Using plot synopses, we see a performance similar to SSCB on Word2Vec features.
However, with longer stories, the attention mechanism in the network is able to sift through thousands of story sentences and perform well on DVS, subtitles and scripts. This shows that complex three-way scoring functions are needed to tackle such complex QA sources.
In terms of modalities, the network performs best for scripts which contain the most information (descriptions, dialogs and speaker information). %On both, the DVS which is a proxy for vision, and subtitles containing the dialogs, the method performs similarly


## Video baselines


We now evaluate two of our best performing QA models, SSCB and MemN2N, on the split of our data that has video.
We evaluate two settings: answering questions by "watching" the full movie, or via the ground-truth video clips (time-stamped sentences from the plot to which the question/answer refers to).
The results are shown in Table~\ref{tab:QS-results-video_baseline}.

Since visual information alone is insufficient to answer high level semantic questions we also combine video and dialog (subtitles).
We encode each subtitle as before using Word2Vec.
%We compare the performance of considering the story made up of video shots only, subtitles dialogs only, or a fusion of the two.
For SSCB we perform late fusion of the CNNs for the two modalities.
For the memory network we create two branches, one for each modality, and sum up the scores before the final softmax. We train the full model jointly.
%The results are shown in Table~\ref{tab:results-video_baseline}.




# Conclusion

We introduced the MovieQA data set which aims to evaluate automatic story comprehension from both video and text.
The dataset currently stands at 7702 multiple choice questions from 294 movies with high semantic diversity.
Our dataset is unique in that it contains several sources of information -- full-length movies, subtitles, scripts, plots and DVS.
We provided several intelligent baselines and extend existing QA techniques to analyze the difficulty of our task. %to show that question-answering with such open-ended semantics is extremely hard.
%We plan to create a benchmark with an active leader board, encouraging inspiring work in this challenging domain.


Owing to the variety in information sources, our data set is applicable to Vision, Language and Machine Learning communities.
We evaluate a variety of answering methods which discover the biases within our data and demonstrate the limitations on this high level semantic task.
Current state-of-the-art methods do not perform well and are often only a little better than random.
Using this data set we will create an evaluation campaign that can help breach the next frontier in improved vision and language understanding.
