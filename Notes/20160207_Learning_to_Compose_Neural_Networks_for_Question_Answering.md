## [Learning to Compose Neural Networks for Question Answering](https://github.com/jacobandreas/nmn2)

Code : [Dynamically predicted neural network structures for multi-domain question answering](https://github.com/jacobandreas/nmn2.git)


# abstract

We describe a question answering model that applies to both images and structured knowledge bases.
The model uses natural language strings to automatically assemble neural networks from a collection of composable modules. Parameters for these modules are learned jointly with network-assembly parameters via reinforcement  learning, with only (world, question, answer) triples as supervision. Our approach, which we term a <span style="color:green">$\mathbf{dynamic neural module network}$</span>,  achieves state-of-the-art results on benchmark datasets in both  visual and structured domains.

# Introduction}

This paper presents a compositional, attentional model for answering questions about a variety of world representations, including images and structured knowledge bases.

The model translates from questions to dynamically assembled neural networks, then applies these networks to world representations (images or knowledge bases) to produce answers. We take advantage of two largely independent lines of work: on  one hand, an extensive literature on answering questions by mapping from strings to logical representations  of meaning; on the other, a series of recent successes in deep neural models for image recognition and captioning. By constructing neural networks instead of logical forms, our model leverages the best aspects of both linguistic compositionality and continuous
representations.

Our model has two components, trained jointly:
1. first, a collection of neural "modules" that can be freely composed
2. second, a network layout predictor  that assembles modules into complete deep networks tailored to each question.


Previous work has used manually-specified modular structures for visual learning (["Deep compositional question answering with neural module networks"]()). Here we:

1. <span style="color:green">$\mathbf{learn}$</span> a network structure predictor jointly with module parameters themselves
2. <span style="color:green">$\mathbf{extend}$</span> visual primitives from previous work to reason over structured world representations

Training data consists of (world, question, answer) triples: our approach requires no supervision of network layouts.
We achieve state-of-the-art performance on two markedly different question answering tasks: one with questions about natural images, and another with more compositional questions about United States geography.


# Deep networks as functional programs}

We begin with a high-level discussion of the kinds of composed networks we would like to learn.

(["Andreas15NMN"]()) describe a heuristic approach for decomposing visual question answering tasks into sequence of modular sub-problems. For example,
the question <span style="color:green">$\mathbf{What color is the bird?}$</span> might be answered in two steps: first, "where is the bird?" (\autoref{fig:examples}a), second, "what color is that part of the image?"

This first step, a generic \textbf{module} called <span style="color:green">$\mathbf{find}, can be expressed as a fragment of a neural
network that maps from image features and a lexical item (here <span style="color:green">$\mathbf{bird}$</span>) to a distribution over pixels.
This operation is commonly referred to as the <span style="color:green">$\mathbf{attention mechanism}, and is a standard tool for
manipulating images (["Show, attend and tell: neural image caption generation with visual  attention"]()) and text representations (["Teaching machines to read and comprehend"]())

The first contribution of this paper is an extension and generalization of this mechanism to enable fully-differentiable reasoning about more structured semantic representations.

Figure1. b shows how the same module can be used to focus on the entity <span style="color:green">$\mathbf{Georgia} in a non-visual grounding domain; more generally, by representing every entity in the
universe of discourse as a feature vector, we can obtain a distribution over entities that corresponds roughly to a logical set-valued denotation.

Having obtained such a distribution, existing neural approaches use it to immediately compute a weighted average of image features and project back into a labeling decision---a


## module
But the logical perspective suggests a number of novel modules that might operate on attentions: e.g. combining them (by analogy to conjunction or disjunction) or inspecting them directly without a return to feature space (by analogy to quantification,Figure1. b).

Unlike their formal counterparts, they are differentiable end-to-end, facilitating their integration into learned models. Building on previous work, we learn behavior for a collection of heterogeneous modules from (world, question, answer) triples.


The second contribution of this paper is a model for learning to assemble such modules compositionally. Isolated modules are of limited use---to obtain expressive power comparable to either formal approaches or monolithic deep networks, they must be composed into larger structures. \autoref{fig:examples} shows simple examples of composed structures, but for
realistic question-answering tasks, even larger networks are required.
Thus our goal is to automatically induce variable-free, tree-structured computation descriptors.
We can use a familiar functional notation from formal semantics (e.g. Liang et al., 2011) to represent these computations.\footnote{But note that unlike formal semantics, the behavior of the primitive
functions here is itself unknown.} We write the two examples in Figure.2 as

```
(describe[color] find[bird])
```

and

```
(exists find[state])
```

respectively. These are $\textbf{network layouts}$: they specify a structure for arranging modules (and their lexical parameters) into a complete network. (["Deep compositional question answering with neural module networks"]()) use hand-written rules to deterministically  transform dependency trees into layouts, and restricted to producing simple structures like the above for non-synthetic data. For full generality, we will need to solve harder problems, like transforming <span style="color:green">$\mathbf{What \ cities \ are \ in \ Georgia?}$</span>

```
    (and
        find[city]
        (relate[in] lookup[Georgia]))
```

In this paper, we present a model for learning to select such structures from a set of automatically generated candidates. We call this model a $\textbf{dynamic neural module network}$.


# Related work

There is an extensive literature on database question answering, in which strings are mapped to logical forms, then evaluated by a black-box execution model to produce answers. Supervision may be provided either by annotated logical forms (["Learning synchronous grammars for semantic parsing with lambda calculus,Inducing probabilistic {CCG} grammars from logical form with higher-order unification,Semantic parsing as machine translation"]()) or from (world, question, answer) triples alone (["Learning dependency-based compositional semantics,Compositional semantic parsing on semi-structured tables"]()). In general the set of primitive functions from which these logical forms can be assembled is fixed, but one recent line of work focuses on inducing new predicates functions automatically, either from perceptual features (["Jointly learning to parse and perceive: connecting natural language to the physical world"]()) or the underlying schema (["Scaling semantic parsers with on-the-fly ontology matching"]()). The model we describe in this paper has a unified framework for handling both the perceptual and schema cases, and differs from existing work primarily in learning a differentiable execution model with continuous evaluation results.

Neural models for question answering are also a subject of current interest. These include approaches that model the task directly as a multiclass classification problem  (["A neural network for factoid question answering over paragraphs"]()), models that attempt to embed questions and answers in a shared vector space (["Question answering with subgraph embeddings"]()) and attentional models that select words from documents sources (["Teaching machines to read and comprehend"]()). Such approaches generally require that answers can be retrieved directly based on surface linguistic features, without requiring intermediate computation. A more structured approach described by (["Neural enquirer: Learning to query tables"]()) learns a query execution model for database tables without any natural language component. Previous efforts toward unifying formal logic and representation learning include those of (["Towards a formal distributional semantics: Simulating logical calculi with tensors"]()) and (["Vector space semantic parsing: A framework for compositional vector space models"]()).

The visually-grounded component of this work relies on recent advances in convolutional networks for computer vision (["Very deep convolutional networks for large-scale image recognition"]()), and in particular the fact that late convolutional layers in networks trained for image recognition contain rich features useful for other downstream vision tasks, while preserving spatial information.
These features have been used for both image captioning (["Show, attend and tell: neural image caption generation with visual attention"]()) and visual question answering (["Stacked attention networks for image question answering"]()).


Most previous approaches to visual question answering either apply a recurrent model to deep representations of both the image and the question (["Image question answering: A visual semantic embedding model and a new dataset,Ask your neurons: A neural-based approach to answering questions about images"]()), or use the question to compute an attention over the input image, and then answer based on both the question and the image features attended to (["Stacked attention networks for image question answering,Ask, attend and answer: Exploring question-guided spatial attention for visual question answering"]()).  Other approaches include the simple classification model described by (["Simple baseline for visual question answering"]()) and the dynamic parameter prediction network described by (["Image question answering using convolutional neural network with dynamic parameter prediction"]()).  All of these models assume that a fixed computation can be performed on the image and question to compute the answer, rather than adapting the structure of the computation to the question.


As noted, (["Deep compositional question answering with neural module networks"]()) previously considered a simple generalization of these attentional approaches in  which small variations in the network structure per-question were permitted, with the structure chosen by (deterministic) syntactic processing of questions. Other approaches in this general family include the "universal parser" sketched by (["From machine learning to machine reasoning"]()), and the recursive neural networks of (["Parsing with compositional vector grammars"]()), which use a fixed tree structure to perform further linguistic analysis without any external world representation.
We are unaware of previous work that succeeds in simultaneously learning both the parameters for and structures of instance-specific neural networks.


# Model

Recall that our goal is to map from questions and world representations to answers. This process involves the following variables:

1. $w$ a world representation
2. $x$ a question
3. $y$ an answer
4. $z$ a network layout
5.$\theta$ a collection of model parameters

Our model is built around two distributions:
- a \textbf{layout model} $p(z|x;\lparam)$ which chooses a layout for a sentence,
- and a \textbf{execution model} $p_z(y|w;\theta_e)$ which applies the network specified by $z$ to $w$.


## Evaluating modules

Given a layout $z$, we assemble the corresponding modules into a full neural network , and apply it to the knowledge representation.
Intermediate results flow between modules until an answer is produced at the root. We denote the output of the network with layout $z$ on input world $w$ as $\denote{z}_w$; when explicitly referencing the substructure of $z$, we can alternatively write <span style="color:green">$\denote{m(h^1, h^2)}$</span> for a top-level module $m$ with submodule outputs $h^1$ and $h^2$.
We then define the execution model:
$$
  p_z(y|w) = (\denote{z}_w)_y
  \label{eq:simple-execution}
$$
(This assumes that the root module of $z$ produces a distribution over labels $y$.)

The set of possible layouts $z$ is restricted by module <span style="color:green">$\mathbf{type constraints}$</span>:
some modules (like <span style="color:green">$\text{find}$</span> above) operate directly on the input representation, while others (like <span style="color:green">$\mathbf{describe}$</span> above) also depend on input from specific earlier modules. Two base types are considered in this paper are <span style="color:green">$\text{(a distribution over pixels or entities)}$</span> and <span style="color:green">$\text{(a distribution over answers)}$</span>.

Parameters are tied across multiple instances of the same module, so different instantiated networks may share some parameters but not others.
Modules have both <span style="color:green">$\text{parameter arguments}$</span> (shown in square brackets) and ordinary inputs (shown in parentheses). Parameter arguments, like the running <span style="color:green">$\mathbf{bird} example above, are provided by the layout, and are used to specialize module behavior for particular lexical items. Ordinary inputs are the result of computation lower in the network. In addition to parameter-specific weights, modules have global weights shared across all instances of the module (but not shared with other modules). We write $A, a, B, b, \dots$ for global weights and $u^i, v^i$ for weights associated with the parameter argument $i$. $\oplus$ and $\odot$ denote (possibly broadcasted) elementwise addition and multiplication respectively. The complete set of
global weights and parameter-specific weights constitutes $\theta_e$.

<span style="color:green">$\mathbf{Every}$</span> module has access to the world representation, represented as a collection of vectors $w^1, w^2, \dots$ (or $W$ expressed as a matrix). The nonlinearity $\sigma$ denotes a rectified linear unit.

The modules used in this paper are shown below, with names and type constraints in the first row and a description of the module's computation following.

||||
|---|---|---|
|$\textbf{Lookup}$ | ($\to$ Attention) |$\textbf{lookup[i]}  = e_{f(i)}$|
||<span style="color:green">$\mathbf{lookup[i]}$</span> produces an attention focused entirely at the index $f(i)$, where the relationship $f$ between words and positions in the input map is known ahead of time (e.g. string matches on database fields).| $e_i$ is the basis vector that is $1$ in the $i$th position and 0 elsewhere. |
|$\textbf{Find}$|($\to$ Attention)|$\mathbf{find[i]} = \text{softmax}(a \odot \sigma(B v^i \oplus C W \oplus d))$|
||<span style="color:green">$\mathbf{find[i]}$</span> computes a distribution over indices by concatenating the parameter argument with each  position of the input feature map, and passing the concatenated vector through a MLP||
|$\textbf{Relate}$|(Attention $\to$ Attention)|$\mathbf{relate[i]}(h) = \text{softmax}(a \odot \sigma(B v^i \oplus C W \oplus D\bar{w}(h) \oplus e))$|
||<span style="color:green">$\mathbf{relate}$</span> directs focus from one region of the input to another. It behaves much like the <span style="color:green">$\mathbf{find}$</span> module, but also conditions its behavior on the current region of attention $h$.|$\bar{w}(h) = \sum_k h_k w^k$, where $h_k$ is the $k^{th}$</span> element of $h$|
|$\textbf{And}$|(Attention* $\to$ Attention)|$[\mathbf{and}(h^1, h^2, ...)] = h^1 \odot h^2 ...$|
||<span style="color:green">$\mathbf{and}$</span> performs an operation analogous to set intersection for attentions. The analogy to probabilistic logic suggests multiplying probabilities|
|$\textbf{Describe}$|(Attention $\to$ Labels)|$\mathbf{describe[i]}(h) = \text{softmax}(A \sigma(B \bar{w}(h) + v^i))$|
||<span style="color:green">$\mathbf{describe[i]}$</span> computes a weighted average of $w$ under the input attention. This average is then used to predict an answer representation. With $\bar{w}$ as above,||
|$\textbf{Exists}$|(Attention $\to$ Labels)|$\mathbf{exists]}(h) = \textrm{softmax}((\max_k h_k)a  + b)$|
||<span style="color:green">$\mathbf{exists}$</span> is the existential quantifiers, and inspects the incoming attention directly to produce a label, rather than producing an intermediate feature 	vector like <span style="color:green">$\mathbf{describe}$</span>||


With $z$ observed, the model we have described so far corresponds largely to that of (["Andreas15NMN}, though the module inventory is different---in particular, our new <span style="color:green">$\mathbf{exists}$</span> and <span style="color:green">$\mathbf{relate}$</span> modules do not depend on the two-dimensional spatial structure of the input. This enables generalization to non-visual world representations.

Learning in this simplified setting is straightforward. Assuming the top-level module in each layout is a <span style="color:green">$\mathbf{describe}$</span> or <span style="color:green">$\mathbf{exists}$</span> module, the fully-instantiated network corresponds to a distribution over labels conditioned on layouts. To train, we maximize $\sum_{(w,y,z)} \log p_z(y|w;\theta_e)$directly.This can  be understood as a parameter-tying scheme, where the decisions about which parameters to tie are governed by the observed layouts $z$.

## Assembling networks


Next we describe the layout model $p(z|x;\theta_l)$. We first use a fixed syntactic parse to generate a small set of candidate layouts, analogously to the way a semantic grammar generates candidate semantic parses in previous work (["Semantic parsing via paraphrasing"]()).

A semantic parse differs from a syntactic parse in two primary ways.
First, lexical items must be mapped onto a (possibly smaller) set of semantic primitives. Second, these semantic primitives must be combined into a structure that closely, but not exactly, parallels the structure provided by syntax.
For example, <span style="color:green">$\mathbf{state}$</span> and <span style="color:green">$\mathbf{province}$</span> might need to be identified with the
same field in a database schema, while <span style="color:green">$\mathbf{all \ states \ have \ a \ capital}$</span> might need
to be identified with the correct (<span style="color:green">$\mathbf{in \ situ})$</span> quantifier scope.


While we cannot avoid the structure selection problem, continuous representations simplify the lexical selection problem. For modules that accept a vector parameter, we associate these parameters with <span style="color:green">$\mathbf{words}$</span> rather than semantic tokens,  and thus turn the combinatorial optimization problem associated with lexicon induction into a continuous one. Now, in order to learn that <span style="color:green">$\mathbf{province}$</span> and <span style="color:green">$\mathbf{state}$</span> have the same denotation, it is sufficient to learn that their associated parameters are close in some embedding space---a task amenable to gradient descent.

(Note that this is easy only in an optimizability sense, and not an information-theoretic one---we must still learn to associate each independent lexical item with the correct vector.) The remaining combinatorial problem is to arrange the provided lexical items into the right computational structure. In this respect, layout prediction is more like syntactic parsing  than ordinary semantic parsing, and we can rely on an off-the-shelf syntactic parser to get most of the way there. In this work, syntactic structure is provided by the Stanford dependency parser.

The construction of layout candidates is depicted . We assume queries are conjunctive at the top level, and collect the set of attributes and prepositional relations that depend on the wh-word or copula in the question. The parser is free to consider subsets of this conjunction, and optionally to insert an existential quantifier. These are strong simplifying assumptions, but appear sufficient to cover most of the examples that appear in both of our tasks. As our approach includes both categories, relations and simple quantification, the range of phenomena considered is generally broader than previous perceptually-grounded QA work (["Jointly learning to parse and perceive: connecting natural language  to the physical world,A joint model of language and perception for grounded attribute learning"]()).

Having generated a set of candidate parses, we need to score them. This is a reranking problem; as in the rest of our approach, we solve it using standard neural machinery. In particular, we produce an LSTM representation of the question, a feature-based representation of the query (with indicators on the type and number of modules used), and pass both representations through a multilayer perceptron (MLP). While one can easily imagine a more sophisticated parse-scoring model, this simple approach works well for our tasks.

Formally, for a question $x$, let $h_q(x)$ be an LSTM encoding of the question (i.e. the last hidden layer of an LSTM applied word-by-word to the input question). Let $\{z_1, z_2, \ldots\} be the proposed layouts for $x$, and let $f(z_i)$ be a feature vector representing the $i$th layout. Then the score $s(z_i|x)$ for the layout $z_i$ is
$$
	s(z_i|x) = a^\top \sigma(B h_q(x) + C f(z_i) + d)
$$
i.e.\ a the output of an MLP with inputs $h_q(x)$ and $f(z_i)$, and parameters $\theta_l = \{a, B, C, d\}$. Finally, we normalize these scores to obtain a distribution:
$$
	p(z|x;\theta_l) = e^{s(z_i|x)} / \sum_{j=1}^n e^{s(z_j|x)}
$$

Having defined a layout selection module $p(z|x;\theta_l)$ and a network execution model $p_z(y|w;\theta_e)$, we are ready to define a model for predicting answers given only (world, question) pairs. The key constraint is that we want to minimize evaluations of $p_z(y|w;\theta_e)$ (which involves expensive application of a deep network
to a large input representation), but can tractably evaluate $p(z|x;\theta_l)$ for all $z$ (which involves application of a shallow network to a relatively small set of candidates). This is the opposite of the situation usually encountered semantic parsing, where calls to the query execution model are fast but the set of candidate
parses is too large to score exhaustively.


In fact, the problem more closely resembles the scenario faced by agents in the reinforcement learning setting (where it is cheap to score actions, but potentially expensive to execute them and obtain rewards).
We adopt a common approach from that literature, and express our model as a stochastic policy. Under this policy, we first <span style="color:green">$\mathbf{sample}$</span> a layout $z$  from a distribution $p(z|x;\theta_l)$, and then apply $z$ to the knowledge source and obtain a distribution over answers $p(y|z,w;\theta_e)$.


After $z$ is chosen, we can train the execution model directly by maximizing $\log p(y|z,w;\theta_e)$ with respect to $\theta_e$ as before (this is ordinary backpropagation).  Because the hard selection of $z$ is non-differentiable, we optimize $p(z|x;\theta_l)$ using a policy gradient method.  (["Williams92Reinforce} showed that the gradient of the reward surface $J$ with respect to the  parameters of the policy is
$$
  \nabla J(\theta_l) = \mathbb{E}[ \nabla \log p(z|x;\theta_l) \cdot r ]
$$
(this is the {\sc reinforce} rule).  Here the expectation is taken with respect to rollouts of the policy, and $r$ is the reward. Because our goal is to select the network that makes the most accurate predictions, we take the reward to be identically the negative log-probability from the execution phase, i.e.
$$
  \mathbb{E} [(\nabla \log p(z|x;\theta_l)) \cdot \log p(y|z,w;\theta_e)]
$$
Thus the update to the layout-scoring model at each timestep is simply the gradient of the log-probability of the chosen layout, scaled by the accuracy of that layout's predictions.

At training time, we approximate the expectation with a single rollout, so at each step we update $\theta_l$ in the direction

$
  (\grad \log p(z|x;\theta_l)) \cdot \log p(y|z,w;\theta_e)
$
%
for a single $z \sim p(z|x;\theta_l)$. $\theta_e$ and $\theta_l$ are optimized using {\sc adadelta} (["{ADADELTA}: {A}n adaptive learning rate method"]()) with $\rho=0.95,$ $\varepsilon=1e-6$ and gradient clipping at a norm of $10$.

# Experiments}

The framework described in this paper is general, and we are interested in how well it1 performs on datasets of varying domain, size and linguistic complexity. To that end, weevaluate our model on tasks at opposite extremes of both these criteria: a large visual question answering dataset, and a small collection of more structured geography questions.

## Questions about images}

Our first task is the recently-introduced Visual Question Answering challenge (VQA) (["{VQA}: Visual question answering"]()). The VQA dataset consists of more than 200,000 images paired with human-annotated questions
and answers.

We use the VQA 1.0 release, employing the development set formodel selection and hyperparameter tuning, and reporting final results from the evaluation server on the test-standard set. For the experiments described in this section, the input feature representations $w_i$ are computed by the the fifth convolutional layer of a 16-layer VGG\-Net after pooling (["Very deep convolutional networks for large-scale image recognition"]()).
Input images are scaled to 448$\times$448 before computing their representations. We found that performance on this task was best if the candidate layouts were relatively simple: only <span style="color:green">$\mathbf{describe}$</span>, <span style="color:green">$\mathbf{and}$</span> and <span style="color:green">$\mathbf{find}$</span> modules are used, and layouts contain at most two conjuncts.

One weakness of this basic framework is a difficulty modeling prior knowledge about answers (of the form <span style="color:green">$\mathbf{bears are brown}$</span>). This kinds of linguistic "prior" is essential for the VQA task, and easily incorporated. We simply introduce an extra hidden layer for recombining the final module network output with the input sentence representation $h_q(x)$ (see \autoref{eq:layout-score}), replacing \autoref{eq:simple-execution} with:
$$
  \log p_z(y|w,x) = (A h_q(x) + B \denote{z}_w)_y
$$
(Now modules with output type \labtype should be understood as producing an answer embedding rather than a distribution over answers.) This allows the question to influence the answer directly.



Results are shown in \autoref{tbl:vqa:quantitative-results}. The use of dynamic networks provides a small gain, most noticeably on yes/no questions. We achieve state-of-the-art results on this task,
outperforming a highly effective visual bag-of-words model (["Simple baseline for visual question answering"]()),a model with dynamic network parameter prediction (but fixed network structure) (["Image question answering using convolutional neural network with dynamic parameter prediction"]()), and a previous approach using neural module networks with no structure prediction (["Deep compositional question answering with neural module networks"]()). For this last model, we report both the numbers from the original paper, and a reimplementation of the model that uses the same image preprocessing as the dynamic module network experiments in this paper. A more conventional attentional model has also been applied to this task (["Stacked attention networks for image question answering"]()); while we also outperform their reported performance, the evaluation uses different  train/test split, so results are not directly comparable.

Some examples are shown in \autoref{fig:vqa:qualitative-results}. In general, the model learns to focus on the correct region of the image, and tends to consider a broad window around the region. This facilitates answering questions like <span style="color:green">$\mathbf{Where is the cat?}$</span>, which requires knowledge of the surroundings as well as the object in question.

## Questions about geography}

The next set of experiments we consider focuses on GeoQA, a geographical
question-answering task first introduced by (["Jointly learning to parse and perceive: connecting natural language to the physical world"]()).
This task was originally paired with a visual question answering task
much simpler than the one just discussed, and is appealing for a
number of reasons. In contrast to the VQA dataset, GeoQA is quite small,
containing only 263 examples. Two baselines are available: one using a
classical semantic parser backed by a database, and another which
induces logical predicates using linear
classifiers over both spatial and distributional features. This allows
us to evaluate the quality of our model relative to other perceptually
grounded logical semantics, as well as strictly logical approaches.

The GeoQA domain consists of a set of entities (e.g.\ states, cities,
parks) which participate in various relations (e.g.\ north-of,
capital-of). Here we take the world representation to consist of two pieces:
a set of category features (used by the <span style="color:green">$\mathbf{find}$</span> module) and a different
set of relational features (used by the <span style="color:green">$\mathbf{relate}$</span> module). For our experiments,
we use a subset of the features originally used by Krishnamurthy et al.

The original dataset includes no quantifiers, and treats the questions
<span style="color:green">$\mathbf{What cities are in Texas?}$</span> and <span style="color:green">$\mathbf{Are there any cities in Texas?}$</span>\
identically. Because we're interested in testing the parser's ability to
predict a variety of different structures, we introduce a new version
of the dataset, GeoQA+Q, which distinguishes these two cases, and expects a
Boolean answer to questions of the second kind.

Results are shown in \autoref{tbl:geo:quantitative}. As in the original work, we report the results
of leave-one-environment-out cross-validation on the set of 10 environments.
Our dynamic model (D-NMN) outperforms both the logical (LSP-F) and perceptual models (LSP-W) described by
(["Krish2013Grounded}, as well as a fixed-structure neural module net (NMN).
This improvement is particularly notable on the dataset with  quantifiers, where dynamic
structure prediction produces a 20\% relative improvement  over the fixed baseline.
A variety of predicted layouts are shown in \autoref{fig:geo:qualitative}.

# Conclusion}

We have introduced a new model, the <span style="color:green">$\mathbf{dynamic neural module network}$</span>,
for answering queries about both structured and unstructured sources of
information. Given only (question, world, answer) triples as training data,
the model learns to assemble neural networks on the fly from an inventory
of neural models, and simultaneously learns weights for these modules so
that they can be composed into novel structures. Our approach achieves
state-of-the-art results on two tasks.

We believe that the success of this work derives from two factors:

<span style="color:green">$\mathbf{Continuous representations improve the expressiveness and learnability of semantic parsers}$</span>: by replacing discrete predicates with differentiable neural
network fragments, we bypass the challenging combinatorial optimization problem
associated with induction of a semantic lexicon. In structured
world representations, neural predicate representations allow the model to
invent reusable attributes and relations not expressed in
the schema. Perhaps more importantly, we can extend
compositional question-answering machinery to complex, continuous world
representations like images.


<span style="color:green">$\mathbf{Semantic structure prediction improves generalization in deep networks}$</span>:
by replacing a fixed network topology with a dynamic one, we can tailor the
computation performed to each problem instance, using deeper networks for more
complex questions and representing combinatorially many queries with
comparatively few parameters.  In practice, this results in considerable gains
in speed and sample efficiency, even with very little training data.

These observations are not limited to the question answering domain, and
we expect that they can be applied similarly to tasks like
instruction following, game playing, and language generation.
