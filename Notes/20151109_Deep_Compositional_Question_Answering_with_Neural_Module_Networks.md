## [Deep Compositional Question Answering with Neural Module Networks](https://arxiv.org/abs/1511.02799)


Home Page: http://www.cs.berkeley.edu/~jda/
Code: http://github.com/jacobandreas/nmn2

<span style="color:purple">
<i>Visual question answering is fundamentally compositional in nature - a question like <b><u>where is the dog</u></b> shares substructure with questions like <u>what color is the dog</u> and <u>where is the cat</u> .This paper seeks to simultaneously exploit the representational capacity of deep networks and the compositional linguistic structure of questions.</i>
</span>

> They describe a procedure for constructing and learning <b>neural module networks</b>, <u>which compose collections of jointly-trained neural modules into deep networks for question answering</u>.

<span style="color:purple"><i>
Their approach decomposes questions into their linguistic substructures, and uses these structures to dynamically instantiate modular networks (with reusable components for recognizing dogs, classifying colors, etc.).
</i></span>

## Related works(General  compositional  semantics)

<span style="color:purple"><i>
There  is  a  large  literature  on  learning  to  answer  questions  about  structured knowledge  representations  from  question–answer  pairs,both with and without joint learning of meanings for simple predicates <font size="2">(["Learning dependency-based compositional semantics"](https://cs.stanford.edu/~pliang/papers/dcs-acl2011.pdf),["Jointly  learning  to  parseand  perceive:  connecting  natural  language  to  the  physical world"](http://www.jayantkrish.com/papers/tacl2013-krishnamurthy-kollar.pdf))</font>.
</i></span>

<span style="color:purple"><i>
Outside of question answering, several models have been proposed for instruction following that impose a discrete “planning structure” over an underlying continuous control signal<font size="2">(["Grounding Language with Points and Paths in Continuous Spaces"](http://people.eecs.berkeley.edu/~jda/papers/ak_paths.pdf),["A  joint  model  of  language  and  perception  for grounded attribute learning"](https://homes.cs.washington.edu/~lsz/papers/mfzbf-icml12.pdf))</font>.
</i></span>

<span style="color:purple"><i>
They are unaware of <u>past use of a semantic parser to predict network structures</u>,or more generally to exploit the natural similarity between set-theoretic approaches to <u>classical semantic parsing and attentional approaches</u> to computer vision.
</i></span>

# Neural module networks for visual QA

Each training datum for this task can be thought of as a <u>3-tuple ($w;x;y$)</u>, where

- $w$ is a natural-language question
- $x$ is an image
- $y$ is an answer

<span style="color:purple"><i>
<span style="color:red">A model is fully specified by a collection of modules $m$</span>, each with associated parameters $\theta_m$, and a network layout predictor $P$ which maps from strings to networks.

Given $(w;x)$ as above, the model instantiates a network based on $P(w)$, passes $x$(and possibly $w$ again) as inputs, and obtains a distribution over labels.  Thus a model ultimately encodes a predictive distribution $p(y|w; x;\theta)$
</i></span>

<span style="color:purple"><i>
Their goal here is to identify a small set of modules that can be assembled into all the configurations necessary for our  tasks.   This  corresponds  to  identifying  a  minimal  set of composable vision primitives.  The modules operate on three  basic  data  types:  <span style="color:red">images,  unnormalized  attentions, and labels</span>.
</i></span>

For the particular task and modules describedin this paper, almost all interesting compositional phenomena occur in the space of attentions, and it is not unreasonable to characterize our contribution more narrowly as an“attention-composition” network. Nevertheless, other types may be easily added in the future (for new applications orfor greater coverage in the VQA domain).


> First, some notation

module   names   are   type-set in   a fixed width font,    and   are   of   the   form TYPE[INSTANCE](ARG1, ...).

<span style="color:red">$TYPE$</span> is  a  high-level  moduletype (attention, classification, etc.) of the kind described in this section.

<span style="color:red">$INSTANCE$</span> is the particular instance of the modelunder consideration—for example,<span style="color:green">$attend[red]$</span> locates red things,  while <span style="color:green">$attend[dog]$</span> locates  dogs.   Weights  may  be shared at both the type and instance level. Modules with noarguments implicitly take the image as input;  higher-levelarguments may also inspect the image.

1. <span style="color:red">Attention </span>
  $$\text{attend}: Image \rightarrow Attention \tag{1}$$
  An  attention  module <span style="color:green">
  $attend[c]$</span> convolves  every  positionin the input image with a weight vector (distinct for eachc)  to  produce  a  heatmap  or  unnormalized  attention.   So,for  example,  the  output  of  the  moduleattend[dog]is  amatrix  whose  entries  should  be  in  regions  of  the  imagecontaining cats, and small everywhere else, as shown above.
2. <span style="color:red">Re-attention </span>
  $$\text{re-attend}: Attention \rightarrow Attention \tag{2}$$
  A  re-attention  module <span style="color:green">$re-attend[c]$</span> is  essentially  just  a multilayer perceptron with rectified nonlinearities (ReLUs),performing a fully-connected mapping from one attentionto another. Again, the weights for this mapping are distinctfor each <span style="color:green">$c$</span>. So <span style="color:green">$re-attend[above]$</span> should take an attentionand  shift  the  regions  of  greatest  activation  upward  (as above), while <span style="color:green">$re-attend$[not]$</span> should move attention away from the active regions.  For the experiments in this paper,the  first  fully-connected  (FC)  layer  produces  a  vector  ofsize 32, and the second is the same size as the input.
3. <span style="color:red">Combination </span>
  $$\text{combine}: Attention \times Attention \rightarrow Attention \tag{3}$$
  A  combination  module <span style="color:green">$combine[c]$</span> merges  two  attentions into a single attention.  For example,<span style="color:green">$combine[and]$</span> should be active only in the regions that are active in both inputs,while <span style="color:green">$combine[except]$</span> should be active where the first input is active and the second is inactive.
4. <span style="color:red">Classification </span>
  $$\text{classify}: Image \times Attention \rightarrow Label \tag{4}$$
  A classification module <span style="color:green">$classify[c]$</span> takes an attention andthe input image and maps them to a distribution over labels. For example, <span style="color:green">$classify[color]$</span> should return a distribution over colors in the region attended to.
5. <span style="color:red">Measurement </span>
  $$\text{measure}:  Attention \rightarrow Label \tag{5}$$
  A measurement module <span style="color:green">$measure[c]$</span> takes an attention aloneand  maps  it  to  a  distribution  over  labels.   Because  attentions passed between modules are unnormalized, <span style="color:green">$measure$</span> is suitable for evaluating the existence of a detected object, orcounting sets of objects


# From strings to networks

1. Having built up an inventory of modules, then assemble them into the layout specified by the question.
2. The transformation from a natural language question to an instantiated neural network takes place in two steps.
3. First map from natural language questions to layouts, which specify both the set of modules used to answer a given question, and the connections between them.
4. Next use these layouts are used to assemble the final prediction networks.We  use  standard  tools  pre-trained  on  existing  linguistic resources to obtained structured representations of ques-tions.
5. Future work might focus on learning (or at least fine-tuning) this prediction process jointly with the rest of thesystem

## Parsing

1. Parsing each question with the Stan-ford Parser ["Accurate unlexicalized parsing"](http://people.eecs.berkeley.edu/~klein/papers/unlexicalized-parsing.pdf). to obtain a universal dependency represen-tation  ["the Stanford typeddependencies representation"](http://nlp.stanford.edu/pubs/dependencies-coling08.pdf).

2. Next,  filter  the  set  of  dependencies  to  those  connected the wh-word in the question (the exact distance wetraverse varies depending on the task).   This gives a simple symbolic form expressing (the primary) part of the sen-tence’s  meaning.</br>
  For  example, <span style="color:green">$what \ is \ standing \ in \ the \ field$</span> be comes <span style="color:green">$what(stand)$</span>; <span style="color:green">$what \ color \ is \ the \ truck$</span> be comes <span style="color:green">$color(truck)$</span>, and <span style="color:green">$is  \ there \ a \ circle \ next \ to \ a \ square$</span> becomes <span style="color:green">$is(circle, next-to(square))$</span>. </br>
  In the process they also strip away function words like determiners and modals, so <span style="color:green">$what \ type \ of \ cakes \  were \ they?$</span> and <span style="color:green">$what \ type \ of \  cake \ is \ it ?$</span> both get  converted  to <span style="color:green">$type(cake)$</span>.
3. The  code  for  transforming parse trees to structured queries will be provided in the accompanying software package. These  representations  bear  a  certain  resemblance  to pieces of a combinatory logic ["Learning dependency-based compositional semantics."](https://cs.stanford.edu/~pliang/papers/dcs-acl2011.pdf):  every leaf is implicitly a function taking the image as input, and the root representsthe final value of the computation.
4. While compositional  and  combinatorial,  is  crucially  not  logical :the  inferential  computations  operate  on  continuous  representations produced by neural networks, becoming discreteonly in the prediction of the final answer.

## Layout

<span style="color:red">These  symbolic  representations  already  determine  the  structure  of  the  predicted  networks</span>,  but  not  the identities of the modules that compose them.
-  <span style="color:red">This final assignment  of  modules  is  fully  determined  by  the  structure of the parse</span>.
- <span style="color:red">All leaves become attend modules</span>,
- <span style="color:red">all internal nodes become re-attend or combine modules dependent on their arity</span>,
- <span style="color:red">and root nodes become measure modules for yes/no questions and classify modules for all other question types</span>.


Given the mapping from queries to network layouts described  above,  we  have  for  each  training  example  a  network  structure,  an  input  image,  and  an  output  label.

In many  cases,  these  network  structures  are  different,  but have  tied  parameters.

Networks  which  have  the  same high-level  structure  but  different  instantiations  of  individual  modules  (for  example <span style="color:green">$what \ color \ is \ the \ cat?$</span>—<span style="color:green">$classify[color](attend[cat])$</span> and <span style="color:green">$where \ is \ the \ truck\ ?$</span>—<span style="color:green">$classify[where](attend[truck]))$</span> can be processed in the same batch, resulting in efficient computation.


## Generalizations

It is easy to imagine applications where the input to the layout stage comes from something other than a natural language parser. Users of an image database,for example, might write SQL-like queries directly in orderto specify their requirements precisely, e.g.

$$IS(cat)\ AND \ NOT \ (IS(dog)$$

or  even  mix  visual  and  non-visual  specifications  in  theirqueries:

$$IS(Cat) \ and \ table \ > \ 2014-11-5$$


## Answering natural language questions

So far our discussion has focused on the neural modulenet architecture.

Their final model combines the output from the neural module network with predictions from a simple LSTM question encoder.

This is important for two reasons.

1. First,because  of  the  relatively  aggressive  simplification  of  the　question that takes place in the parser, grammatical cues that　do not substantively change the semantics of the question,but which might affect the answer, are discarded. For example,<span style="color:green">$what　\ is \ flying?$</span> and <span style="color:green">$what \ are \ flying?$</span> both get converted to <span style="color:green">$what(fly)$</span>, but their answers should be <span style="color:green">$kite$</span> and <span style="color:green">$kites$</span> respectively, even given the same underlying image features.The question encoder thus allows us to model underlying syntactic regularities  in  the  data.
2. Second,  it  allows  us  to capture semantic regularities:  with missing or low-quality image data, it is reasonable to guess that <span style="color:green">$what \ color \ is \ the \ bear?$</span> is  answered  by <span style="color:green">$brown$</span>,  and  unreasonable  to  guess <span style="color:green">$green$</span>. The question encoder also allows us to model effects of this kind. All experiments in this paper use a standard single-layer LSTM  with  1024  hidden  units.
3. The  question  modeling component predicts a distribution over the set of answers,like the root module of the NMN. The final prediction fromthe model is a geometric average of these two probability distributions,  dynamically reweighted using both text andimage  features.
4. The  complete  model,  including  both  the NMN and sequence modeling component, is trained jointly.

<p align="center"><img src="https://dl.dropboxusercontent.com/s/tebddvvstfdmo2i/Screenshot%20from%202016-05-25%2022%3A55%3A00.png?dl=0" width="500" ></p>


## Training neural module networks

Their training objective is simply to find module parameters maximizing the likelihood of the data.

By design, the last module in every network outputs a distribution over labels, and so each assembled network also represents a probability distribution.

Because of the dynamic network structures used to answer questions, some weights are updated much more frequently than others.

For this reason they found that learning algorithms with adaptive per-weight learning rates per-formed  substantially  better  than  simple  gradient  descent.All  the  experiments  described  below  use  AdaDelta  ["ADADELTA:  An  adaptive  learning  ratemethod"](https://arxiv.org/abs/1212.5701) (thus there was no hyperparameter search over step sizes).

It is important to emphasize that the labels we have assigned to distinguish instances of the same module type—cat,and, etc.—are a notational convenience, and do not reflect  any  manual  specification  of  the  behavior  of  the  corresponding modules. <span style="color:green">$detect[cat]$</span> is not fixed or even initialized  as  cat  recognizer  (rather  than  a  couch  recognizeror a dog recognizer), and <span style="color:green">$combine[and]$</span> isn’t fixed to com-pute intersections of attentions (rather than unions or differences).

Instead, they acquire these behaviors as a by product of the end-to-end training procedure.   As can be seen in  Figure below,  the  image–answer  pairs  and  parameter  tying together encourage each module to specialize in the appropriate way.

<p align="center"><img src="https://dl.dropboxusercontent.com/s/dmkt50mgwo4pumn/Screenshot%20from%202016-05-25%2023%3A02%3A23.png?dl=0" width="700" ></p>


## Experiments: compositionality

They begin with a set of motivating experiments on synthetic data.

Compositionality, and the corresponding ability to answer questions with arbitrarily complex structure,is an essential part of the kind of deep image understanding  visual  QA  datasets  are  intended  to  test.

At  the  same time, <span style="color:red">questions in most existing natural image datasets are quite simple</span>, for the most part requiring that only one or two pieces of information be extracted from an image in order to answer it successfully,  and with little evaluation of robustness in the presence of distractors (e.g.  asking <span style="color:red">is there a blue house in an image of a red house and a blue car</span>).

They  have  created [SHAPES](https://github.com/jacobandreas/nmn2/tree/shapes), a synthetic dataset that places such compositional phenomena at the forefront.

## Experiments: natural images

1. They consider the model’s ability to handle hard perceptual problems involving natural images. Here they evaluate on the VQA dataset.

2. They out perform the best published results on this task.  A break down of their questions by answer type reveals that our model performs especially well on questions answered by <span style="color:red">an object, attribute,  or  number</span>,  but  worse  than  <span style="color:red">a  sequence  baseline in  the  yes/no  category</span>.

3. <span style="color:red"> Inspection  of  training-set  accuracies suggests that performance on yes/no questions is due to overfitting</span>.

4. An ensemble with a sequence-only system might  achieve  even  better  results;  future  work  within  the NMN framework should focus on <span style="color:red"> redesigning the measure module to reduce effects from overfitting</span>.

5. Inspection of <span style="color:red">parser outputs</span> also suggests that there is substantial room to improve the system using a better parser. A hand inspection of the first 50 parses in the training setsuggests that <span style="color:red">most (80–90%) of questions asking for simple properties of objects are correctly analyzed</span>,  but <span style="color:red">more complicated questions</span> are more prone to picking up irrelevant predicates.

6. For example <span style="color:red">are these people most likely experiencing a work day?</span> is parsed as <span style="color:red">be(people, likely)</span>,when the desired analysis is <span style="color:red">is(people, work)</span>.

7. Parser errors of this kind could be fixed with joint learning.

# Conclusions and future work

1. Introduced neural  module  networks(NMN),  which  provide  a  general-purpose  framework  for learning  collections  of  neural  modules  which  <span style="color:red">can  be  dynamically assembled into arbitrary deep networks</span>.

2. Introduced <span style="color:red">a new dataset of highly compositional questions about simple arrangements of shapes</span>,  and shown that ourapproach substantially outperforms previous work.

3. They have  maintained  a  strict  separation  between predicting network structures and learning network parameters. And <span style="color:red">they think it is easy to imagine that these two problems mightbe solved jointly, with uncertainty maintained over network structures throughout training and decoding</span>.

4. This might be accomplished  either  with  a  monolithic  network,  by  using some higher-level mechanism to “attend” to relevant portions of the computation, or else by integrating with existing tools for learning semantic parsers.
