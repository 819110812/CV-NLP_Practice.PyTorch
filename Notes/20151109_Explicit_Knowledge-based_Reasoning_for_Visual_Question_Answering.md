# [Explicit Knowledge-based Reasoning for Visual Question Answering](https://arxiv.org/abs/1511.02570)

## Recent advances

Despite the implied need to perform general reasoning about the content of images, most VQA methods perform <span style="color:green">no explicit reasoning at all</span>. There are a number of problems with the LSTM approach:

1. The first is that the method does <span style="color:green">not explain how it arrived at its answer</span>.
2. This means that it is <span style="color:green">impossible to tell whether it is answering the question based on image information, or just the prevalence of a particular answer in the training set</span>.
3. The second problem is that <span style="color:green">the amount of prior information</span> that can be encoded within a LSTM system is very limited.
4. DBpedia ["DBpedia:  A  nucleus  for  a  webof open data"](), with millions of concepts and hundred millions of relationships, contains a small subset of the information required to truly reason about the world in general.
5. The third, and major, problem with the LSTM approach is that it is <span style="color:green">incapable of explicit reasoning except in very limited situations</span>.

# Contribution

1. They describe a method for VQA which  is  capable  of  <span style="color:red">reasoning  about  contents  of  an image</span> on the basis of information extracted from a <span style="color:red">large-scale knowledge base</span>. It is capable of correctly answering a far broader range of image-based questions than competing methods, and provides an explanation of the reasoning by which it arrived at the answer. Ahab exploits DBpedia as its source of external information, and requires no VQA training data (it does use ImageNet and MS COCO to train the visual concept detector).
2. The method not only answers natural language questions using concepts not contained in the image, but can <span style="color:red">provide an explanation of the reasoning by which it developed its answer</span>.
3. The  method  is  capable  of  answering  far  <span style="color:red">more  complex questions than the predominant LSTM-based approach, and outperforms it signicantly in the testing</span>.
4. They provide a <span style="color:red">dataset</span> and a <span style="color:red">protocol</span> by which to evaluate such methods, thus addressing one of the key issues in general visual question answering. They propose a dataset, and protocol for measuring performance, for general visual question answering. The questions in the dataset are generated by human subjects based on a number of predefined templates.

# Introduction

They propose Ahab, a new approach to VQA which is based on explicit reasoning about the content of images.

1. Ahab first detects <span style="color:red">relevant content in the image</span>, and relates it to information available in a knowledge base.
2. A natural language question is processed into a suitable query which is run over the <span style="color:red">combined image and knowledge base information</span>.
3. This process allows complex questions to be asked which rely on information not available in the image. Examples include questioning the relationships between two images, or asking whether two depicted animals are close taxonomic relatives.

# Related works

## LSTM approach

1. [“A multi-world approach to question answering about real-world scenes based on uncertain input”]() proposed to process questions using semantic parsing ["Learning dependency-based compositional semantics"]() and obtain answers through Bayesian reasoning.
2. ["Are You Talking to a Machine? Dataset and Methods for Multilingual Image Question Answering"]() used independent LSTM networks for question encoding and answer decoding
3. ["Ask Your Neurons: A Neural-based Approach to Answering Questions about Images"]() used one LSTM for both tasks.

## Knowledge base mthods

Significant advances have been made, however, in the construction of large-scale structured Knowledge Bases (KBs):

1. [VQA: Visual Question Answering](),
2. [Open information extraction for the web](),
3. [Freebase: a collaboratively created graph database for structuring human knowledge](),
4. [Toward an Architecture for Never-Ending Language Learning](),
5. [Neil: Ex- tracting visual knowledge from web data](),
6. [A multi-world approach to question answering about real-world scenes based on uncertain input](),
7. [Wikidata: a free collaborative knowledgebase]().

In structured KBs, <span style="color:red">knowledge is typically represented by a large number of triples of the form $(arg1,rel,arg2)$, where $arg1$ and $arg2$ denote two entities in the KB and rel denotes a predicate representing the relationship between these two entities</span>.

A collection of such triples can be seen as a large interlinked graph. Such triples are often described in terms of a Resource Description Framework [Resource description framework]() (RDF) specification, and housed in a relational database management system (RDBMS), or triple-store, which allows queries over the data. The knowledge that “a cat is a domesticated animal”, for instance, is stored in an RDF KB by the triple (cat,is-a,domesticated animal). The information in KBs can be accessed efficiently using a query language. In this work we use SPARQL Protocol [SPARQL query language for RDF]() to query the Open-Link Virtuoso [a Hybrid RDBMS/Graph Column Store]() RDBMS. For example, the query ?x:(?x,is-a,domesticated animal) returns all domesticated animals in the graph.

Popular large-scale structured KBs are constructed either by
1. Mmanual-annotation/crowd-sourcing:
    - DBpedia [DBpedia: A nucleus for a web of open data](),
    - Freebase [Freebase: a collaboratively created graph database for structuring human knowledge]()
    - Wikidata [Wikidata: a free collaborative knowledgebase]()),
2. Automatically extracting from unstructured/semi-structured data
    - YAGO[YAGO2: A spatially and temporally enhanced knowledge base from Wikipedia](),
    - [YAGO3:A knowledge base from multilingual Wikipedias](),
    - OpenIE[Open information extraction for the web](),
    - [Open Information Extraction: The Second Generation](),
    - [Identifying relations for open information extraction](),
    - NELL [Toward an Architecture for Never-Ending Language Learning](),
    - NEIL [Neil: Ex- tracting visual knowledge from web data]()).

<span style="color:red">The KB they use here is DBpedia</span>, which contains structured information extracted from Wikipedia. Compared to KBs extracted automatically from unstructured data (such as <span style="color:red">OpenIE</span>), the data in DBpedia is more accurate and has a well-defined ontology.

The method they propose is applicable to any KB that admits SPARQL queries, however, including those listed above and the huge variety of subject-specific RDF databases available.

The advances in structured KBs, have driven an increasing interest in the NLP and AI communities in the problem of natural language question answering using structured KBs (refer to as KB-QA):

1. [Semantic Parsing on Freebase from Question-Answer Pairs](),
2. [Question answering with subgraph embeddings](),
3. [Large-scale Semantic Parsing via Schema Matching and Lexicon Extension](),
4. [Open question answering over curated and extracted knowl- edge bases](),
5. [A survey on question answering technology from an information retrieval perspective](),
6. [Scaling semantic parsers with on-the-fly ontology matching](),
7. [Learning dependency-based compositional semantics](),
8. [nformation extraction over structured data: Question answering with Free- base](),
9. [Template-based question answering over RDF data]()).

The VQA approach which is closest to KB-QA is that of Zhu et al. [Building a large-scale multimodal Knowledge Base for Answer- ing Visual Queries]() as they use a KB and RDBMS to answer image-based questions. They build the KB for the purpose, however, using an [MRF model](https://en.wikipedia.org/wiki/Markov_random_field), with image features and scene/attribute/affordance labels as nodes. The undirected links between nodes represent mutual compatibility/incompatibility relationships. The KB thus relates specific images to specified image-based quantities to the point where the database schema prohibits recording general information about the world. The queries that this approach can field are crafted in terms of this particular KB, and thus relate to the small number of attributes specified by the schema. The questions are framed in an RDMBS query language, rather than natural language.

# The Ahab VQA approach

## RDF Graph Construction

In order to reason about the content of an image we need to amass the relevant information. This is achieved by detecting concepts in the <span style="color:red">query image</span> and <span style="color:red">linking them to the relevant parts of the KB</span>.

Visual Concepts Three types of visual concepts are detected in the query image, including:

− <span style="color:red">Objects</span>: They trained two Fast-RCNN detectors on MS COCO 80-class objects and ImageNet 200-class objects. Some classes with low precision were removed from the models, such as “ping-pong ball” and “nail”. The finally merged detector contains 224 object classes, which can be found in the supplementary material.
− <span style="color:red">Image Scenes</span>: The scene classifier is obtained from [42], which is a VGG-16 [36] CNN model trained on the MIT Places205 dataset. In our system, the scene classes corresponding to the top-3 scores are selected.
− <span style="color:red">Image Attributes</span>: The vocabulary of attributes defined in [Image Captioning with an Intermediate Attributes Layer]() covers a variety of high-level concepts related to an image, such as actions, objects, sports and scenes.

They select the <span style="color:red">top-10 attributes for each image</span>. Linking to the KB Having extracted a set of concepts of interest from the image, we now need to relate them to the appropriate information in the KB.

The <span style="color:red">visual concepts (object, scene and attribute cat- egories) are stored as RDF triples</span>.

For example, the information that “The image contains a giraffe object” is expressed as:

$$\text{(Img,contain,Obj-1) and (Obj-1,name,ObjCat-giraffe)}$$

Each visual concept is linked to DBpedia entities with the same semantic meaning (identified through a uniform resource identifier2 (URI)), for ex- ample (ObjCat-giraffe, same-concept, KB:Giraffe). The resulting RDF graph includes all of the relevant information in DBpedia, linked as appropriate to the visual concepts extracted from the query image.

# Answering Questions

Having gathered all of the relevant information from the image and DBpedia, now use them to answer questions.

Parsing NLQs Given a question posed in natural language,

1. first need to <span style="color:red">translate it to a format which can be used to query the RDBMS</span>(Quepy3 is a Python framework designed within the NLP community to achieve exactly this task).
2. To achieve this Quepy requires a set of templates, framed in terms of regular expressions.  Quepy begins by tagging each word in the question using NLTK, which is composed of a tokenizer, a part-of-speech tagger and a lemmatizer. The tagged question is then parsed by a set of regular expressions (regex), each defined for a specific question template. These regular expressions are built using REfO4 to increase the flexibility of question expression as much as possible.
3. Once a regex matches the question, it will extract the slot- phrases and forward them for further processing.  Mapping Slot-Phrases to KB-entities Note that the slot-phrases are still expressed in natural language.
4. The next step is to <span style="color:red">find the correct correspondences between the slot-phrases and entities in the constructed graph</span>.

# Experiments

Metrics Performance evaluation in VQA is complicated by the fact that two answers can have no words in common and yet both be perfectly correct.

Malinowski and Fritz ["A multi-world approachto question answering about real-world scenes basedon uncertain input"]() used the <span style="color:red">Wu-Palmer similarity (WUPS)</span> to measure the similarity between two words based on their common subsequence in the taxonomy tree. However, this evaluation metric restricts the answer to be a single word.

Antol et al. ["QA: Visual Questio nAnswering"]() provided an evaluation metric for the open-answer task which records the percentage of answers in agreement with ground truth from several human subjects. This evaluation metric requires around 10 ground truth answers for each question, and only partly solves the problem (as indicated by the fact that even human performance is very low in some cases in ["QA: Visual Questio nAnswering"](), such as ‘Why ...?’ questions.).

In the paper's case, the existing evaluation metrics are particularly unsuitable because most of the questions in their dataset are open-ended, especially for the “KB- knowledge” questions. In addition, there is no automated method for assessing the reasons provided by our system.

Extending VQA forms Typically, a VQA problem involves one image and one natural language question (IMG+NLQ). Here we extend VQA to problems involving more images. With this extension, we can ask more interesting questions and more clearly demonstrate the value of using a structured knowl- edge base.  The first type of question (Q7-Q9) asks for the common properties between two whole images; the second type (Q10-Q12) gives a concept and asks which image is the most related to this concept.

For the first question type, Ahab obtains the answers by searching all common transitive categories shared by the visual concepts extracted from the two query images. For example, although the two images in Q9 are significantly different visually (even at the object level), and share no attributes in common,their scene categories (railway station and airport) are linked to the same concept “transport infrastruc- ture” in DBpedia. For the second type, the corre- lation between each visual concept and the query concept is measured by a scoring function and the correlation be- tween an image and this concept is calculated by aver- aging the top three scores. As we can see in Q11 and Q12, attributes “kitchen” and “computer” are most related to the concepts “chef” and “programmer” re- spectively, so it is easy to judge that the answer for Q11 is the left image and the one for Q12 is the right.
The flexibility of Quepy, and the power of Python, make adding additional question types quite sim- ple. It would be straightforward to add question types requiring an image as an answer, for instance (IMG1+NLQ → IMGs).

# Conclusion

Described a method capable of reasoning about the content of general images, and interactively answering a wide variety of questions about them. The method develops a structured represen- tation of the content of the image, and relevant information about the rest of the world, on the basis of a large external knowledge base.

It is capable of explaining its reasoning in terms of the entities in the knowledge base, and the connections between them.

Ahab is applicable to any knowledge base for which a SPARQL interface is available. This includes any of the over a thousand RDF datasets online ["State of the LOD Cloud2014"]() which relate information on taxonomy, music, UK government statistics, Brazilian politicians, and the articles of the New York Times, amongst a host of other topics. Each could be used to provide a specific visual question answering capability, but many can also be linked by common identifiers to form larger repositories. If a knowledge base containing common sense were available, the method we have described could use it to draw sensible general conclusions about the content of images.
