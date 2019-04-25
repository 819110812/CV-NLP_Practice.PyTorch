## Parsing three


(1) Generate parset tree<br/>
<span style="color:red">***Reference***</span>:[深度学习在自然语言处理的应用](http://www.csdn.net/article/2015-11-09/2826166)
/ [Machine Translation Useful Links: Techniques, Toolkits, Videos （机器翻译中的有用链接：相关技术、工具和视频）](http://blog.csdn.net/tianliang0123/article/details/7036050)
/ [CS224d lecture 16札记](http://blog.csdn.net/neighborhoodguo/article/details/47617297)
/ [ CS224d lecture 9札记](http://blog.csdn.net/neighborhoodguo/article/details/47193885)
/ [CS224d lecture 13札记](http://blog.csdn.net/neighborhoodguo/article/details/47387229)
/ [ 什么是解析树？What is a Parse Tree?](http://blog.csdn.net/lixiaohuiok111/article/details/6736529)
/ [基于Stanford Parser 及OpenNLP Shallow Parser构建句子语法解析树](http://blog.csdn.net/yangliuy/article/details/8061039)

(2) Find common tree<br/>
[SOFTWARE - gSpan: Frequent Graph Mining Package](http://www.cs.ucsb.edu/~xyan/software/gSpan.htm)
/ [Graph Boosting Toolbox for Matlab](http://www.nowozin.net/sebastian/gboost/)

</font>

10. [Im2Text: Describing Images Using 1 Million Captioned Photographs](http://tlberg.cs.unc.edu/vicente/sbucaptions/)

11. [Framing image description as a ranking task: data, models and evaluation metrics](http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.html)

12. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](http://kelvinxu.github.io/projects/capgen.html)/ [code](https://github.com/kelvinxu/arctic-captions)

13. [Exploring Nearest Neighbor Approaches for Image Captioning]()<br/>
They explore a variety of nearest neighbor baseline approaches for image captioning

14. [TreeTalk: Composition and Compression of Trees for Image Descriptions. ](http://www.cs.unc.edu/~vicente/files/treetalk_camera_ready.pdf)<br/>



## Parsing

### nltk dependency parse

<font color="#000000" size = "2px">
The source code of [nltk.parse.dependencygraph](Source code for nltk.parse.dependencygraph),</br>
- [nltk.parse.stanford](http://www.nltk.org/_modules/nltk/parse/stanford.html)</br>
- [nltk.parse.dependencygraph](http://www.nltk.org/_modules/nltk/parse/dependencygraph.html)</br>
- [nltk.tree](http://www.nltk.org/_modules/nltk/tree.html)</br>
</font>

<font color="#000000" size = "2px">
Some problems:
[How to parse more than one sentence from text file using Stanford dependency parse?](http://stackoverflow.com/questions/34249579/how-to-parse-more-than-one-sentence-from-text-file-using-stanford-dependency-par)
/ [Dagger - The File Dependency Graph Engine](https://pythonhosted.org/dagger/)
/ [Using The Stanford Parser in Python on Chinese text not working](http://stackoverflow.com/questions/29201833/using-the-stanford-parser-in-python-on-chinese-text-not-working).
/ [NLTK Convert Tree to Array?](http://stackoverflow.com/questions/32548732/nltk-convert-tree-to-array)
/ [Typed Dependency Parsing in NLTK Python](http://stackoverflow.com/questions/29049974/typed-dependency-parsing-in-nltk-python)
/ [Stanford Parser and NLTK](http://stackoverflow.com/questions/13883277/stanford-parser-and-nltk)
/ [Unit tests for nltk.tree.Tree](http://www.nltk.org/howto/tree.html)
/ [Different Output for Stanford Parser Online Tool and Stanford Parser Code](http://stackoverflow.com/questions/21921625/different-output-for-stanford-parser-online-tool-and-stanford-parser-code)
/ [What does the dependency-parse output of TurboParser mean?](http://stackoverflow.com/questions/24394196/what-does-the-dependency-parse-output-of-turboparser-mean)
/ [How do I do dependency parsing in NLTK?](http://stackoverflow.com/questions/7443330/how-do-i-do-dependency-parsing-in-nltk)
/ [Constituent-based Syntactic Parsing with NLTK](http://www.cs.bgu.ac.il/~elhadad/nlp11/nltk-pcfg.html)
/ [Stanford NLP parse tree format](http://stackoverflow.com/questions/34395127/stanford-nlp-parse-tree-format)
/ [How to Traverse an NLTK Tree object?](http://stackoverflow.com/questions/31689621/how-to-traverse-an-nltk-tree-object)
/ [Source code for nltk.parse.stanford](http://www.nltk.org/_modules/nltk/parse/stanford.html)

</br>
<span style="color:red">**Reference:**</span>
[NLP | 自然语言处理 - 考虑词汇的语法解析（Lexicalized PCFG）](http://blog.csdn.net/lanxu_yy/article/details/38336129)
/ [【龙书笔记】语法分析涉及的基础概念简介](http://blog.csdn.net/slvher/article/details/44899745)
/ [Intro to NLP with spaCy](https://nicschrading.com/project/Intro-to-NLP-with-spaCy/)
/ [Guidelines for theClearStyleConstituent to Dependency Conversion](http://www.mathcs.emory.edu/~choi/doc/clear-dependency-2012.pdf)
/ [spaCy](https://spacy.io/#example-use)
/ [spacy Online Demo](https://spacy.io/#example-use)
/ [nltk.grammar DependencyGrammar](http://www.nltk.org/howto/dependency.html)
/ [DependenSee: A Dependency Parse Visualisation/Visualization Tool](http://chaoticity.com/dependensee-a-dependency-parse-visualisation-tool/)
/ [Redshift](https://github.com/syllog1sm/redshift)
/ [Parsing English in 500 lines of Python](https://spacy.io/blog/parsing-english-in-python)
/ [Stanford Dependencies](http://nlp.stanford.edu/software/stanford-dependencies.shtml)
/ [The Stanford Parser: A statistical parser](http://nlp.stanford.edu/software/lex-parser.shtml)
/ [spaCy online demo](https://api.spacy.io/displacy/index.html?full=Click+the+button+to+see+this+sentence+in+displaCy.)
/ [spaCy spaCy.io DOCS](https://spacy.io/docs)
/ [NetworkX](https://networkx.github.io/documentation/latest/overview.html)
/ [How do I do dependency parsing in NLTK?](http://stackoverflow.com/questions/7443330/how-do-i-do-dependency-parsing-in-nltk)
/ [dfs_tree](https://networkx.github.io/documentation/latest/reference/generated/networkx.algorithms.traversal.depth_first_search.dfs_tree.html)
/ [ARK Syntactic & Semantic Parsing Demo](http://demo.ark.cs.cmu.edu/parse)
/ [Converting a String to a List of Words?](http://stackoverflow.com/questions/6181763/converting-a-string-to-a-list-of-words)
/ [Stanford University Natural Language Processing](https://www.coursera.org/course/nlp)

</font>
