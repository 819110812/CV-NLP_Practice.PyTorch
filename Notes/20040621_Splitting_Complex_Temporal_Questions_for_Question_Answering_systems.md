# [Splitting　Complex　Temporal　Questions　for　Question　Answering　systems](http://www.aclweb.org/old_anthology/P/P04/P04-1072.pdf)

This paper presents a multi-layered QA architecture suitable for enhancing current QA capabilities with the <span style="color:red">possibility of processing complex questions</span>.

That is, questions whose answer needs to be gathered from pieces of factual information scattered in different documents.

Specifically, they have designed a layer oriented to process the different types of temporal questions. Complex temporal questions are first decomposed into simpler ones, according to the temporal relationships expressed in the original question.

In the same way, the answers of each simple question are re-composed, fulfilling the temporal restrictions of the original complex question.


## Architecture of a Question Answering System applied to Temporality

The main components of the Temporal Question Answering System are top-down: Question Decomposition Unit, General purpose Q.A. system and Answer, Recomposition Unit.

<p align="center"><img src="https://dl.dropboxusercontent.com/s/e8ev31r5z6b5vng/Screenshot%20from%202016-05-25%2015%3A10%3A12.png?dl=0" width="400" ></p>
