# [A reinforcement learning framework for answering complex questions](https://www.semanticscholar.org/paper/A-reinforcement-learning-formulation-to-the-Chali-Hasan/ecda3eb54926ed58e5aeea6271dcbe33b869f19b/pdf)

```
They use extractive multi-document summarization techniques to perform complex question answering and formulate it as a reinforcement learning problem. Given a set of complex questions, a list of relevant documents per question, and the corresponding human generated summaries (i.e. answers to the questions) as training data, the reinforcement learning module iteratively learns a number of feature weights in order to facilitate the automatic generation of summaries i.e. answers to previously unseen complex questions.
```

> So they use multi-document and corresponding human generated summaries to training.

```
A reward function is used to measure the similarities between the candidate (machine generated) summary sentences and the abstract summaries. 
```

>  This is the RL learning strategy.

```
In the training stage, the learner iteratively selects the important document sentences to be included in the candidate summary, analyzes the reward function and updates the related feature weights accordingly.The final weights are used to generate summaries as answers to unseen complex questions in the testing stage. Evaluation results show the effectiveness of our system. 
```

```
They also incorporate user interaction into the reinforcement learner to guide the candidate summary sentence selection process. 
```

> User interaction is hard to use.


# Related knowledge

QA systems can address this challenge effectively (["Advances in Open Domain Question Answering"](http://www.aclweb.org/anthology/J07-4007)).

```
Another widely known QA service is Yahoo! Answers which is a community-driven knowledge market website launched by Yahoo!. 
```


```
Furthermore, Google launched a QA system4in April 2002 that was based on paid editors. However, the system was closed in December 2006. The main limitation of these QA systems is that they rely on human expertise to help provide the answers.
```

```
QA research can handle different types of questions: fact, list, definition, how, why, etc. Some questions, which we call simple questions, are easier to answer. For example, the question: ‘‘Who is the prime minister of Canada?’’ asks for a person’sname.
```