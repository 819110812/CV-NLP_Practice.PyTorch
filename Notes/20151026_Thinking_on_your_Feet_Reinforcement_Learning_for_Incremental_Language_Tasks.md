# [Thinking on your Feet: Reinforcement Learning for Incremental Language Tasks](https://www.cs.colorado.edu/~jbg/projects/IIS-1320538.html)

[https://www.youtube.com/embed/YArUk9QcMe0]()

- [Deep Averaging Networks (DAN)](https://github.com/miyyer/dan)
- [Question Answering System](https://github.com/Pinafore/qb.git)


The goal of this project is to create algorithms that can “think on their feet”, i.e. to incrementally process input and to decide when enough information has been received to act on those data. 

This research requires innovation in two areas: 
1. content models (to make accurate predictions, even when not all available information is available) 
2. and policies (to know when to trust the outputs of the content models---or know they won't get better---versus waiting for more information).

They are applying these models to two problems: 
1. synchronous machine translation (or "machine simultaneous interpretation") 
2. and question answering (when questions are revealed one piece at a time).

For question answering, they use a specially designed dataset that challenges humans: 
1. a trivia game called quiz bowl. 
 
These questions are written so that they can be interrupted by someone who knows more about the answer; that is, harder clues are at the start of the question and easier clues are at the end of the question. The content model produces guesses of what the answer could be and the policy must decide when to accept the guess.

