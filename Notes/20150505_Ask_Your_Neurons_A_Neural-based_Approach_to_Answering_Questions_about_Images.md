## [Ask Your Neurons: A Neural-based Approach to Answering Questions about Images](http://arxiv.org/abs/1505.01121)



<p align="center"><img src="https://camo.githubusercontent.com/f8f40bf902311a4ddf0f1ecc7a1e6d520bdddf28/687474703a2f2f7333322e706f7374696d672e6f72672f736b397868396f39312f53637265656e5f53686f745f323031365f30355f30385f61745f335f33325f34355f504d2e706e67" width="500" ></p>


- This is the first paper that tries to tackle the visual QA problem

- Doubles accuracy of existing none deep-learning approaches

- Uses a **single LSTM network**, responsible for both encoding and decoding the question and answer

- Input is the **raw concatenation** of the word embedding of a word with the representation from the last layer of ImageNet

- Tested on DAQUAR, measure using accuracy and **WUPS**

- WUPS is like accuracy, but also accounts into similar words (e.g. cat, kitty)

- Introduced the idea of a blind model, so no looking at the picture

- Surprisingly, the **blind model produces only slightly worse accuracy than with the CNN input**. Meaning that the questions are biased and not diverse enough
