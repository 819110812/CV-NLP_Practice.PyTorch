## [Stacked attention networks for image question answering)(http://arxiv.org/abs/1511.02274)

<p align="center"><img src="https://camo.githubusercontent.com/a9b9da7f67a9ec5484a69cd4e45c3756bb9f84c2/687474703a2f2f7333322e706f7374696d672e6f72672f3871717872776b75642f53637265656e5f53686f745f323031365f30355f30385f61745f365f31315f33375f504d2e706e67" width="500" ></p>

- Not all parts of the image are relevant to the QA

- Introduces the use of attention layers to select which part of the CNN representation should be passed on

- Most other CNN models use the last fully connected layer in CNN

- This paper takes features from the last pooling layer

- The image is divided into 14x14 sections, and each section has 512 features from the filters

- The LSTM question embeddings and the CNN representations are passed into a stacked attention network

- Note: use stacked attention network instead of a single attention network to learn complicated relationships

- The output is a softmax layer that figures out which section of the image is important, which is linearly combined as input to a fully connected classification net
