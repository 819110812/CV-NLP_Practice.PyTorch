# [Language Generation with Recurrent Generative Adversarial Networks without Pre-training](https://arxiv.org/pdf/1706.01399.pdf)

This paper first points out that MLE has some drawbacks.
1. First, MLE suffers from "explore bias", that is, at training time the model is exposed to gold data, but at test time it observes its own prediction. Thus, wrong predictions quickly accumulate, resulting in bad text generation.
2. The MLE loss function is very stringent, A languge model trained with MLE objective aims to allocate all probability mass to the i-th character of the training set given the previous i-1 characters.

In generative adversarial training, the objective of fooling the discriminator is more dynamic and can evolve as the training precess unfolds.

- The loss of the improved WGAN generator is:
$$
L_G= -\mathbb{E}_{\tilde{x}\sim \mathbb{P}_g}[D(\tilde{x})]
$$
- The loss of the discriminator is :
$$
L_D = \mathbb{E}_{\tilde{x}\sim \mathbb{P}_g}[D(\tilde{x})]-\mathbb{E}_{x\sim \mathbb{P}_r}[D(x)]+\lambda \mathbb{E}_{\hat{x}\sim \mathbb{P}_x}[(\|\nabla_{\hat{x}}D(\hat{x})\|_2-1)^2]
$$

where the last term of the objective controls for the complexity of the discriminator function and penalizes functions that have high gradient norm and change too rapidly.

## Recurrent Models

More discussions can be found [here](https://www.reddit.com/r/MachineLearning/comments/6fl9cl/r_language_generation_with_recurrent_generative/)
