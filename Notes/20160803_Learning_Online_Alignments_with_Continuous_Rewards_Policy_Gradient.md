# [Learning Online Alignments with Continuous Rewards Policy Gradient](https://arxiv.org/pdf/1608.01281v1.pdf)

<iframe width="560" height="315" src="https://www.youtube.com/embed/KHZVXao4qXs" frameborder="0" allowfullscreen></iframe>

<iframe width="560" height="315" src="https://www.youtube.com/embed/_tn-FN-jEPk" frameborder="0" allowfullscreen></iframe>

TLDR: They present a new method forsolving sequence-to-sequence problems using <u>hard online alignments</u> instead of soft offline alignments. They use <u>hard binary stochastic decisions</u> to select the timesteps at which outputs will be produced. At each time step $i$, a RNN decides whether to emit an output token (with a stochastic binary logistic unit $b_i$), the previous time step $\tilde{b}_{i-1}$ and the previous target $t_{i-1}$ are fed into the model as input. This feedback ensures that the model’s outputs are maximallydependent and thus the model is from the sequence to sequence family.

## Their method

![](https://www.dropbox.com/s/4y7ja0t49xqro8h/Online_Alignments_Continuous_Rewards_Policy_Gradient.png?dl=1)

## Conclusions

In this work, they presented a simple model that can solve sequence-to-sequence problems without the need to process the entire input sequence first. Their model directly maximizes the log probability ofthe correct answer by combining standard supervised backpropagation and a policy gradient method. Their results also suggest that policy gradient methods are reasonably powerful, and that they can trainhighly complex neural networks that learn to make nontrivial stochastic decisions
