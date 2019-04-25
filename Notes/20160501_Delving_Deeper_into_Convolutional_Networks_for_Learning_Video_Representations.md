# [Delving Deeper into Convolutional Networks for Learning Video Representations](https://arxiv.org/abs/1511.06432)

## Motivation
Previous works on Recurrent CNNs has tended to focus on high-level features extracted from the 2D CNN top-layers. High-level features contain highly discriminative information, they tend to have a low-spatial resolution. Thus, we argue that current RCN architectures are not well suited for capturing fine motion information. Instead, they are more
likely focus on global appearance changes.

Low-level features, on the other hand, preserve a higher spatial resolution from which we can model finer motion patterns. However, applying an RNN directly on intermediate convolutional maps, inevitably results in a drastic number of parameters characterizing the input-to-hidden transformation due to the convolutional maps size. On the other hand, convolutional maps preserve the frame spatial topology. To leverage these, we extend the GRU model and replace the fc RNN linear product operation with a convolution. Our GRU extension therefore encodes the locality and temporal smoothness prior of videos directly in the model structure. Thus, all neurons in the CNN are recurrent.

## Architecture
See Fig. 19.10. The inputs are RGB and flow representations of videos. Networks are pre-trained on ImageNet. We apply average pooling on the hidden-representations of the last time-step to reduce their spatial dimension to 1 × 1, and feed the representations to 5 classifiers, composed by a linear layer with a softmax nonlineary. The classifier outputs are then averaged to get the final decision.
