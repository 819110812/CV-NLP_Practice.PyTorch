# [Interpretable Convolutional Neural Networks](https://arxiv.org/abs/1710.00935)

## CONTRIBUTION
1. Slightly revised CNNs are propsed to improve their interpretability, which can be broadly applied to CNNs with different network structures.
2. No annotations of object parts and/or textures are needed to ensure each high-layer filter to have a certain semantic meaning. Each filter automatically learns a meaningful object-part representation without any additional human supervision.
3. When a traditional CNN is modified to an interpretable CNN, experimental settings need not to be changed for learning. I.e. the interpretable CNN does not change the previous loss function on the top layer and uses exactly the same training samples.
4. The design for interpretability may decrease the discriminative power of the network a bit, but such a decrease is limited within a small range.


![](https://raw.githubusercontent.com/joshua19881228/my_blogs/master/Computer_Vision/Reading_Note/figures/Reading_Note_20171012_interpretableCNN.png)
