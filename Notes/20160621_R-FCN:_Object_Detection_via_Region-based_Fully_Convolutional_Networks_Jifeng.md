# [R-FCN: Object Detection via Region-based Fully Convolutional Networks]()

Compared with R-CNN/Faster R-CNN, their region-based detector is fully convolutional with almost all computation shared on the entire image. They apply **position-sensitive score maps** to address a dilemma between **translation-invariance** in image classification and **translation-variance** in object detection.

## CONTRIBUTIONS

1. A framework called Region-based Fully Convolutional Network (R-FCN) is developed for object detection, which consists of shared, fully convolutional architectures.
2. A set of position-sensitive score maps are introduced to enable FCN representing translation variance.
3. A unique ROI pooling method is proposed to shepherd information from mentioned score maps.

![](http://joshua881228.webfactional.com/media/uploads/ReadingNote/arXiv_R-FCN/R-FCN.jpg)


## method

1. The image is processed by a FCN manner network.
2. At the end of FCN, a RPN (Region Proposal Network) is used to generate ROIs.
3. On the other hand, a score map of $k^2(C+1)$ channels is generated using a bank of specialized convolutional layers.
4. For each ROI, a selective ROI pooling is utilized to generate a $C+1$ channel score map.
5. The scores in the score map are averaged to vote for category.
6. Another $4k^2$ dim convolutional layer is learned for bounding box regression.

Position sensitive score maps and RoI pooling:
$$
r_C(i,j|\Theta)=\sum_{(x,y)\in \text{bin}(ij)}z_{i,j,C}(x+x_0,y+y_0|\Theta)/n
$$

## Training Details
1. R-FCN is trained end-to-end with pre-computed region proposals. Both category and position are learnt with the loss function: $L(s,t_x,y,w,h)=L_{cls}(s_c^{\ast})+\lambda[c^{\star}>0]L_{reg}(t,t^{\ast})$. The first loss is classification loss and the second loss is detection loss.
2. For each image, N proposals are generated and B out of N proposals are selected to train weights according to the highest losses. B is set to 128 in this work.
3. 4-step alternating training is utilized to realizing feature sharing between R-FCN and RPN.

## ADVANTAGES
1. It is fast (170ms/image, 2.5-20x faster than Faster R-CNN).
2. End-to-end training is easier to process.
3. All learnable layers are convolutional and shared on the entire image, yet encode spatial information required for object detection.

## DISADVANTAGES

Compared with Single Shot methods, more computation resource is needed.
