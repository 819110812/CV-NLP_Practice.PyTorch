# [Generating images with recurrent adversarial networks](http://arxiv.org/pdf/1602.05110v4.pdf)

<iframe width="560" height="315" src="https://www.youtube.com/embed/QPkb5VcgXAM" frameborder="0" allowfullscreen></iframe>

http://arxiv.org/abs/1602.05110

![](https://www.dropbox.com/s/yoyz8pnjk3ecupn/Generating%20images%20with%20recurrent%20adversarial%20networks.png?dl=1)

- Motivation: generate good image samples
- Abbreviation: GRAN
- Similar models: LAGAN and DRAW
- Idea:
    - image generation is an iterative process.
    - Start with a sample from the distribution of latent variables, feed it to the decoder to get an image. Feed the generated sample to the encoder, which will generate a vector representation of the image. Concatenate that representation with the sample from the distribution and feed that to the decoder, to generate another image. This process is repeated t times, with t fixed aprori. To generate the final image, the t image generated samples are added together and the tanh function is applied to ensure the final image has entries in between 0 and 1.
    - encoders and decoders can be represented by any function, they use deep convolutional adversarial nets
- Evaluation idea:
    - to evaluate two GAN models, with discriminator D1 and generator G1 and discriminator D2 and generator G2, one cam compare them by seeing how well D1 can discriminate samples from G2 and how well D2 can discriminate samples from G1
- Differences between proposed model and DRAW:
    - in DRAW at each time step, a new sample from the latent space is generated. In GRAN, only one sample is generated and then reused at each time step.
    - GRAN starts with the decoding phase, not with the encoding phase
    - GRAN has no attention mechanism
    - A discriminator from GRAN can differentiate between DRAW generated samples and images from MNIST with a 10% error
