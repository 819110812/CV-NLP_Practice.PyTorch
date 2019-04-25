# [Learning to Generate Images of Outdoor Scenes from Attributes and Semantic Layouts](https://arxiv.org/pdf/1612.00215.pdf)/[Project Page](http://web.cs.hacettepe.edu.tr/~karacan/projects/al-cgan/)


![](http://web.cs.hacettepe.edu.tr/~karacan/projects/al-cgan/segmentation_conditioned_model.jpeg)


## Attribute-Layout Conditioned Generative Adversarial Networks (AL-CGANs)

The generator and discriminator are formulated as $\begin{cases}G:\mathbb{R}^Z\times \mathbb{R}^S \times \mathbb{R}^{A}\rightarrow \mathbb{R}^M\\D:\mathbb{R}^S\times \mathbb{R}^A\rightarrow \{0,1\}\end{cases}$, where the <u>noise</u> vector is $Z$-dim, the <u>semantic</u> layout is $S$-dim, the <u>transient</u> attribute vector is $A$-dim.

The objective function of AL-CGAN is :

$$
\begin{aligned}
\min_G\max_DV(D,G)&=E_D+E_D\\
&=E_{x,s,a\sim p_{data}(x,s,a)[\log D(x,s,a)]}+E_{x\sim p_z(z);s, a\sim p_{data}(s,a)}[\log(1-D(G(z,a)))]
\end{aligned}
$$

where $z\sim \mathcal{N}(0,1)$, and $s$ is the spatial layout, and $a$ is transient attributes.
