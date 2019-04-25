# [Semantic Segmentation using Adversarial Networks](https://arxiv.org/pdf/1611.08408v1.pdf)

![](https://lh3.googleusercontent.com/-Y5qzzy9s6aA/WEWTKPcZzBI/AAAAAAAAALA/w6NmSoAcx98alGdbNgewyPwH1nWUv8mkQCLcB/s320/ss_an.PNG)


The contributions of our work are the following:

1. They present the first application of adversarial training to semantic segmentation.
2. The adversarial training approach enforces long-range spatial label contiguity, without adding complexity to the model used at test time.
3. Their experimental results on the Stanford Background and PASCAL VOC 2012 dataset show that our approach leads to improved labeling accuracy.

## Adversarial training

Give a data set of $N$ training images $x_n$ and a corresponding labe maps $y_n$, they define the loss as:

$$\begin{aligned}
\ell(\theta_s,\theta_a)&=\sum_{n=1}^N \underbrace{\ell_{mce}(s(x_n),y_n)-\lambda \overbrace{[\ell_{bce}(a(x_n,y_n),1)+\ell_{bce}(a(x_n,s(x_n)),0)]}^{adversarial \ model}}_{Segmentation \ model}\\
\ell_{mce}(\hat{y},y)&=-\sum_{i=1}^{H\times W}\sum_{i=1}^Cy_{ic}\ln \hat{y}_{iC}\\
\ell_{bce}(\hat{z},z)&=-[z\ln\hat{z}+(1-z)\ln(1-\hat{z})]
\end{aligned}
$$

where $\theta_s$ and $\theta_a$ denote the parameters of the <u>segmentation</u> and <u>adversarial</u> model respectively.
