# [Convolutional Sequence to Sequence Learning](http://pwnote.paperweekly.site/paper_detail?uri=http://cn.arxiv.org/pdf/1705.03122)

![](https://camo.githubusercontent.com/9aa0e6eca9c3ed3c1607079f44fcca97387f2d93/68747470733a2f2f73636f6e74656e742d736561312d312e78782e666263646e2e6e65742f762f7433392e323336352d362f31383135383931325f3832313531313531343636383333375f383735383039363631303437363432353231365f6e2e6769663f5f6e635f6c6f673d31266f683d6331353361656564386637346538633636613831303639353138653362303539266f653d3539414446453235)

![](https://pic2.zhimg.com/v2-ca881b1ada1ed96d58c956331ee84315_b.png)


## Weakness
1. <font style="color:red">Their results are based on trainingwith 8 GPUs for about 37 days and batch size 32 on each worker.</font>
2. During the inference period, generating words with CNN still an recurrent process, as it is a temporal information. I'm not sure whether it is really boost up so much.


## Reference
1. [Convolutional Sequence to Sequence Learning](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1705.03122)
2. [Language modeling with gated linear units](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1612.08083)
