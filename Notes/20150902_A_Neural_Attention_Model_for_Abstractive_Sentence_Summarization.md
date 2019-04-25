## [A Neural Attention Model for Abstractive Sentence Summarization](http://arxiv.org/abs/1509.00685)

TLDR; The authors apply a neural seq2seq model to sentence summarization. The model uses an attention mechanism (soft alignment).


#### Key Points

- Summaries generated on the sentence level, not paragraph level
- Summaries have fixed length output
- Beam search decoder
- Extractive tuning for scoring function to encourage the model to take words from the input sequence
- Training data: Headline + first sentence pair.


在CS领域，beam search是一种启发式搜索，在优化领域，其属于一种最佳优先算法，最佳优先算法是一种图搜索算法，其会将所有可能的解依据启发式规则进行排序，该规则用来衡量得到的解与目标解到底有多接近。但是对于beam search与最佳优先算法有有一些地方不同，beam search只会保存一部分解作为候选解，而最佳优先算法则会将所有解都作为候选，其具体过程如下所述:
beam search是使用宽度优先搜索来构建它的搜索树。在每一层，其都会生成一系列的解，然后对这些解进行排序，选择最好的K个解作为候选解，这里的K我们称为集束宽度。只有被选中的这些解可以向下继续扩展下去。因此，集束宽度越大，被裁减掉的解越少。由于存在裁减，目标解有可能会被裁减掉，因此该算法是不完全的，即无法保证能够找到全局最优解

在seq2seq中在test阶段使用了beam search来寻找解码时最优的结果，我们假设集束宽度为2，词典大小为3（a,b,c），那么其解码过程如下所示:

- 生成第1个词的时候，选择概率最大的2个词，假设为a,c，那么当前序列就是a,c；
- 生成第2个词的时候，我们将当前序列a和c，分别与词表中的所有词进行组合，得到新的6个序列aa ab ac ca cb cc，然后从其中选择2个得分最高的，作为当前序列，加入aa cb
- 后面不断重复这个过程，直到遇到结束符为止。最终输出2个得分最高的序列
