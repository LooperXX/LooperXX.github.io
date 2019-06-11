# CS224n-2019 学习笔记

-   结合每课时的课件、笔记与推荐读物等整理而成

-   作业部分将单独整理

## Lecture 02 Word Vectors and Word Senses

**Lecture Plan**

-   Finish looking at word vectors and word2vec
-   Optimization basics
-   Can we capture this essence more effectively by counting?
-   The GloVe model of word vectors
-   Evaluating word vectors 
-   Word senses
-   The course

>    Goal: be able to read word embeddings papers by the end of class

### Review: Main idea of word2vec

![1560257599521](imgs/1560257599521.png)

$$
P(o | c)=\frac{\exp \left(u_{o}^{T} v_{c}\right)}{\sum_{w \in V} \exp \left(u_{w}^{T} v_{c}\right)}
$$

-   遍历整个语料库中的每个单词
-   使用单词向量预测周围的单词
-   更新向量以便更好地预测

**Word2vec parameters and computations**

![1560257888161](imgs/1560257888161.png)

>   $n \times d \cdot d \to 1 \rightarrow n \times 1 \overset{softmax} \to n \times 1$ 

-   每行代表一个单词的词向量，点乘后得到的分数通过softmax映射为概率分布，并且我们得到的概率分布是对于该中心词而言的上下文中单词的概率分布，该分布于上下文所在的具体位置无关，所以在每个位置的预测都是一样的
-   我们希望模型对上下文中(相当频繁)出现的所有单词给出一个合理的高概率估计
-   the, and, that, of 这样的停用词，是每个单词点乘后得到的较大概率的单词
    -   去掉这一部分可以使词向量效果更好

### Optimization: Gradient Descent

Gradient Descent 每次使用全部样本进行更新

Stochastic Gradient Descent 每次只是用单个样本进行更新

Mini-batch具有以下优点

-   通过平均值，减少梯度估计的噪音
-   在GPU上并行化运算，加快运算速度

**Stochastic gradients with word vectors**

$\nabla_{\theta} J_{t}(\theta)$ 将会非常稀疏，所以我们可能只更新实际出现的向量

解决方案

-   需要稀疏矩阵更新操作来只更新矩阵U和V中的特定行
-   需要保留单词向量的散列

如果有数百万个单词向量，并且进行分布式计算，那么重要的是不必到处发送巨大的更新

**Word2vec: More details**

为什么两个向量？

-   更容易优化，最后都取平均值
-   可以每个单词只用一个向量

两个模型变体

-   Skip-grams (SG)
    -   输入中心词并预测上下文中的单词
-   Continuous Bag of Words (CBOW)
    -   输入上下文中的单词并预测中心词

之前一直使用naive的softmax(简单但代价很高的训练方法)，接下来使用负采样方法加快训练速率

**The skip-gram model with negative sampling (HW2)**

softmax中用于归一化的分母的计算代价太高
$$
P(o | c)=\frac{\exp \left(u_{o}^{T} v_{c}\right)}{\sum_{w \in V} \exp \left(u_{w}^{T} v_{c}\right)}
$$
我们将在作业2中实现使用 **negative sampling** 负采样方法的 `skip-gram` 模型

-   使用一个 true pair (中心词及其上下文窗口中的词)与几个 noise pair (中心词与随机词搭配) 形成的样本，训练二元逻辑回归

原文中的(最大化)目标函数是

$$
J(\theta)=\frac{1}{T} \sum_{t=1}^{T} J_{t}(\theta)
$$

$$
J_{t}(\theta)=\log \sigma\left(u_{o}^{T} v_{c}\right)+\sum_{i=1}^{k} \mathbb{E}_{j \sim P(w)}\left[\log \sigma\left(-u_{j}^{T} v_{c}\right)\right]
$$

本课以及作业中的目标函数是
$$
J_{\text {neg-sample}}\left(\boldsymbol{o}, \boldsymbol{v}_{c}, \boldsymbol{U}\right)=-\log \left(\sigma\left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)\right)-\sum_{k=1}^{K} \log \left(\sigma\left(-\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)\right)
$$

-   我们希望中心词与真实上下文单词的向量点积更大，中心词与随机单词的点积更小
-   k是我们负采样的样本数目

$$
P(w)=U(w)^{3 / 4} / Z
$$

使用上式作为抽样的分布，$U(w)$ 是 unigram 分布，通过 $\frac{3}{4}$ 次方，相对减少常见单词的频率，增大稀有词的概率。 $Z$ 用于生成概率分布。

### But why not capture co-occurrence counts directly?

共现矩阵 X 

-   两个选项：windows vs. full document
-   Window ：与word2vec类似，在每个单词周围都使用Window，包括语法(POS)和语义信息
-   Word-document 共现矩阵的基本假设是在同一篇文章中出现的单词更有可能相互关联。假设单词 $i$ 出现在文章 $j$ 中，则矩阵元素 $X_{ij}$ 加一，当我们处理完数据库中的所有文章后，就得到了矩阵 X ，其大小为 $|V|\times M$ ，其中 $|V|$ 为词汇量，而 $M$ 为文章数。这一构建单词文章co-occurrence matrix的方法也是经典的Latent Semantic Analysis所采用的。{>>潜在语义分析<<}

利用某个定长窗口中单词与单词同时出现的次数来产生window-based (word-word) co-occurrence matrix。下面以窗口长度为1来举例，假设我们的数据包含以下几个句子：

-   I like deep learning.
-   I like NLP.
-   I enjoy flying.

则我们可以得到如下的word-word co-occurrence matrix:

![1560263567193](imgs/1560263567193.png)

使用共现次数衡量单词的相似性，但是会随着词汇量的增加而增大矩阵的大小，并且需要很多空间来存储这一高维矩阵，后续的分类模型也会由于矩阵的稀疏性而存在稀疏性问题，使得效果不佳。我们需要对这一矩阵进行降维，获得低维(25-1000)的稠密向量。

**Method 1: Dimensionality Reduction on X (HW1)**

使用SVD方法将共现矩阵 X 分解为 $U \Sigma V^{\top}$ ，$\sum$ 是对角线矩阵，对角线上的值是矩阵的奇异值。 $U,V$ 是对应于行和列的正交基。

为了减少尺度同时尽量保存有效信息，可保留对角矩阵的最大的k个值，并将矩阵 $U,V$ 的相应的行列保留。这是经典的线性代数算法，对于大型矩阵而言，计算代价昂贵。

**Hacks to X (several used in Rohde et al. 2005)**

按比例调整 counts 会很有效

-   对高频词进行缩放(语法有太多的影响)
    -   使用log进行缩放
    -   $min(X, t), t \approx 100$
    -   直接全部忽视
-   在基于window的计数中，提高更加接近的单词的计数
-   使用Person相关系数

>   Conclusion：对计数进行处理是可以得到有效的词向量的

![1560264868083](imgs/1560264868083.png)

$drive \to driver , swim \to swimmer, teach \to teacher$

在向量中出现的有趣的句法模式：语义向量基本上是线性组件，虽然有一些摆动，但是基本是存在动词和动词实施者的方向。

基于计数：使用整个矩阵的全局统计数据来直接估计

-   优点
    -   训练快速
    -   统计数据高效利用
-   缺点
    -   主要用于捕捉单词相似性
    -   对大量数据给予比例失调的重视

转换计数：定义概率分布并试图预测单词

-   优点
    -   提高其他任务的性能
    -   能捕获除了单词相似性以外的复杂的模式
-   缺点
    -   与语料库大小有关的量表
    -   统计数据的低效使用（**采样是对统计数据的低效使用**）

### Encoding meaning in vector differences

将两个流派的想法结合起来，在神经网络中使用计数矩阵

>   关于Glove的理论分析需要阅读原文，也可以阅读[CS224N笔记(二)：GloVe](<https://zhuanlan.zhihu.com/p/60208480>) 

**关键思想**：共现概率的比值可以对meaning component进行编码

![1560266202421](imgs/1560266202421.png)

重点不是单一的概率大小，重点是他们之间的比值，其中蕴含着meaning component。

>   例如我们想区分热力学上两种不同状态ice冰与蒸汽steam，它们之间的关系可通过与不同的单词 x 的co-occurrence probability 的比值来描述。
>
>   例如对于solid固态，虽然 $P(solid|ice)$ 与 $P(solid|steam)$ 本身很小，不能透露有效的信息，但是它们的比值$ \frac{P(solid|ice)}{P(solid|steam)}$ 却较大，因为solid更常用来描述ice的状态而不是steam的状态，所以在ice的上下文中出现几率较大
>
>   对于gas则恰恰相反，而对于water这种描述ice与steam均可或者fashion这种与两者都没什么联系的单词，则比值接近于1。所以相较于单纯的co-occurrence probability，实际上co-occurrence probability的相对比值更有意义

我们如何在词向量空间中以线性meaning component的形式捕获共现概率的比值？

log-bilinear 模型 : $w_{i} \cdot w_{j}=\log P(i | j)$

向量差异 : $w_{x} \cdot\left(w_{a}-w_{b}\right)=\log \frac{P(x | a)}{P(x | b)}$

-   如果使向量点积等于共现概率的对数，那么向量差异变成了共现概率的比率

$$
J=\sum_{i, j=1}^{V} f\left(X_{i j}\right)\left(w_{i}^{T} \tilde{w}_{j}+b_{i}+\tilde{b}_{j}-\log X_{i j}\right)^{2}
$$

-   使用平方误差促使点积尽可能得接近共现概率的对数
-   使用 $f(x)$ 对常见单词进行限制

![1560267029780](imgs/1560267029780.png)

-   优点
    -   训练快速
    -   可以扩展到大型语料库
    -   即使是小语料库和小向量，性能也很好

### How to evaluate word vectors?

与NLP的一般评估相关：内在与外在

-   内在
    -   对特定/中间子任务进行评估
    -   计算速度快
    -   有助于理解这个系统
    -   不清楚是否真的有用，除非与实际任务建立了相关性
-   外在
    -   对真实任务的评估
    -   计算精确度可能需要很长时间
    -   不清楚子系统是问题所在，是交互问题，还是其他子系统
    -   如果用另一个子系统替换一个子系统可以提高精确度

**Intrinsic word vector evaluation**

词向量类比 `a:b :: c:?`
$$
d=\arg \max _{i} \frac{\left(x_{b}-x_{a}+x_{c}\right)^{T} x_{i}}{\left\|x_{b}-x_{a}+x_{c}\right\|}
$$

-   通过加法后的余弦距离是否能很好地捕捉到直观的语义和句法类比问题来评估单词向量
-   从搜索中丢弃输入的单词
-   问题:如果有信息但不是线性的怎么办？

Glove可视化效果

![1560267650486](imgs/1560267650486.png)

![1560267659926](imgs/1560267659926.png)

![1560267671216](imgs/1560267671216.png)

>   可以使用数据集评估语法和语义上的效果

**Analogy evaluation and hyperparameters**

![1560267825912](imgs/1560267825912.png)

-   300是一个很好的词向量维度
-   不对称上下文(只使用单侧的单词)不是很好，但是这在下游任务重可能不同
-   window size 设为 8 对 Glove向量来说比较好

-   分析
    -   window size设为2的时候实际上有效的，并且对于句法分析是更好的，因为句法效果非常局部

**On the Dimensionality of Word Embedding**

利用矩阵摄动理论，揭示了词嵌入维数选择的基本的偏差与方法的权衡

当持续增大词向量维度的时候，词向量的效果不会一直变差并且会保持平稳

**Analogy evaluation and hyperparameters**

-   训练时间越长越好

![1560268494152](imgs/1560268494152.png)

-   数据集越大越好，并且维基百科数据集比新闻文本数据集要好
    -   因为维基百科就是在解释概念以及他们之间的相互关联，更多的说明性文本显示了事物之间的所有联系
    -   而新闻并不去解释，而只是去阐述一些事件

![1560268567031](imgs/1560268567031.png)

**Another intrinsic word vector evaluation**

使用 cosine similarity 衡量词向量之间的相似程度

### Word senses and word sense ambiguity

大多数单词都是多义的

-   特别是常见单词
-   特别是存在已久的单词

**Improving Word Representations Via Global Context**
**And Multiple Word Prototypes (Huang et al. 2012)**

将常用词的所有上下文进行聚类，通过该词得到一些清晰的簇，从而将这个常用词分解为多个单词，例如`bank_1, bank_2, bank_3`

虽然这很粗糙，并且有时sensors之间的划分也不是很明确甚至相互重叠

**Linear Algebraic Structure of Word Senses, with**
**Applications to Polysemy、**

-   单词在标准单词嵌入(如word2vec)中的不同含义以线性叠加(加权和)的形式存在，$f$ 指频率

$$
v_{\text { pike }}=\alpha_{1} v_{\text { pike }_{1}}+\alpha_{2} v_{\text { pike }_{2}}+\alpha_{3} v_{\text { pike }_{3}} \\ \alpha_{1}=\frac{f_{1}}{f_{1}+f_{2}+f_{3}}
$$

令人惊讶的结果，只是加权平均值就已经可以获得很好的效果

-   由于从稀疏编码中得到的概念，你实际上可以将感官分离出来(前提是它们相对比较常见)
-   可以理解为由于单词存在于高维的向量空间之中，不同的纬度所包含的含义是不同的，所以加权平均值并不会损害单词在不同含义所属的纬度上存储的信息

**Extrinsic word vector evaluation**

单词向量的外部评估：词向量可以应用于NLP的很多任务

## Notes 02  GloVe, Evaluation and Training



## Reference

以下是学习本课程时的可用参考书籍：

[《神经网络与深度学习》](<https://nndl.github.io/>)

以下是整理笔记的过程中参考的博客：

[斯坦福CS224N深度学习自然语言处理2019冬学习笔记目录](<https://zhuanlan.zhihu.com/p/59011576>)

[斯坦福NLP课程 CS224N Winter 2019 学习笔记](<https://zhuanlan.zhihu.com/p/61625439>)

[斯坦福大学 CS224n自然语言处理与深度学习笔记汇总](<https://zhuanlan.zhihu.com/p/31977759>) {>>这是针对note部分的翻译<<}

