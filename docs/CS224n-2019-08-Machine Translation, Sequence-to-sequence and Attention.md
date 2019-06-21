# CS224n-2019 学习笔记

-   结合每课时的课件、笔记与推荐读物等整理而成
-   作业部分将单独整理

## Lecture 08 Machine Translation, Sequence-to-sequence and Attention

??? abstract "Lecture Plan"

    -   引入新任务：机器翻译
    -   引入一种新的神经结构：sequence-to-sequence
        -   机器翻译是sequence-to-sequence的一个主要用例
    -   引入一种新的神经技术：注意力
        -   sequence-to-sequence通过attention得到提升

### Section 1: Pre-Neural Machine Translation

**Machine Translation**

机器翻译(MT)是将一个句子 x 从一种语言( **源语言** )转换为另一种语言( **目标语言** )的句子 y 的任务。

![1561084448114](imgs/1561084448114.png)

**1950s: Early Machine Translation**

机器翻译研究始于20世纪50年代初。

-   俄语 $\to$ 英语(冷战的推动)
-   系统主要是基于规则的，使用双语词典来讲俄语单词映射为对应的英语部分

**1990s-2010s: Statistical Machine Translation**

-   <u>核心想法</u>：从数据中学习概率模型。
-   假设我们正在翻译法语 $\to$ 英语。
-   我们想要找到最好的英语句子 y ，给定法语句子 x 

$$
argmax_yP(y|x)
$$

-   使用Bayes规则将其分解为两个组件从而分别学习

$$
argmax_yP(x|y)P(y)
$$

-   $P(x|y)$ 
    -   翻译模型
    -   分析单词和短语应该如何翻译(逼真)
    -   从并行数据中学习
-   $P(y)$ 
    -   语言模型
    -   模型如何写出好英语(流利)
    -   从单语数据中学习

-   <u>问题</u>：如何学习翻译模型 $P(x|y)$
-   首先，需要大量的并行数据(例如成对的人工翻译的法语/英语句子)

**Learning alignment for SMT**

-   <u>问题</u>：如何从并行语料库中学习翻译模型 $P(x|y)$
-   进一步分解:我们实际上想要考虑

$$
P(x,a|y)
$$



-    $a$ 是对齐，即法语句子 x 和英语句子 y 之间的单词级对应

**What is alignment?**

对齐是翻译句子中特定词语之间的对应关系。

-   注意：有些词没有对应词

![1561085294365](imgs/1561085294365.png)

**Alignment is complex**

对齐可以是多对一的

![1561085319355](imgs/1561085319355.png)

对齐可以是一对多的

![1561085341219](imgs/1561085341219.png)

有些词很丰富

![1561085385801](imgs/1561085385801.png)

![1561085404516](imgs/1561085404516.png)

对齐可以是多对多(短语级)

![1561085431218](imgs/1561085431218.png)

**Learning alignment for SMT**

-   我们学习很多因素的组合，包括
    -   特定单词对齐的概率(也取决于发送位置)
    -   特定单词具有特定生育率的概率(对应单词的数量)
    -   等等

**Decoding for SMT**

**<u>问题</u>** ：如何计算argmax

-   我们可以列举所有可能的 y 并计算概率？$\to$ 太贵了
-   使用启发式搜索算法搜索最佳翻译，丢弃概率过低的假设
-   这个过程称为解码

![1561085566749](imgs/1561085566749.png)

![1561085578462](imgs/1561085578462.png)

-   SMT是一个巨大的研究领域
-   最好的系统非常复杂
    -   数以百计的重要细节我们还没有提到
    -   系统有许多分别设计子组件工程
        -   很多功能需要设计特性来获取特定的语言现象
    -   需要编译和维护额外的资源
        -   比如等价短语表
    -   需要大量的人力来维护
        -   对于每一对语言都需要重复操作

### Section 2: Neural Machine Translation

**What is Neural Machine Translation?**

-   神经机器翻译是利用单个神经网络进行机器翻译的一种方法

-   神经网络架构称为sequence-to-sequence (又名seq2seq)，它包含两个RNNs

**Neural Machine Translation (NMT)**

![1561085889832](imgs/1561085889832.png)

-   编码器RNN生成源语句的编码
    -   源语句的编码为解码器RNN提供初始隐藏状态
-   解码器RNN是一种以编码为条件生成目标句的语言模型
-   注意：此图显示了测试时行为 $\to$ 解码器输出作为下一步的输入

**Sequence-to-sequence is versatile!**

-   序列到序列不仅仅对MT有用
-   许多NLP任务可以按照顺序进行表达
    -   摘要(长文本 $\to$ 短文本)
    -   对话(前一句话 $\to$ 下一句话)
    -   解析(输入文本 $\to$ 输出解析为序列)
    -   代码生成(自然语言 $\to$ Python代码)

-   **sequence-to-sequence** 模型是 **Conditional Language Model** 条件语言模型的一个例子
    -   语言模型，因为解码器正在预测目标句的下一个单词 y
    -   有条件的，因为它的预测也取决于源句 x
-   NMT直接计算 $P(y|x)$

$$
P(y | x)=P\left(y_{1} | x\right) P\left(y_{2} | y_{1}, x\right) P\left(y_{3} | y_{1}, y_{2}, x\right) \ldots P\left(y_{T} | y_{1}, \ldots, y_{T-1}, x\right)
$$

-   上式中最后一项为，给定到目前为止的目标词和源句 x ，下一个目标词的概率
-   **<u>问题</u>** ：如何培训NMT系统？
-   **<u>回答</u>** ：找一个大的平行语料库

![1561086295541](imgs/1561086295541.png)

-   ​	Seq2seq被优化为一个单一的系统。反向传播运行在“端到端”中

**Greedy decoding**

-   我们了解了如何生成(或“解码”)目标句，通过对解码器的每个步骤使用 argmax

![1561086920417](imgs/1561086920417.png)

-   这是贪婪解码(每一步都取最可能的单词)
-   **这种方法有问题吗？**

**Problems with greedy decoding**

-   贪婪解码没有办法撤销决定

![1561086971545](imgs/1561086971545.png)

-   如何修复？

**Exhaustive search decoding**

-   理想情况下，我们想要找到一个(长度为 T )的翻译 y 使其最大化



-   我们可以尝试计算所有可能的序列 y 
    -   这意味着在解码器的每一步 t ，我们跟踪 $V^t$ 个可能的部分翻译，其中 $V$ 是 vocab 大小
    -   这种 $O(V^T)$ 的复杂性太昂贵了！

**Beam search decoding**

-   <u>核心思想</u> ：在解码器的每一步，跟踪 k 个最可能的部分翻译(我们称之为 **hypotheses 假设** )
    -   k是Beam的大小(实际中大约是5到10)

$$
\operatorname{score}\left(y_{1}, \ldots, y_{t}\right)=\log P_{\mathrm{LM}}\left(y_{1}, \ldots, y_{t} | x\right)=\sum_{i=1}^{t} \log P_{\operatorname{LM}}\left(y_{i} | y_{1}, \ldots, y_{i-1}, x\right)
$$

-   假设 $y_{1}, \dots, y_{t}$ 有一个分数，即它的对数概率
    -   分数都是负数，分数越高越好
    -   我们寻找得分较高的假设，跟踪每一步的 top k 个部分翻译
-   波束搜索 **不一定能** 找到最优解
-   但比穷举搜索效率高得多

**Beam search decoding: example** 

Beam size = k = 2

蓝色的数字是 $\operatorname{score}\left(y_{1}, \ldots, y_{t}\right)=\sum_{i=1}^{t} \log P_{\operatorname{LM}}\left(y_{i} | y_{1}, \ldots, y_{i-1}, x\right)$ 的结果

-   计算下一个单词的概率分布
-   取前k个单词并计算分数
-   对于每一次的 k 个假设，找出最前面的 k 个单词并计算分数
-   在 $k^2$ 的假设中，保留 k 个最高的分值，如 t = 2 时，保留分数最高的 hit 和 was

![1561087894142](imgs/1561087894142.png)

**Beam search decoding: stopping criterion**

-   在贪心解码中，我们通常解码到模型产生一个 $\text{<END>}$ 令牌
    -   例如:$\text{<START>}$ he hit me with a pie $\text{<END>}$
-   在 Beam Search 解码中，不同的假设可能在不同的时间步长上产生 $\text{<END>}$ 令牌
    -   当一个假设生成了 $\text{<END>}$ 令牌，该假设完成
    -   把它放在一边，通过 Beam Search 继续探索其他假设
-   通常我们继续进行 Beam Search ，直到
    -   我们到达时间步长 T (其中 T 是预定义截止点)
    -   我们至少有 n 个已完成的假设(其中 n 是预定义截止点)

**Beam search decoding: finishing up**

-   我们有完整的假设列表
-   如何选择得分最高的？
-   我们清单上的每个假设 $y_{1}, \dots, y_{t}$ 都有一个分数

$$
\operatorname{score}\left(y_{1}, \ldots, y_{t}\right)=\log P_{\mathrm{LM}}\left(y_{1}, \ldots, y_{t} | x\right)=\sum_{i=1}^{t} \log P_{\operatorname{LM}}\left(y_{i} | y_{1}, \ldots, y_{i-1}, x\right)
$$

-   <u>问题在于</u> ：较长的假设得分较低
-   <u>修正</u> ：按长度标准化。用下式来选择top one

$$
\frac{1}{t} \sum_{i=1}^{t} \log P_{\mathrm{LM}}\left(y_{i} | y_{1}, \ldots, y_{i-1}, x\right)
$$

**Advantages of NMT**

与SMT相比，NMT有很多优点

-   更好的性能
    -   更流利
    -   更好地使用上下文
    -   更好地使用短语相似性
-   单个神经网络端到端优化
    -   没有子组件需要单独优化
-   对所有语言对使用相同的方法

**Disadvantages of NMT?**

SMT相比

-   NMT的可解释性较差
    -   难以调试
-   NMT很难控制
    -   例如，不能轻松指定翻译规则或指南
    -   安全问题

**How do we evaluate Machine Translation?**

**BLEU (Bilingual Evaluation Understudy)**

-   你将会在 Assignment 4 中看到BLEU的细节
-   BLEU将机器翻译和人工翻译(一个或多个)，并计算一个相似的分数
    -   n-gram 精度 (通常为1-4)
    -   对过于短的机器翻译的加上惩罚
-   BLEU很有用,但不完美
    -   有很多有效的方法来翻译一个句子
    -   所以一个好的翻译可以得到一个糟糕的BLEU score，因为它与人工翻译的n-gram重叠较低

**MT progress over time**

![1561102393496](imgs/1561102393496.png)

**NMT: the biggest success story of NLP Deep Learning**

神经机器翻译于2014年从边缘研究活动到2016年成为领先标准方法

-   2014：第一篇 seq2seq 的文章发布
-   2016：谷歌翻译从 SMT 换成了 NMT
-   这是惊人的
    -   由数百名工程师历经多年打造的SMT系统，在短短几个月内就被少数工程师训练过的NMT系统超越

**So is Machine Translation solved?**

-   不！
-   许多困难仍然存在
    -   词表外的单词处理
    -   训练和测试数据之间的 **领域不匹配**
    -   在较长文本上维护上下文
    -   资源较低的语言对

-   使用常识仍然很难

![1561103152409](imgs/1561103152409.png)

-   NMT在训练数据中发现偏差

![1561103171522](imgs/1561103171522.png)

-   无法解释的系统会做一些奇怪的事情

![1561103187808](imgs/1561103187808.png)

**NMT research continues**

NMT是NLP深度学习的核心任务

-   NMT研究引领了NLP深度学习的许多最新创新
-   2019年：NMT研究将继续蓬勃发展
    -   研究人员发现，对于我们今天介绍的普通seq2seq NMT系统，有很多、很多的改进。
    -   但有一个改进是如此不可或缺

![1561103279423](imgs/1561103279423.png)



### Section 3: Attention

**Sequence-to-sequence: the bottleneck problem**

![1561104067334](imgs/1561104067334.png)

-   源语句的编码需要捕获关于源语句的所有信息
-   信息瓶颈！

**Attention**

-   注意力为瓶颈问题提供了一个解决方案
-   <u>核心理念</u> ：在解码器的每一步，使用 **与编码器的直接连接** 来专注于源序列的特定部分
-   首先我们将通过图表展示(没有方程)，然后我们将用方程展示

![1561104190887](imgs/1561104190887.png)

-   将解码器部分的第一个token $\text{<START>}$ 与源语句中的每一个时间步的隐藏状态进行 Dot Product 得到每一时间步的分数
-   通过softmax将分数转化为概率分布
    -   在这个解码器时间步长上，我们主要关注第一个编码器隐藏状态(“he”)

![1561104356688](imgs/1561104356688.png)

-   利用注意力分布对编码器的隐藏状态进行加权求和
-   注意力输出主要包含来自于受到高度关注的隐藏状态的信息

![1561104403007](imgs/1561104403007.png)

-   连接的 注意力输出 与 解码器隐藏状态 ，然后用来计算 $\hat y_1$

![1561104507830](imgs/1561104507830.png)

-   有时，我们从前面的步骤中提取注意力输出，并将其输入解码器(连同通常的解码器输入)。我们在作业4中做这个。

![1561104538792](imgs/1561104538792.png)

**Attention: in equations**

-   我们有编码器隐藏状态 $h_{1}, \ldots, h_{N} \in \mathbb{R}^{h}$
-   在时间步 t 上，我们有解码器隐藏状态 $s_{t} \in \mathbb{R}^{h}$
-   我们得到这一步的注意分数

$$
e^{t}=\left[s_{t}^{T} \boldsymbol{h}_{1}, \ldots, \boldsymbol{s}_{t}^{T} \boldsymbol{h}_{N}\right] \in \mathbb{R}^{N}
$$

-   我们使用softmax得到这一步的注意分布 $\alpha^{t}$ (这是一个概率分布，和为1)

$$
\alpha^{t}=\operatorname{softmax}\left(e^{t}\right) \in \mathbb{R}^{N}
$$

-   我们使用 $\alpha^{t}$ 来获得编码器隐藏状态的加权和，得到注意力输出 $\boldsymbol{a}_{t}$

$$
\boldsymbol{a}_{t}=\sum_{i=1}^{N} \alpha_{i}^{t} \boldsymbol{h}_{i} \in \mathbb{R}^{h}
$$



-   最后，我们将注意输出 $\boldsymbol{a}_{t}$ 与解码器隐藏状态连接起来，并按照非注意seq2seq模型继续进行

$$
\left[\boldsymbol{a}_{t} ; \boldsymbol{s}_{t}\right] \in \mathbb{R}^{2 h}
$$

**Attention is great**

-   注意力显著提高了NMT性能
    -   这是非常有用的，让解码器专注于某些部分的源语句
-   注意力解决瓶颈问题
    -   注意力允许解码器直接查看源语句；绕过瓶颈
-   注意力帮助消失梯度问题
    -   提供了通往遥远状态的捷径
-   注意力提供了一些可解释性
    -   通过检查注意力的分布，我们可以看到解码器在关注什么
    -   我们可以免费得到(软)对齐
    -   这很酷，因为我们从来没有明确训练过对齐系统
    -   网络只是自主学习了对齐

**Attention is a general Deep Learning technique**

-   我们已经看到，注意力是改进机器翻译的序列到序列模型的一个很好的方法
-   <u>然而</u> ：您可以在许多体系结构(不仅仅是seq2seq)和许多任务(不仅仅是MT)中使用注意力
-   注意力的更一般定义
    -   给定一组向量 **值** 和一个向量 **查询** ，注意力是一种根据查询，计算值的加权和的技术
-   我们有时说 query attends to the values
-   例如，在seq2seq + attention模型中，每个解码器的隐藏状态(查询)关注所有编码器的隐藏状态(值)

-   直觉
    -   加权和是值中包含的信息的选择性汇总，查询在其中确定要关注哪些值
    -   注意是一种获取任意一组表示(值)的固定大小表示的方法，依赖于其他一些表示(查询)。

**There are several attention variants**

-   我们有一些值 $\boldsymbol{h}_{1}, \ldots, \boldsymbol{h}_{N} \in \mathbb{R}^{d_{1}}$ 和一个查询 $s \in \mathbb{R}^{d_{2}}$

-   注意力总是包括

    1.  计算注意力得分 $e \in \mathbb{R}^{N}$ （很多种计算方式）

    2.  采取softmax来获得注意力分布 $\alpha$
        $$
        \alpha=\operatorname{softmax}(\boldsymbol{e}) \in \mathbb{R}^{N}
        $$

    3.  使用注意力分布对值进行加权求和：从而得到注意输出 $\boldsymbol{a}$ (有时称为上下文向量)
        $$
        \boldsymbol{a}=\sum_{i=1}^{N} \alpha_{i} \boldsymbol{h}_{i} \in \mathbb{R}^{d_{1}}
        $$

**Attention variants**

有几种方法可以从 $ \boldsymbol{h}_{1}, \ldots, \boldsymbol{h}_{N} \in \mathbb{R}^{d_{1}}$ 计算 $e \in \mathbb{R}^{N}$ 和 $\boldsymbol{s} \in \mathbb{R}^{d_{2}}$ 

-   基本的点乘注意力 $\boldsymbol{e}_{i}=\boldsymbol{s}^{T} \boldsymbol{h}_{i} \in \mathbb{R}$
    -   注意：这里假设 $d_1 = d_2$
    -   这是我们之前看到的版本
-   乘法注意力 $e_{i}=s^{T} \boldsymbol{W} \boldsymbol{h}_{i} \in \mathbb{R}$
    -   $\boldsymbol{W} \in \mathbb{R}^{d_{2} \times d_{1}}$ 是权重矩阵
-   加法注意力 $e_{i}=\boldsymbol{v}^{T} \tanh \left(\boldsymbol{W}_{1} \boldsymbol{h}_{i}+\boldsymbol{W}_{2} \boldsymbol{s}\right) \in \mathbb{R}$
    -   其中 $\boldsymbol{W}_{1} \in \mathbb{R}^{d_{3} \times d_{1}}, \boldsymbol{W}_{2} \in \mathbb{R}^{d_{3} \times d_{2}}$ 是权重矩阵，$\boldsymbol{v} \in \mathbb{R}^{d_{3}}$ 是权重向量
    -   $d_3$(注意力维度)是一个超参数

-   你们将在作业4中考虑这些的相对优势/劣势！

**Summary of today’s lecture**

![1561106188514](imgs/1561106188514.png)

## Notes 06 Neural Machine Translation, Seq2seq and Attention

??? abstract "Keyphrases"

    Seq2Seq and Attention Mechanisms, Neural Machine Translation, Speech Processing




## Reference

以下是学习本课程时的可用参考书籍：

[《基于深度学习的自然语言处理》](<https://item.jd.com/12355569.html>) （车万翔老师等翻译）

[《神经网络与深度学习》](<https://nndl.github.io/>)

以下是整理笔记的过程中参考的博客：

[斯坦福CS224N深度学习自然语言处理2019冬学习笔记目录](<https://zhuanlan.zhihu.com/p/59011576>) (课件核心内容的提炼，并包含作者的见解与建议)

[斯坦福大学 CS224n自然语言处理与深度学习笔记汇总](<https://zhuanlan.zhihu.com/p/31977759>) {>>这是针对note部分的翻译<<}