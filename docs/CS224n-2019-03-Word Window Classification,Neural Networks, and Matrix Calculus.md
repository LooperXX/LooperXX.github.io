# CS224n-2019 学习笔记

-   结合每课时的课件、笔记与推荐读物等整理而成

-   作业部分将单独整理

## Lecture 03 Word Window Classification,Neural Networks, and Matrix Calculus

??? abstract "Lecture Plan"

      -   Classification review/introduction
      -   Neural networks introduction
      -   Named Entity Recognition
      -   Binary true vs. corrupted word window classification
      -   Matrix calculus introduction

!!! info "提示"

    这对一些人而言将是困难的一周，课后需要阅读提供的资料。

### Classification setup and notation

通常我们有由样本组成的训练数据集

$$
\left\{x_{i}, y_{i}\right\}_{i=1}^{N}
$$

 $x_i$ 是输入，例如单词（索引或是向量），句子，文档等等，维度为 $d$

 $y_i$ 是我们尝试预测的标签（ $C$ 个类别中的一个），例如：

-   类别：感情，命名实体，购买/售出的决定
-   其他单词
-   之后：多词序列的

**Classification intuition**

![1560343189656](imgs/1560343189656.png)

训练数据： $\left\{x_{i}, y_{i}\right\}_{i=1}^{N}$

简单的说明情况

-   固定的二维单词向量分类
-   使用softmax/logistic回归
-   线性决策边界

**传统的机器学习/统计学方法**：假设 $x_i$ 是固定的，训练 softmax/logistic 回归的权重 $W \in \mathbb{R}^{C \times d}$ 来决定决定边界(超平面)

**方法**：对每个 $x$ ，预测
$$
p(y | x)=\frac{\exp \left(W_{y} . x\right)}{\sum_{c=1}^{C} \exp \left(W_{c} \cdot x\right)}
$$
我们可以将预测函数分为两个步骤：

1.  将 $W$ 的 $y^{th}$ 行和 $x$ 中的对应行相乘得到分数
    $$
    W_{y} \cdot x=\sum_{i=1}^{d} W_{y i} x_{i}=f_{y}
    $$
    
    计算所有的 $f_c, for \ c=1,\dots,C$

2.  使用softmax函数获得归一化的概率

$$
p(y | x)=\frac{\exp \left(f_{y}\right)}{\sum_{c=1}^{C} \exp \left(f_{c}\right)}=\operatorname{softmax}\left(f_{y}\right)
$$

**Training with softmax and cross-entropy loss**

对于每个训练样本 $(x,y)$ ，我们的目标是最大化正确类 $y$ 的概率，或者我们可以最小化该类的负对数概率
$$
-\log p(y | x)=-\log \left(\frac{\exp \left(f_{y}\right)}{\sum_{c=1}^{C} \exp \left(f_{c}\right)}\right)
$$
**Background: What is “cross entropy” loss/error?**

-   交叉熵”的概念来源于信息论，衡量两个分布之间的差异
-   令真实概率分布为 $p$
-   令我们计算的模型概率为 $q$
-   交叉熵为

$$
H(p, q)=-\sum_{c=1}^{C} p(c) \log q(c)
$$

-   假设 groud truth (or true or gold or target)的概率分布在正确的类上为1，在其他任何地方为0：$p = [0,…,0,1,0,…0]$ 
-   因为 $p$ 是独热向量，所以唯一剩下的项是真实类的负对数概率

**Classification over a full dataset**

在整个数据集 $\left\{x_{i}, y_{i}\right\}_{i=1}^{N}$ 上的交叉熵损失函数，是所有样本的交叉熵的均值

$$
J(\theta)=\frac{1}{N} \sum_{i=1}^{N}-\log \left(\frac{e^{f_{y_{i}}}}{\sum_{c=1}^{C} e^{f_{c}}}\right)
$$

我们不使用

$$
f_{y}=f_{y}(x)=W_{y} \cdot x=\sum_{j=1}^{d} W_{y j} x_{j}
$$

我们使用矩阵来表示 $f$

$$
f = Wx
$$

**Traditional ML optimization**

-   一般机器学习的参数 $\theta$ 通常只由W的列组成

$$
\theta=\left[\begin{array}{c}{W_{\cdot 1}} \\ {\vdots} \\ {W_{\cdot d}}\end{array}\right]=W( :) \in \mathbb{R}^{C d}
$$

-   因此，我们只通过以下方式更新决策边界

$$
\nabla_{\theta} J(\theta)=\left[\begin{array}{c}{\nabla_{W_{1}}} \\ {\vdots} \\ {\nabla_{W_{d}}}\end{array}\right] \in \mathbb{R}^{C d}
$$

### Neural Network Classifiers

![1560345898614](imgs/1560345898614.png)

-   单独使用Softmax(≈logistic回归)并不十分强大
-   Softmax只给出线性决策边界
    -   这可能是相当有限的，当问题很复杂时是无用的
    -   纠正这些错误不是很酷吗?

**Neural Nets for the Win!**

神经网络可以学习更复杂的函数和非线性决策边界

![1560346033994](imgs/1560346033994.png)

??? tip "更高级的分类需要"

    -   词向量
    -   更深层次的深层神经网络

**Classification difference with word vectors**

一般在NLP深度学习中

-   我们学习了矩阵 $W$ 和词向量 $x$
-   我们学习传统参数和表示
-   词向量是对独热向量的重新表示——在中间层向量空间中移动它们——以便使用(线性)softmax分类器通过 x = Le 层进行分类
    -   即将词向量理解为一层神经网络，输入单词的独热向量并获得单词的词向量表示，并且我们需要对其进行更新。其中，$Vd$ 是数量很大的参数

$$
\nabla_{\theta} J(\theta)=\left[\begin{array}{c}{\nabla_{W_{1}}} \\ {\vdots} \\ {\nabla_{W_{d a r d v a r k}}} \\ {\vdots} \\ {\nabla_{x_{z e b r a}}}\end{array}\right] \in \mathbb{R}^{C d + V d}
$$

**Neural computation**

![1560346664232](imgs/1560346664232.png)

**An artificial neuron**

-   神经网络有自己的术语包
-   但如果你了解 softmax 模型是如何工作的，那么你就可以很容易地理解神经元的操作

![1560346716435](imgs/1560346716435.png)

**A neuron can be a binary logistic regression unit**

$f = nonlinear activation fct. (e.g. sigmoid), w = weights, b = bias, h = hidden, x = inputs$
$$
\begin{array}{l}{h_{w, b}(x)=f\left(w^{\top} x+b\right)} \\ {f(z)=\frac{1}{1+e^{-z}}}\end{array}
$$
$b$ : 我们可以有一个“总是打开”的特性，它给出一个先验类，或者将它作为一个偏向项分离出来

$w,b$ 是神经元的参数

**A neural network**
**= running several logistic regressions at the same time**

![1560347357837](imgs/1560347357837.png)

如果我们输入一个向量通过一系列逻辑回归函数，那么我们得到一个输出向量，但是我们不需要提前决定这些逻辑回归试图预测的变量是什么。

![1560347481494](imgs/1560347481494.png)

我们可以输入另一个logistic回归函数。损失函数将指导中间隐藏变量应该是什么，以便更好地预测下一层的目标。我们当然可以使用更多层的神经网络。

**Matrix notation for a layer**

![1560347762809](imgs/1560347762809.png)
$$
\begin{array}{l}{a_{1}=f\left(W_{11} x_{1}+W_{12} x_{2}+W_{13} x_{3}+b_{1}\right)} \\ {a_{2}=f\left(W_{21} x_{1}+W_{22} x_{2}+W_{23} x_{3}+b_{2}\right)}\\ {z=W x+b} \\ {a=f(z)} \\ f\left(\left[z_{1}, z_{2}, z_{3}\right]\right)=\left[f\left(z_{1}\right), f\left(z_{2}\right), f\left(z_{3}\right)\right] \end{array}
$$

-   $f(x)$ 在运算时是 element-wise 逐元素的

**Non-linearities (aka “f ”): Why they’re needed**

例如：函数近似，如回归或分类

-   没有非线性，深度神经网络只能做线性变换
-   多个线性变换可以组成一个的线性变换 $W_1 W_2 x = Wx$ 
    -   因为线性变换是以某种方式旋转和拉伸空间，多次的旋转和拉伸可以融合为一次线性变换
-   对于非线性函数而言，使用更多的层，他们可以近似更复杂的函数

### Named Entity Recognition (NER)

-   任务：例如，查找和分类文本中的名称

![1560359392887](imgs/1560359392887.png)

-   可能的用途
    -   跟踪文档中提到的特定实体（组织、个人、地点、歌曲名、电影名等）
    -   对于问题回答，答案通常是命名实体
    -   许多需要的信息实际上是命名实体之间的关联
    -   同样的技术可以扩展到其他 slot-filling 槽填充 分类
-   通常后面是命名实体链接/规范化到知识库

**Named Entity Recognition on word sequences**

![1560359650543](imgs/1560359650543.png)

我们通过在上下文中对单词进行分类，然后将实体提取为单词子序列来预测实体

**Why might NER be hard?**

-   很难计算出实体的边界
    -   ![1560359674788](imgs/1560359674788.png)
    -   第一个实体是 “First National Bank” 还是 “National Bank”
-   很难知道某物是否是一个实体
    -   是一所名为“Future School” 的学校，还是这是一所未来的学校？
-   很难知道未知/新奇实体的类别
    -   ![1560359774508](imgs/1560359774508.png)
    -   “Zig Ziglar” ?  一个人
-   实体类是模糊的，依赖于上下文
    -   ![1560359806734](imgs/1560359806734.png)
    -   这里的“Charles Schwab”  是 PER
        不是 ORG

### Binary word window classification

为在上下文中的语言构建分类器

-   一般来说，很少对单个单词进行分类
-   有趣的问题，如上下文歧义出现
-   例子：auto-antonyms
    -   "To sanction" can mean "to permit" or "to punish”
    -   "To seed" can mean "to place seeds" or "to remove seeds"
-   例子：解决模糊命名实体的链接
    -   Paris → Paris, France vs. Paris Hilton vs. Paris, Texas
    -   Hathaway → Berkshire Hathaway vs. Anne Hathaway

**Window classification**

-   思想：在**相邻词的上下文窗口**中对一个词进行分类
-   例如，上下文中一个单词的命名实体分类
    -   人、地点、组织、没有
-   在上下文中对单词进行分类的一个简单方法可能是对窗口中的单词向量进行**平均**，并对平均向量进行分类
    -   问题：**这会丢失位置信息**

**Window classification: Softmax**

-   训练softmax分类器对中心词进行分类，方法是在一个窗口内**将中心词周围的词向量串联起来**
-   例子：在这句话的上下文中对“Paris”进行分类，窗口长度为2

![1560360448681](imgs/1560360448681.png)

-   结果向量 $x_{window} = x \in R^{5d}$  是一个列向量

**Simplest window classifier: Softmax**

对于 $x = x_{window}$ ，我们可以使用与之前相同的softmax分类器

![1560360599779](imgs/1560360599779.png)

-   如何更新向量？
-   简而言之：就像上周那样，求导和优化

**Binary classification with unnormalized scores**

-   之前的例子：$X_{\text { window }}=[\begin{array}{ccc}{\mathrm{X}_{\text { museums }}} & {\mathrm{X}_{\text { in }}} & {\mathrm{X}_{\text { paris }} \quad \mathrm{X}_{\text { are }} \quad \mathrm{X}_{\text { amazing }} ]}\end{array}$

-   假设我们要对中心词是否为一个地点，进行分类
-   与word2vec类似，我们将遍历语料库中的所有位置。但这一次，它将受到监督，只有一些位置能够得到高分。
-   例如，在他们的中心有一个实际的NER Location的位置是“真实的”位置会获得高分

**Binary classification for NER Location**

-   例子：Not all museums in Paris are amazing

-   这里：一个真正的窗口，以Paris为中心的窗口和所有其他窗口都“损坏”了，因为它们的中心没有指定的实体位置。
    -   museums in Paris are amazing

-   “损坏”窗口很容易找到，而且有很多：任何中心词没有在我们的语料库中明确标记为NER位置的窗口
    -   Not all museums in Paris

**Neural Network Feed-forward Computation**

使用神经激活 $a$ 简单地给出一个非标准化的分数
$$
score(x)=U^{T} a \in \mathbb{R}
$$
我们用一个三层神经网络计算一个窗口的得分

-   $s = score("museums  \ in \ Paris \ are \ amazing”)$

$$
\begin{array}{l}{s=U^{T} f(W x+b)} \\ {x \in \mathbb{R}^{20 \times 1}, W \in \mathbb{R}^{8 \times 20}, U \in \mathbb{R}^{8 \times 1}}\end{array}
$$

![1560361207976](imgs/1560361207976.png)

**Main intuition for extra layer**

中间层学习输入词向量之间的**非线性交互**

例如：只有当“museum”是第一个向量时，“in”放在第二个位置才重要

**The max-margin loss**

![1560361550807](imgs/1560361550807.png)

-   关于训练目标的想法：让真实窗口的得分更高，而破坏窗口的得分更低(直到足够好为止)
-   $s = score("museums  \ in \ Paris \ are \ amazing”)$
-   $s_c = score("Not \ all \ museums  \ in \ Paris)$
-   最小化 $J=\max \left(0,1-s+s_{c}\right)$
-   这是不可微的，但它是连续的→我们可以用SGD。

-   每个选项都是连续的

-   单窗口的目标函数为

$$
J=\max \left(0,1-s+s_{c}\right)
$$

-   每个中心有NER位置的窗口的得分应该比中心没有位置的窗口高1分

![1560361673756](imgs/1560361673756.png)

-   要获得完整的目标函数：为每个真窗口采样几个损坏的窗口。对所有培训窗口求和
-   类似于word2vec中的负抽样

-   使用SGD更新参数
    -   $\theta^{n e w}=\theta^{o l d} - \alpha \nabla_{\theta} J(\theta)$
    -   $a$ 是 步长或是学习率
-   如何计算 $\nabla_{\theta} J(\theta)$ ？
    -   手工计算（本课）
    -   算法：反向传播（下一课）

**Computing Gradients by Hand**

-   回顾多元导数
-   矩阵微积分：完全矢量化的梯度
    -   比非矢量梯度快得多，也更有用
    -   但做一个非矢量梯度可以是一个很好的实践；以上周的讲座为例
    -   **notes** 更详细地涵盖了这些材料

### Gradients

-   给定一个函数，有1个输出和1个输入

$$
f(x) = x^3
$$

-   斜率是它的导数

$$
\frac{d f}{d x}=3 x^{2}
$$

-   给定一个函数，有1个输出和 n 个输入

$$
f(\boldsymbol{x})=f\left(x_{1}, x_{2}, \ldots, x_{n}\right)
$$

-   梯度是关于每个输入的偏导数的向量

$$
\frac{\partial f}{\partial \boldsymbol{x}}=\left[\frac{\partial f}{\partial x_{1}}, \frac{\partial f}{\partial x_{2}}, \ldots, \frac{\partial f}{\partial x_{n}}\right]
$$

**Jacobian Matrix: Generalization of the Gradient**

-   给定一个函数，有 m 个输出和 n 个输入

$$
\boldsymbol{f}(\boldsymbol{x})=\left[f_{1}\left(x_{1}, x_{2}, \ldots, x_{n}\right), \ldots, f_{m}\left(x_{1}, x_{2}, \ldots, x_{n}\right)\right]
$$

-   其雅可比矩阵是一个$m \times n$的偏导矩阵

$$
\frac{\partial \boldsymbol{f}}{\partial \boldsymbol{x}}=\left[\begin{array}{ccc}{\frac{\partial f_{1}}{\partial x_{1}}} & {\cdots} & {\frac{\partial f_{1}}{\partial x_{n}}} \\ {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial f_{m}}{\partial x_{1}}} & {\cdots} & {\frac{\partial f_{m}}{\partial x_{n}}}\end{array}\right]
$$

$$
\left(\frac{\partial f}{\partial x}\right)_{i j}=\frac{\partial f_{i}}{\partial x_{j}}
$$

**Chain Rule**

对于单变量函数：乘以导数
$$
\begin{array}{l}{z=3 y} \\ {y=x^{2}} \\ {\frac{d z}{d x}=\frac{d z}{d y} \frac{d y}{d x}=(3)(2 x)=6 x}\end{array}
$$
对于一次处理多个变量：乘以雅可比矩阵
$$
\begin{array}{l}{\textbf{h}=f(\textbf{z})} \\ {\textbf{z}=\textbf{W} \textbf{x}+\textbf{b}} \\ {\frac{\partial \textbf{h}}{\partial \textbf{x}}=\frac{\partial \textbf{h}}{\partial \textbf{z}} \frac{\partial \textbf{z}}{\partial \textbf{x}}=\dots}\end{array}
$$
**Example Jacobian: Elementwise activation Function**

$h=f(z)$ , $\frac{\partial \textbf{h}}{\partial \textbf{z}} = ?, \textbf{h},\textbf{z} \in \mathbb{R}^{n}$  

由于使用的是 element-wise，所以 $h_{i}=f\left(z_{i}\right)$

函数有n个输出和n个输入 → n×n 的雅可比矩阵

$$
\begin{aligned}\left(\frac{\partial h}{\partial z}\right)_{i j} &=\frac{\partial h_{i}}{\partial z_{j}}=\frac{\partial}{\partial z_{j}} f\left(z_{i}\right), \text{definition of Jacobian} \\ &=\left\{\begin{array}{ll}{f^{\prime}\left(z_{i}\right)} & {\text { if } i=j} \\ {0} & {\text { if otherwise }} , \text{regular 1-variable derivative} \end{array}\right.\end{aligned}
$$

$$
\frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}}= \left(\begin{array}{ccc}{f^{\prime}\left(z_{1}\right)} & { } & {0} \\ {} & {\ddots} & { } \\ {0} & { } & {f^{\prime}\left(z_{n}\right)}\end{array}\right)=\operatorname{diag}\left(\boldsymbol{f}^{\prime}(\boldsymbol{z})\right)
$$

**Other Jacobians**
$$
\begin{array}{l}{\frac{\partial}{\partial \boldsymbol{x}}(\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b})=\boldsymbol{W}} \\ {\frac{\partial}{\partial \boldsymbol{b}}(\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b})=\boldsymbol{I} \text { (Identity matrix) }} \\ \frac{\partial}{\partial \boldsymbol{u}}\left(\boldsymbol{u}^{T} \boldsymbol{h}\right)=\boldsymbol{h}^{\boldsymbol{T}} \end{array}
$$
这是正确的雅可比矩阵。稍后我们将讨论“形状约定”；用它则答案是 $h$ 。

**Back to our Neural Net!**

![1560363626037](imgs/1560363626037.png)

如何计算 $\frac{\partial s}{\partial b}$ ？

实际上，我们关心的是损失的梯度，但是为了简单起见，我们将计算分数的梯度

**Break up equations into simple pieces**

![1560363713598](imgs/1560363713598.png)

**Apply the chain rule**
$$
\begin{array}{l}{s=\boldsymbol{u}^{T} \boldsymbol{h}} \\ {\boldsymbol{h}=f(\boldsymbol{z})} \\ {\boldsymbol{z}=\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b}} \\ {\boldsymbol{x}} \text{ (input) }\end{array}
$$

$$
\frac{\partial s}{\partial \boldsymbol{b}}=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{b}}
$$

![1560363929934](imgs/1560363929934.png)

如何计算 $\frac{\partial s}{\partial \textbf{W}}$ ？
$$
\begin{aligned} \frac{\partial s}{\partial \boldsymbol{W}} &=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}} \\ \frac{\partial s}{\partial \boldsymbol{b}} &=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{b}} \end{aligned}
$$
前两项是重复的，无须重复计算
$$
\begin{aligned} \frac{\partial s}{\partial \boldsymbol{W}} &=\boldsymbol{\delta} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}} \\ \frac{\partial s}{\partial \boldsymbol{b}} &=\boldsymbol{\delta} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{b}}=\boldsymbol{\delta} \\ \boldsymbol{\delta} &=\frac{\partial s}{\partial \boldsymbol{h}} \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}}=\boldsymbol{u}^{T} \circ f^{\prime}(\boldsymbol{z}) \end{aligned}
$$
其中，$\delta$ 是局部误差符号

**Derivative with respect to Matrix: Output shape**

-   $\boldsymbol{W} \in \mathbb{R}^{n \times m}$ ，$\frac{\partial s}{\partial \boldsymbol{W}}$ 的形状是
-   1个输出，$n \times m$ 个输入：1 × nm 的雅可比矩阵？
    -   不方便更新参数 $\theta^{n e w}=\theta^{o l d}-\alpha \nabla_{\theta} J(\theta)$
-   而是遵循惯例：导数的形状是参数的形状 （形状约定）
    -   $\frac{\partial s}{\partial \boldsymbol{W}}$ 的形状是 $n \times m$ 

$$
\left[\begin{array}{ccc}{\frac{\partial s}{\partial W_{11}}} & {\cdots} & {\frac{\partial s}{\partial W_{1 m}}} \\ {\vdots} & {\ddots} & {\vdots} \\ {\frac{\partial s}{\partial W_{n 1}}} & {\cdots} & {\frac{\partial s}{\partial W_{n m}}}\end{array}\right]
$$

**Derivative with respect to Matrix**

-   $\frac{\partial s}{\partial \boldsymbol{W}}=\boldsymbol{\delta} \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}}$
    -   $\delta$ 将出现在我们的答案中
    -   另一项应该是 $x$ ，因为 $\boldsymbol{z}=\boldsymbol{W} \boldsymbol{x}+\boldsymbol{b}$ 
-   这表明 $\frac{\partial s}{\partial \boldsymbol{W}}=\boldsymbol{\delta}^{T} \boldsymbol{x}^{T}$

![1560364755148](imgs/1560364755148.png)

**Why the Transposes?**
$$
\begin{array}{l}{\frac{\partial s}{\partial \boldsymbol{W}}=\boldsymbol{\delta}^{T} \quad \boldsymbol{x}^{T}} \\ {[n \times m]} {[n \times 1]} {[1 \times m]}\end{array}
$$

-   粗糙的回答是：这样就可以解决尺寸问题了
    -   检查工作的有用技巧
-   课堂讲稿中有完整的解释
    -   每个输入到每个输出——你得到的是外部积

$$
\frac{\partial s}{\partial \boldsymbol{W}}=\boldsymbol{\delta}^{T} \boldsymbol{x}^{T}=\left[\begin{array}{c}{\delta_{1}} \\ {\vdots} \\ {\delta_{n}}\end{array}\right]\left[x_{1}, \ldots, x_{m}\right]=\left[\begin{array}{ccc}{\delta_{1} x_{1}} & {\dots} & {\delta_{1} x_{m}} \\ {\vdots} & {\ddots} & {\vdots} \\ {\delta_{n} x_{1}} & {\dots} & {\delta_{n} x_{m}}\end{array}\right]
$$

**What shape should derivatives be?**

-   $\frac{\partial s}{\partial \boldsymbol{b}}=\boldsymbol{h}^{T} \circ f^{\prime}(\boldsymbol{z})$ 是行向量
    -   但是习惯上说梯度应该是一个列向量因为 $b$ 是一个列向量
-   雅可比矩阵形式(这使得链式法则很容易)和形状约定(这使得SGD很容易实现)之间的分歧
    -   我们希望答案遵循形状约定
    -   但是雅可比矩阵形式对于计算答案很有用

-   两个选择
    -   尽量使用雅可比矩阵形式，最后按照约定进行整形
        -   我们刚刚做的。但最后转置 $\frac{\partial s}{\partial \boldsymbol{b}}$ 使导数成为列向量，得到 $\delta ^ T$
    -   始终遵循惯例
        -   查看维度，找出何时转置 和/或 重新排序项。

反向传播

-   算法高效地计算梯度
-   将我们刚刚手工完成的转换成算法
-   用于深度学习软件框架(TensorFlow, PyTorch, Chainer, etc.)

## Notes 03 Neural Networks, Backpropagation

??? abstract "Keyphrases"

    Neural networks.Forward computation.Backward.propagation.Neuron Units.Max-margin Loss.Gradient checks.Xavier parameter initialization.Learning rates.Adagrad.

## Reference

以下是学习本课程时的可用参考书籍：

[《神经网络与深度学习》](<https://nndl.github.io/>)

以下是整理笔记的过程中参考的博客：

[斯坦福CS224N深度学习自然语言处理2019冬学习笔记目录](<https://zhuanlan.zhihu.com/p/59011576>) (课件核心内容的提炼，并包含作者的见解与建议)

[斯坦福大学 CS224n自然语言处理与深度学习笔记汇总](<https://zhuanlan.zhihu.com/p/31977759>) {>>这是针对note部分的翻译<<}