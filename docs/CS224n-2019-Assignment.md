# CS224n-2019 Assignment

本文档将简要记录作业中的要点

课程笔记参见[此处](https://looperxx.github.io/CS224n-2019-01-Introduction%20and%20Word%20Vectors/)

## Assignment 01

-   逐步完成共现矩阵的搭建，并调用 `sklearn.decomposition` 中的 `TruncatedSVD` 完成传统的基于SVD的降维算法
-   可视化展示，观察并分析其在二维空间下的聚集情况。
-   载入Word2Vec，将其与SVD得到的单词分布情况进行对比，分析两者词向量的不同之处。
-   学习使用`gensim`，使用`Cosine Similarity` 分析单词的相似度，对比单词和其同义词与反义词的`Cosine Distance` ，并尝试找到正确的与错误的类比样例
-   探寻Word2Vec向量中存在的 `Independent Bias` 问题

## Assignment 02

### 1  Written: Understanding word2vec

$$
P(O=o | C=c)=\frac{\exp \left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)}{\sum_{w \in \mathrm{Vocab}} \exp \left(\boldsymbol{u}_{w}^{\top} \boldsymbol{v}_{c}\right)}
$$

$$
J_{\text { naive-softmax }}\left(v_{c}, o, U\right)=-\log P(O=o | C=c)
$$

真实(离散)概率分布 $p$ 与另一个分布 $q$ 的交叉熵损失为 $-\sum_i p_{i} \log \left(q_{i}\right)$

!!! question "Question a"

    Show that the naive-softmax loss given in Equation (2) is the same as the cross-entropy loss between $y$ and $\hat y$; i.e., show that
    
    $$
    -\sum_{w \in Vocab} y_{w} \log \left(\hat{y}_{w}\right)=-\log \left(\hat{y}_{o}\right)
    $$
    
    Your answer should be one line.

**Answer a** : 

因为 $\textbf{y}$ 是独热向量，所以 $-\sum_{w \in Vocab} y_{w} \log (\hat{y}_{w})=-y_o\log (\hat{y}_{o}) -\sum_{w \in Vocab,w \neq o} y_{w} \log (\hat{y}_{w}) = -\log (\hat{y}_{o})$ 

!!! question "Question b"

    Compute the partial derivative of $J_{\text{naive-softmax}}(v_c, o, \textbf{U})$ with respect to $v_c$. Please write your answer in terms of $\textbf{y}, \hat {\textbf{y}}, \textbf{U}$.

**Answer b** : 

$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial v_{c}}} &={-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial v_{c}}+\frac{\partial \left(\log \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial v_{c}}} 
\\ &={-u_{o}+\frac{1}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)}} \frac{\partial \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)}{\partial v_{c}}
\\ &={-u_{o}+\sum_{w} \frac{\exp \left(u_{w}^{T} v_{c}\right)u_w}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)}} 
\\ &={-u_{o}+\sum_{w} p(O=w | C=c)u_{w}}
\\ &={-u_{o}+\sum_{w} \hat{y_w} u_{w}}
\\ &={U(\hat{y}-y)}
\end{array}
$$

!!! question "Question c"

    Compute the partial derivatives of $J_{\text{naive-softmax}}(v_c, o, \textbf{U})$ with respect to each of the ‘outside' word vectors, $u_w$'s. There will be two cases: when $w = o$, the true ‘outside' word vector, and $w \neq o$, for all other words. Please write you answer in terms of $\textbf{y}, \hat {\textbf{y}}, \textbf{U}$.

**Answer c** : 
$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial u_{w}}}  
&={-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial u_{w}}+\frac{\partial \left(\log \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial u_{w}}} 
\end{array}
$$

When $w \neq o$ :

$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial u_{w}}}  
&= 0 + p(O=w | C=c) v_{c}
\\ &=\hat{y}_{w} v_{c}
\end{array}
$$

When $w = o$ :

$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial u_{w}}}  
&= -v_c + p(O=o | C=c) v_{c}
\\ &=\hat{y}_{w} v_{c} - v_c
\\ &=(\hat{y}_{w} - 1)v_c
\end{array}
$$

Then : 

$$
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial U}} = v_c(\hat y - y)^T
$$

!!! question "Question d"

    The sigmoid function is given by the follow Equation :
    
    $$
    \sigma(x)=\frac{1}{1+e^{-x}}=\frac{e^{x}}{e^{x}+1}
    $$
    
    Please compute the derivative of $\sigma (x)$ with respect to $x$, where $x$ is a vector.

**Answer d** : 

$$
\begin{array}{l}
\frac{\partial \sigma(x_i)}{\partial x_i}
&=\frac{1}{(1+\exp (-x_i))^{2}} \exp (-x_i)=\sigma(x_i)(1-\sigma(x_i)) \\
\frac{\partial \sigma(x)}{\partial x}
&= \left[\frac{\partial \sigma\left(x_{j}\right)}{\partial x_{i}}\right]_{d \times d}
\\ &=\left[\begin{array}{cccc}{\sigma^{\prime}\left(x_{1}\right)} & {0} & {\cdots} & {0} \\ {0} & {\sigma^{\prime}\left(x_{2}\right)} & {\cdots} & {0} \\ {\vdots} & {\vdots} & {\vdots} & {\vdots} \\ {0} & {0} & {\cdots} & {\sigma^{\prime}\left(x_{d}\right)}\end{array}\right]
\\ &=\text{diag}(\sigma^\prime(x))
\end{array}
$$

!!! question "Question e"

    Now we shall consider the Negative Sampling loss, which is an alternative to the Naive
    Softmax loss. Assume that $K$ negative samples (words) are drawn from the vocabulary. For simplicity
    of notation we shall refer to them as $w_{1}, w_{2}, \dots, w_{K}$ and their outside vectors as $u_{1}, \dots, u_{K}$. Note that
    $o \notin\left\{w_{1}, \dots, w_{K}\right\}$. For a center word $c$ and an outside word $o$, the negative sampling loss function is
    given by:
    
    $$
    J_{\text { neg-sample }}\left(v_{c}, o, U\right)=-\log \left(\sigma\left(u_{o}^{\top} v_{c}\right)\right)-\sum_{k=1}^{K} \log \left(\sigma\left(-u_{k}^{\top} v_{c}\right)\right)
    $$
    
    for a sample $w_{1}, w_{2}, \dots, w_{K}$, where $\sigma(\cdot)$ is the sigmoid function.
    
    Please repeat parts b and c, computing the partial derivatives of $J_{\text { neg-sample }}$ respect to $v_c$, with
    respect to $u_o$, and with respect to a negative sample $u_k$. Please write your answers in terms of the
    vectors $u_o, v_c,$ and $u_k$, where $k \in[1, K]$. After you've done this, describe with one sentence why this
    loss function is much more efficient to compute than the naive-softmax loss. Note, you should be able
    to use your solution to part (d) to help compute the necessary gradients here.

**Answer e** : 

For $v_c$ :

$$
\begin{array}{l}
\frac{\partial J_{\text {neg-sample}}}{\partial v_c}
&=(\sigma(u_o^T v_c) - 1) u_o	+ \sum_{k=1}^{K}\left(1-\sigma\left(-u_{k}^{T} v_{c}\right)\right) u_{k} 
\\ &= (\sigma(u_o^T v_c) - 1) u_o+ \sum_{k=1}^{K}\sigma\left(u_{k}^{T} v_{c}\right) u_{k}
\end{array}
$$

For $u_o$, Remeber : $o \notin\left\{w_{1}, \dots, w_{K}\right\}$ :cry: :

$$
\frac{\partial J_{\text {neg-sample}}}{\partial u_o}=(\sigma(u_o^T v_c) - 1)v_c
$$

For $u_k$ :

$$
\frac{\partial J}{\partial \boldsymbol{u}_{k}}=-\left(\sigma\left(-\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)-1\right) \boldsymbol{v}_{c} = \sigma\left(\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right)\boldsymbol{v}_{c}, \quad for\ k=1,2, \ldots, K
$$

Why this
loss function is much more efficient to compute than the naive-softmax loss?

For naive softmax loss function:

$$
\begin{array}{l}
{\frac{\partial J\left(v_{c}, o, U\right)}{\partial v_{c}}}  
&={U(\hat{y}-y)}
\\ {\frac{\partial J\left(v_{c}, o, U\right)}{\partial U}} 
&= v_c(\hat y - y)^T
\end{array}
$$


For negative sampling loss function:

$$
\begin{aligned} \frac{\partial J}{\partial \boldsymbol{v}_{c}} &=\left(\sigma\left(\boldsymbol{u}_{o}^{\top} v_{c}\right)-1\right) \boldsymbol{u}_{o} + \sum_{k=1}^{K}\sigma\left(\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right) \boldsymbol{u}_{k} 
=\sigma\left(-\boldsymbol{u}_{o}^{\top} v_{c}\right) \boldsymbol{u}_{o} + \sum_{k=1}^{K}\sigma\left(\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right) \boldsymbol{u}_{k}
\\ \frac{\partial J}{\partial \boldsymbol{u}_{o}} &=\left(\sigma\left(\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)-1\right) \boldsymbol{v}_{c} 
= \sigma\left(-\boldsymbol{u}_{o}^{\top} \boldsymbol{v}_{c}\right)\boldsymbol{v}_{c} 
\\ \frac{\partial J}{\partial \boldsymbol{u}_{k}} &=\sigma\left(\boldsymbol{u}_{k}^{\top} \boldsymbol{v}_{c}\right) \boldsymbol{v}_{c}, \quad \text { for all } k=1,2, \ldots, K \end{aligned}
$$

从求得的偏导数中我们可以看出，原始的softmax函数每次对 $v_c$ 进行反向传播时，需要与 output vector matrix 进行大量且复杂的矩阵运算，而负采样中的计算复杂度则不再与词表大小 $V$ 有关，而是与采样数量 $K$ 有关。

!!! question "Question f"

    Suppose the center word is $c = w_t$ and the context window is $\left[w_{t-m}, \ldots, w_{t-1}, w_{t}, w_{t+1}, \dots,w_{t+m} \right]$, where $m$ is the context window size. Recall that for the skip-gram version of **word2vec**, the
    total loss for the context window is
    
    $$
    J_{\text { skip-gram }}\left(v_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right)=\sum_{-m \leq j \leq m \atop j \neq 0} \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)
    $$
    
    Here, $J\left(v_{c}, w_{t+j}, \boldsymbol{U}\right)$ represents an arbitrary loss term for the center word $c = w_t$ and outside word
    $w_t+j$ . $J\left(v_{c}, w_{t+j}, \boldsymbol{U}\right)$ could be $J_{\text {naive-softmax}}\left(v_{c}, w_{t+j}, \boldsymbol{U}\right)$ or $J_{\text {neg-sample}}\left(v_{c}, w_{t+j}, \boldsymbol{U}\right)$, depending on your
    implementation.
    
    Write down three partial derivatives:
    
    $$
    \partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right) / \partial \boldsymbol{U} \\ \partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right) / \partial v_{c}
    \\ \partial \boldsymbol{J}_{\text { skip-gram }}\left(\boldsymbol{v}_{c}, w_{t-m}, \ldots w_{t+m}, \boldsymbol{U}\right) / \partial v_{w} \text { when } w \neq c
    $$
    
    Write your answers in terms of $\partial \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right) / \partial \boldsymbol{U}$ and $\partial \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right) / \partial \boldsymbol{v_c}$. This is very simple -
    each solution should be one line.
    
    ***Once you're done***: Given that you computed the derivatives of $\partial \boldsymbol{J}\left(\boldsymbol{v}_{c}, w_{t+j}, \boldsymbol{U}\right)$ with respect to all the
    model parameters $U$ and $V$ in parts a to c, you have now computed the derivatives of the full loss
    function $J_{skip-gram}$ with respect to all parameters. You're ready to implement ***word2vec*** !

**Answer f** : 

$$
\begin{array}{l} 
\frac{\partial J_{s g}}{\partial U} &= \sum_{-m \leq j \leq m, j \neq 0} \frac{\partial J\left(v_{c}, w_{t+j}, U\right)}{\partial U} 
\\ \frac{\partial J_{s g}}{\partial v_{c}}&= \sum_{-m \leq j \leq m, j \neq 0} \frac{\partial J\left(v_{c}, w_{t+j}, U\right)}{\partial v_{c}} 
\\ \frac{\partial J_{s g}}{\partial v_{w}}&=0(\text { when } w \neq c) \end{array}
$$


## Reference

-   [从SVD到PCA——奇妙的数学游戏](<https://my.oschina.net/findbill/blog/535044>)