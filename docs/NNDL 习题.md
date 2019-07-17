# NNDL 习题

## 第 2 章机器学习概述

!!! question "习题2-1"

    分析为什么平方损失函数不适用于分类问题。

**Answer**

-   假设分类问题的类别是 ${1,2,3}$
-   那么对于一个真实类别为 2 的样本 X 而言，模型的分类结果是 1 或 3 ，平方损失函数得到的结果都是一样的
-   这显然不合理，无法让通过这一损失函数训练模型

!!! question "习题2-2"


    在线性回归中，如果我们给每个样本 $\left(\mathbf{x}^{(n)}, y^{(n)}\right)$ 赋予一个权重 $r^{(n)}$ ，经验风险函数为
    
    $$
    \mathcal{R}(\mathbf{w})=\frac{1}{2} \sum_{n=1}^{N} r^{(n)}\left(y^{(n)}-\mathbf{w}^{\mathrm{T}} \mathbf{x}^{(n)}\right)^{2}
    $$
    
    计算其最优参数$\mathbf{W}^{*}$，并分析权重 $r^{(n)}$ 的作用。

**Answer**

-   

!!! question "习题2-3"

    证明在线性回归中，如果样本数量 $N$ 小于特征数量 $d+1$，则 $X X^{\mathrm{T}}$ 的秩最大为 $N$ 。

**Answer**

-   已知定理：设 $A,B$ 分别为 $n\times m,m \times s$ 的矩阵，则 $\mathrm{rank}(\mathrm{AB}) \leqslant \min \{\mathrm{rank}(\mathrm{A}), \mathrm{rank}(\mathrm{B})\}$
-   而 $X \in \mathbb{R}^{(d+1) \times N}$ , $ X^T \in \mathbb{R}^{N \times (d+1)}$ , $rank(X)=rank(X^{T})=min((d+1),N)$ , $N < d+1$ .可知 $rank(X)=N$
-   可知 $rank(XX^T) \leq \{N,N\} = N$

!!! question "习题2-4"

	在线性回归中，验证岭回归的解为**结构风险最小化准则**下的最小二乘法估计，见公式(2.45)。

**Answer**

已知

$$
\mathcal{R}(\mathbf{w})=\frac{1}{2}\left\|\mathbf{y}-X^{\mathrm{T}} \mathbf{w}\right\|^{2}+\frac{1}{2} \lambda\|\mathbf{w}\|^{2}
$$

$$
\mathbf{w}^{*}=\left(X X^{\mathrm{T}}+\lambda I\right)^{-1} X \mathbf{y}
$$

可得

$$
\begin{aligned} 
\frac{\partial \mathcal{R}(\mathbf{w})}{\partial \mathbf{w}} &=\frac{1}{2} \frac{\partial\left\|\mathbf{y}-X^{\mathrm{T}} \mathbf{w}\right\|^{2}+ \lambda\|\mathbf{w}\|^{2}}{\partial \mathbf{w}} \\ 
&=-X\left(\mathbf{y}-X^{\mathrm{T}} \mathbf{w}\right)+\lambda \mathbf{w} 
\end{aligned}
$$

令 $\frac{\partial}{\partial \mathbf{w}} \mathcal{R}(\mathbf{w})=0$ 可得

$$
-XY + XX^{\mathrm{T}}\mathbf{w}+\lambda \mathbf{w}=0\\
(XX^{\mathrm{T}}+\lambda I)\mathbf{w}=XY \\
$$

即

$$
\mathbf{w}^{*}=\left(X X^{\mathrm{T}}+\lambda I\right)^{-1} X \mathbf{y}
$$

!!! question "习题2-5"

    在线性回归中，若假设标签 $y \sim \mathcal{N}\left(\mathbf{w}^{\mathrm{T}} \mathbf{x}, \beta\right)$，并用最大似然估计来优化参数时，验证最优参数为公式(2.51) 的解。

**Answer**

已知

$$
\log p(\mathbf{y} | X ; \mathbf{w}, \sigma)=\sum_{n=1}^{N} \log \mathcal{N}(y^{(n)} | \mathbf{w}^{\mathrm{T}} \mathbf{x}^{(n)}, \sigma^{2} )
$$

令 $\frac{\partial \log p(\mathbf{y} | X ; \mathbf{w}, \sigma)}{\partial \mathbf{w}}=0$ ，即有

$$
\frac{\partial \left(\sum _{n=1}^{N} - \frac{\left(y^{(n)}-\mathbf{w}^{\mathrm{T}} \mathbf{x}^{(n)}\right)^2}{2 \beta}\right)}{\partial \mathbf{w}}=0\\
\frac{\partial \frac{1}{2}\left\|\mathbf{y}-X^{\mathrm{T}} \mathbf{w}\right\|^{2}}{\partial \mathbf{w}} = 0 \\
-X\left(\mathbf{y}-X^{\mathrm{T}} \mathbf{w}\right) = 0
$$

则

$$
\mathbf{w}^{M L}=\left(X X^{\mathrm{T}}\right)^{-1} X \mathbf{y}
$$


!!! question "习题2-6"

    假设有 $N$ 个样本 $x^{(1)}, x^{(2)}, \cdots, x^{(N)}$ 服从正态分布 $\mathcal{N}\left(\mu, \sigma^{2}\right)$ ，其中 $\mu$ 未知
    
    (1) 使用最大似然估计来求解最优参数 $\mu^{M L}$
    
    (2) 若参数  $\mu$ 为随机变量，并服从正态分布 $\mathcal{N}\left(\mu_{0}, \sigma_{0}^{2}\right)$ ，使用最大后验估计来求解最优参数 $\mu^{M A P}$ 。

**Answer**

-   

!!! question "习题2-7"

	在习题2-6中，证明当$N \rightarrow \infty$时，最大后验估计趋向于最大似然估计。

**Answer**

-   

!!! question "习题2-8"

	验证公式(2.60)

**Answer**

-   

!!! question "习题2-9"

	试分析在什么因素会导致模型出现图2.6所示的高偏差和高方差情况。

**Answer**

-   

!!! question "习题2-10"

	验证公式(2.65)

**Answer**

-   

!!! question "习题2-11"

	分别用一元、二元和三元特征的词袋模型表示文本“我打了张三”和“张三打了我”，并分析不同模型的优缺点。

**Answer**

-   

!!! question "习题2-12"

    对于一个三类分类问题，数据集的真实标签和模型的预测标签如下：
    
    ![image-20190717200915371](imgs/image-20190717200915371.png)
    
    分别计算模型的查准率、查全率、F1 值以及它们的宏平均和微平均。

**Answer**

-   

