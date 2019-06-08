# Concepts

## ML

[AdaBoost](<https://www.bilibili.com/video/av28102016/>) {>>真滴是非常非常形象得讲解 :thumbsup:<<}​

### AutoEncoder & Variational autoencoder

[AutoEncoder详解](<https://www.zhihu.com/question/41490383/answer/103006793>)

[VAE详解](<https://zhuanlan.zhihu.com/p/34998569> )

>   VAE 是为每个样本构造专属的正态分布，然后采样来重构
>
>   本质上就是在我们常规的自编码器的基础上，对 encoder 的结果（在VAE中对应着计算均值的网络）加上了“高斯噪声”，使得结果 decoder 能够对噪声有鲁棒性；而那个额外的 KL loss（目的是让均值为 0，方差为 1），事实上就是相当于对 encoder 的一个正则项，希望 encoder 出来的东西均有零均值。另外一个 encoder（对应着计算方差的网络）是用来**动态调节噪声的强度**的。<https://zhuanlan.zhihu.com/p/32486725> <https://zhuanlan.zhihu.com/p/55557709> <https://www.cnblogs.com/fxjwind/p/9099931.html>

## DL

[L1与L2正则化](<https://www.cnblogs.com/weizc/p/5778678.html>)

>   L1 力求稀疏 L2 防止过拟合