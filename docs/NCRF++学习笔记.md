# NCRF++学习笔记

## 文档学习

### Readme.md

#### Introduction

序列标记模型在许多NLP任务中非常流行，例如命名实体识别（NER），词性（POS）标记和分词。最先进的序列标记模型主要利用具有输入字特征的CRF结构。 LSTM（或双向LSTM）是序列标记任务中流行的基于深度学习的特征提取器。并且由于更快的计算，也可以使用CNN。此外，单词中的特征对于表示单词也是有用的，可以通过字符LSTM或字符CNN结构或人类定义的神经特征来捕获单词。

NCRF ++是一个基于PyTorch的框架，可灵活选择输入特征和输出结构。使用NCRF ++的神经序列标记模型的设计可通过配置文件完全配置，该配置文件不需要任何代码工作。 NCRF ++可以被视为 [CRF++](http://taku910.github.io/crfpp/) 的神经网络版本，这是一个著名的统计CRF框架。

该框架已被 [ACL 2018](https://arxiv.org/abs/1806.05626) 接受为 demo 论文。使用NCRF ++的详细实验报告和分析已被 [COLING 2018](https://arxiv.org/abs/1806.04470) 接受为最佳论文。

NCRF ++支持三个级别的不同结构组合：character sequence layer; word sequence layer and inference layer. 

-   字符序列表示：字符LSTM，字符GRU，字符CNN和手工制作的单词特征。
-   单词序列表示：单词LSTM，单词GRU，单词CNN。
-   推理层：Softmax，CRF。

#### Requirement

```
Python: 2 or 3  
PyTorch: 1.0 
```

[PyTorch 0.3 compatible version is here.](https://github.com/jiesutd/NCRFpp/tree/PyTorch0.3)

#### Advantages

-   完全可配置：可以使用配置文件设置所有神经模型结构。
-   最先进的系统性能：与最先进的模型相比，基于NCRF ++的模型可以提供相当或更好的结果。
-   灵活的特征：用户可以定义自己的特征和预训练的特征嵌入。
-   快速运行速度：NCRF ++利用完全批量操作，在GPU的帮助下使系统高效（> 1000sent / s用于训练，> 2000sents / s用于解码）

-   N best output: NCRF++ 支持 `nbest` decoding (with their probabilities).

#### Usage

NCRF ++支持通过配置文件设计神经网络结构。该程序可以运行两种状态 **训练和解码**。 （示例配置和数据已包含在此存储库中）

In ***training*** status: `python main.py --config demo.train.config`

In ***decoding*** status: `python main.py --config demo.decode.config`

配置文件控制网络结构，I / O，训练设置和超参数。

***Detail configurations and explanations are listed [here](https://github.com/jiesutd/NCRFpp/blob/master/readme/Configuration.md).***

NCRF ++设计为三层（如下所示）：字符序列层;单词序列层和推理层。通过使用配置文件，可以轻松复制大多数最先进的模型而无需编码。另一方面，用户可以通过设计自己的模块来扩展每一层（例如，他们可能想要设计除CNN / LSTM / GRU之外的其他神经结构）。我们的层设计使模块扩展方便，模块扩展的指令可以在 [这里](https://github.com/jiesutd/NCRFpp/blob/master/readme/Extension.md) 找到。

![alt text](imgs/architecture.png)

-   图中的红色圆圈指word embedding，黄色指 charCNN/RNN 生成的 word embedding，灰色是自定义的 handcrafted 特征。

#### Data Format

-   You can refer the data format in [sample_data](https://github.com/jiesutd/NCRFpp/blob/master/sample_data).
-   NCRF++ supports both BIO and BIOES(BMES) tag scheme.
-   Notice that IOB format (***different*** from BIO) is currently not supported, because this tag scheme is old and works worse than other schemes [Reimers and Gurevych, 2017](https://arxiv.org/pdf/1707.06799.pdf).
-   The difference among these three tag schemes is explained in this [paper](https://arxiv.org/pdf/1707.06799.pdf).
-   I have written a [script](https://github.com/jiesutd/NCRFpp/blob/master/utils/tagSchemeConverter.py) which converts the tag scheme among IOB/BIO/BIOES. Welcome to have a try.

#### Performance

Results on CONLL 2003 English NER task are better or comparable with SOTA results with the same structures.

CharLSTM+WordLSTM+CRF: 91.20 vs 90.94 of [Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf);

CharCNN+WordLSTM+CRF: 91.35 vs 91.21 of [Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf).

By default, `LSTM` is bidirectional LSTM.

| ID   | Model        | Nochar | CharLSTM  | CharCNN   |
| ---- | ------------ | ------ | --------- | --------- |
| 1    | WordLSTM     | 88.57  | 90.84     | 90.73     |
| 2    | WordLSTM+CRF | 89.45  | **91.20** | **91.35** |
| 3    | WordCNN      | 88.56  | 90.46     | 90.30     |
| 4    | WordCNN+CRF  | 88.90  | 90.70     | 90.43     |

We have compared twelve neural sequence labeling models (`{charLSTM, charCNN, None} x {wordLSTM, wordCNN} x {softmax, CRF}`) on three benchmarks (POS, Chunking, NER) under statistical experiments, detail results and comparisons can be found in our COLING 2018 paper [Design Challenges and Misconceptions in Neural Sequence Labeling](https://arxiv.org/abs/1806.04470).

#### Add Handcrafted Features

NCRF++ 集成了最先进的几个神经特征序列特征提取器：CNN ([Ma .etc, ACL16](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf)), LSTM ([Lample .etc, NAACL16](http://www.aclweb.org/anthology/N/N16/N16-1030.pdf)) and GRU ([Yang .etc, ICLR17](https://arxiv.org/pdf/1703.06345.pdf)). 此外，手工制作的特征已被证明在序列标记任务中很重要。 NCRF ++允许用户设计自己的特征，如大写，POS标签或任何其他特征（上图中的灰色圆圈）。用户可以通过配置文件配置自定义特征（特征嵌入大小，预训练特征嵌入等）。样本输入数据格式在 [train.cappos.bmes](https://github.com/jiesutd/NCRFpp/blob/master/sample_data/train.cappos.bmes) 中，其中包括两个人为定义的特征[POS]和[Cap]（[POS]和[Cap]是两个示例，您可以为您的特征提供所需的任何名称，只需按照格式[xx]并在配置文件中配置相同名称的特征。）用户可以配置配置文件中的每个特征，通过使用

```
feature=[POS] emb_size=20 emb_dir=%your_pretrained_POS_embedding
feature=[Cap] emb_size=20 emb_dir=%your_pretrained_Cap_embedding
```

没有预训练嵌入的特征将被随机初始化。

#### Speed

NCRF ++使用完全批量计算实现，使其在模型训练和解码方面都非常有效。在GPU（Nvidia GTX 1080）和大批量大小的帮助下，使用NCRF ++构建的LSTMCRF模型分别在训练和解码状态下可达到1000个sents / s和2000个sents / s。

![alt text](imgs/speed.png)

#### N best Decoding

传统的CRF结构仅解码具有最大可能性的一个标签序列（即1个最佳输出）。而NCRF ++可以提供大量选择，它可以解码具有 top n 概率的n个标签序列（即n-best output）。nbest 解码已得到几个流行的 **统计** CRF框架的支持。然而据我们所知，NCRF ++是神经CRF模型中唯一支持nbest解码的工具包。

在我们的实现中，当 nbest = 10 时，在NCRF ++中构建的CharCNN + WordLSTM + CRF模型在CoNLL 2003 NER任务上可以给出97.47％的oracle F1值（当nbest = 1时F1 = 91.35％）。

![alt text](imgs/nbest.png)

#### Reproduce Paper Results and Hyperparameter Tuning

To reproduce the results in our COLING 2018 paper, you only need to set the `iteration=1` as `iteration=100` in configuration file `demo.train.config` and configure your file directory in this configuration file. The default configuration file describes the `Char CNN + Word LSTM + CRF` model, you can build your own model by modifing the configuration accordingly. The parameters in this demo configuration file are the same in our paper. (Notice the `Word CNN` related models need slightly different parameters, details can be found in our COLING paper.)

If you want to use this framework in new tasks or datasets, here are some tuning [tips](https://github.com/jiesutd/NCRFpp/blob/master/readme/hyperparameter_tuning.md) by @Victor0118.

### Configuration.md

本文档与 `demo.train.config` 相对应

#### I/O

| 指令                   | 解释                                                         | 备注                                                         |
| ---------------------- | ------------------------------------------------------------ | :----------------------------------------------------------- |
| train_dir=xx           | string (necessary in training). Set training file directory. | 训练集位置                                                   |
| dev_dir=xx             | string (necessary in training). Set dev file directory.      | 开发集位置                                                   |
| test_dir=xx            | string . Set test file directory.                            | 测试集位置                                                   |
| model_dir=xx           | string (optional). Set saved model file directory.           | 输出 模型存储位置 `model_dir=lstmcrf` 则 model 保存为`lstm.0.model` ，相关参数词典为`lstm.dset` |
| word_emb_dir=xx        | string (optional). Set pretrained word embedding file directory. | 预训练词嵌入的位置                                           |
| raw_dir=xx             | string (optional). Set input raw file directory.             | 原始数据文件的位置                                           |
| decode_dir=xx          | string (necessary in decoding). Set decoded file directory.  | 输出 解码结果的位置                                          |
| dset_dir=xx            | string (necessary). Set saved model file directory.          | 模型参数词典保存的位置                                       |
| load_model_dir=xx      | string (necessary in decoding). Set loaded model file directory. (when decoding) | 模型的位置                                                   |
| char_emb_dir=xx        | string (optional). Set pretrained character embedding file directory. | 预训练的字符嵌入文件的位置                                   |
| norm_word_emb=False    | boolen. If normalize the pretrained word embedding.          | 是否对预训练的词嵌入标准化                                   |
| norm_char_emb=False    | boolen. If normalize the pretrained character embedding.     | 是否对预训练的字符嵌入标准化                                 |
| number_normalized=True | boolen. If normalize the digit into `0` for input files.     | 是否将数字标准化为输入文件的“0”                              |
| seg=True               | boolen. If task is segmentation like, tasks with token accuracy evaluation (e.g. POS, CCG) is False; tasks with F-value evaluation(e.g. Word Segmentation, NER, Chunking) is True . | 如果任务是 segmentation，令牌准确度评估的任务（例如POS，CCG）为 `False` ; F值评估的任务（例如，Word Segmentation, NER, Chunking）为 `True` 。 |
| word_emb_dim=50        | int. Word embedding dimension, if model use pretrained word embedding, word_emb_dim will be reset as the same dimension as pretrained embedidng. | 词嵌入维度 (如果模型使用预训练单词嵌入，`word_emb_dim`将被重置为与预训练`embedidng`相同的维度) |
| char_emb_dim=30        | int. Character embedding dimension, if model use pretrained character embedding, char_emb_dim will be reset as the same dimension as pretrained embedidng. | 字符嵌入维度 (如果模型使用预训练单词嵌入，`char_emb_dim`将被重置为与预训练`embedidng`相同的维度) |

#### NetworkConfiguration

| 指令                                                | 解释                                                         | 备注                                                         |
| --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| use_crf=True                                        | boolen (necessary in training). Flag of if using CRF layer. If it is set as False, then Softmax is used in inference layer. | `True` : CRF; `False` : Softmax                              |
| use_char=True                                       | boolen (necessary in training). Flag of if using character sequence layer. | `True` : 使用字符序列层                                      |
| word_seq_feature=XX                                 | boolen (necessary in training): CNN/LSTM/GRU. Neural structure selection for word sequence. | 词序列特征提取模型：CNN/LSTM/GRU                             |
| char_seq_feature=CNN                                | boolen (necessary in training): CNN/LSTM/GRU. Neural structure selection for character sequence, it only be used when use_char=True. | 字符特征提取模型：CNN/LSTM/GRU                               |
| feature=[POS] emb_size=20 emb_dir=xx emb_norm=false | feature configuration. It includes the feature prefix [POS], pretrained feature embedding file and the embedding size. | 特征配置：`feature=[特征名称] emb_size=20 emb_dir=xx emb_norm=false` |
| feature=[Cap] emb_size=20 emb_dir=xx emb_norm=false | feature configuration. Another feature [Cap].                | 同上                                                         |
| nbest=1                                             | int (necessary in decoding). Set the nbest size during decoding. | 设置 nbest 的 size                                           |

#### TrainingSetting

| 指令                 | 解释                                                         | 备注                           |
| -------------------- | ------------------------------------------------------------ | ------------------------------ |
| status=train         | string: train or decode. Set the program running in training or decoding mode. | train / decode 模式            |
| optimizer=SGD        | string: SGD/Adagrad/AdaDelta/RMSprop/Adam. optimizer selection. | 优化器选择                     |
| iteration=1          | int. Set the iteration number of training.                   | 迭代次数                       |
| batch_size=10        | int. Set the batch size of training or decoding.             | 批量大小                       |
| ave_batch_loss=False | boolen. Set average the batched loss during training.        | 是否设置 loss 为批量损失的均值 |

#### Hyperparameters

| 指令                | 解释                                                         | 备注                                                   |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| cnn_layer=4         | int. CNN layer number for word sequence layer.               | 词序列层的CNN 层的深度，从而更好地学习长距离的依赖关系 |
| char_hidden_dim=50  | int. Character hidden vector dimension for character sequence layer. | 字符序列层的字符隐层向量的维度                         |
| hidden_dim=200      | int. Word hidden vector dimension for word sequence layer.   | 词序列层的词隐层向量的维度                             |
| dropout=0.5         | float. Dropout probability.                                  | Dropout 概率                                           |
| lstm_layer=1        | int. LSTM layer number for word sequence layer.              | LSTM 层数                                              |
| bilstm=True         | boolen. If use bidirection lstm for word seuquence layer.    | 是否在词序列层中使用双向 LSTM                          |
| learning_rate=0.015 | float. Learning rate.                                        | 学习率                                                 |
| lr_decay=0.05       | float. Learning rate decay rate, only works when optimizer=SGD. | SGD 中的学习率衰减概率                                 |
| momentum=0          | float. Momentum                                              | 动量值                                                 |
| l2=1e-8             | float. L2-regulization.                                      | L2 正则化参数                                          |
| gpu=True            | boolen. If use GPU, generally it depends on the hardward environment. | 是否使用 GPU                                           |
| clip=               | float. Clip the gradient which is larger than the setted number. | 梯度裁剪的设定数值                                     |

### Extension.md

#### Module Extension.

If you want to extend character sequence layer: please refer to the file [charlstm.py](model/charlstm.py).

If you want to extend word sequence layer: please refer to the file [wordsequence.py](model/wordsequence.py).

More details will be updated soon.

### Hyperparameter_tuning.md

#### Hyperparamter tuning on CoNLL 2003 English NER task

1.  如果您使用大批量（例如batch_size> 100），您最好设置 `avg_batch_loss = True` 以获得稳定的训练过程。对于小批量，`avg_batch_loss = True` 将更快收敛，有时会提供更好的性能（例如CoNLL 2003 NER）。
2.  如果使用100-d预训练单词向量 [此处](https://nlp.stanford.edu/projects/glove/) 而不是50-d预训练单词向量，则可以在CoNLL 2003英语数据集上获得更好的性能。
3.  如果要编写脚本来调整超参数，可以使用 `main_parse.py` 在命令行参数中设置超参数
4.  模型性能对 `lr` 敏感，需要在不同结构下仔细调整：
    -   Word level LSTM models (e.g. char LSTM + word LSTM + CRF) would prefer a `lr` around 0.015.
    -   Word level CNN models (e.g. char LSTM + word CNN + CRF) would prefer a `lr` around 0.005 and with more iterations.
    -   You can refer the COLING paper "[Design Challenges and Misconceptions in Neural Sequence Labeling](https://arxiv.org/pdf/1806.04470.pdf)" for more hyperparameter settings.

## 源码学习

### 流程梳理

In ***training*** status: `python main.py --config demo.train.config`

In ***decoding*** status: `python main.py --config demo.decode.config`

### 架构梳理