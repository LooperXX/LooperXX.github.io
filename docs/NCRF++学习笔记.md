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

### 架构梳理

![alt text](imgs/architecture.png)

框架首先将所有的数据处理的部分，都放在了模型之外，模型的输入均为处理后的 word / char 的 index

模型分为三层：

-   char sequence layer : 支持预训练以及随机初始化两种方式，获得 char-level embedding ，再得到 word embedding。方法包括 charRNN, charCNN，并且可以同时使用 (concatenate embedding_dim 即可)
-   word sequence layer : 支持预训练以及随机初始化两种方式，获得 word-level embedding，而后与 char sequence layer 得到的 word embedding 以及 handcrafted feature (如 POS ) 的embedding（同样支持两种方式）进行 concatenate，最后得到 hybrid 的 word embedding，再经过 多层 CNN / LSTM / GRU ，得到最终的 hidden_state
-   inference layer : 支持 CRF 与 Softmax 两种方式

### 流程梳理

框架首先读取用户定义好的 config 文件，其中包括了 I/O, Network, Training, Hyperparameters 四部分的参数设置。用户可以通过修改 config 文件，或是以 `--parameter value` 的形式在运行时修改。运行指令如下：

-   `python main.py --config train.config` 

**training status**

首先读取初始化运行参数，读取配置覆盖初始参数，而后对数据进行初步处理并读取/初始化所需的 embedding ，包括

-   建立 feature 的字母表 **alphabet** 和 训练集 / 开发集 / 测试集 的 word / char / label / feature_list 的字母表
    -   字母表主要是包括存储 instance 的 list 以及 $\{ \mathbf{key:instance}, \mathbf{value:index}\}$ 的 dict
-   建立 **instance_text** 和 **instance_Ids** 两个 **list** ，其每一项为 `[words, features, chars, labels]` 和 `[word_Ids, feature_Ids, char_Ids, label_Ids]`  （以 NER 任务为例，特征使用的是 POS），分别是 sequence 中的所有单词构成的 `[‘good’]` ，所有特征构成的 `[‘JJ’]` ，所有字符构成的 `[[‘g’, ‘o’, ’o’, ‘d’]]` ，所有 label 构成的 `['O']` ，以及上述内容在各自的字母表中对应的 index 构成的 `xxx_Ids`
-   对建立好的 word / char / label / feature_list 的字母表，遍历字母表中的所有 instance ，分别建立 Embedding 矩阵。读取 word / char / feature 的 pretrain embedding，如果未指定 pretrain embedding 的文件位置，那么均随机初始化，否则会分为以下三种情况分别处理
    -   perfect match : 如果在 pretrain embedding 中找到了字母表里的 instance ，则将其 embedding 对应赋值，否则进行下一步
    -   case_match : 如果在 pretrain embedding 中找到了字母表里的 instance 的 lower 版本的 embedding，则同样对应赋值，否则进行下一步
    -   not_match : 如果在 pretrain embedding 中没有找到字母表里的 instance 的 raw / lower 版本，则随机初始化

接着处理模型各层的输入数据（ batch_size 自定义），包括 `batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask` 

-   将 **instance_Ids** 提取出 batch 个 item，每个 item 都是 `[words, features, chars, labels]` 构成的嵌套 list 
-   将 batch 数据中四类数据分别抽取建立为 list ，包括 `words, features, chars, labels` ，而后建立起对应的 zeros Tensor 以及 mask Tensor，并统计最长句子长度等信息
-   分别在 zeros Tensor 中填充 `words, labels, mask, features` 的 Tensor

Pytorch 中提供了 `torch.nn.utils.rnn.PackedSequence` 的相关 API，输入时需要将 sequence 按照真实长度降序排列，并输入 sequence 以及其对应的真实长度，于是：

-   ```python
    # 统计 batch 中每个 sequence 的长度
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    # 降序排序后获得其排序结果以及该结果中的对应每一项在原序列中的 index
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    ```

举个例子：

-   原序列 word_seq_lengths = [9, 7, 11]，那么排序后的返回值为
    -   word_seq_lengths = [11, 9, 7] ，word_perm_idx = [2, 0, 1]

那么我们就可以将 length 对应的 size 为 $\text{batch_size} \times \text{max_seq_len}$ 的 word_seq_tensor 以及 label_seq_tensor 等进行重排列，即
-   `word_seq_tensor = word_seq_tensor[word_perm_idx]` 那么现在的 word / label / length 就是一一对应的了，feature, mask 也是同理操作

如上操作后，sequence 中的 word-level 的 padding 以及排序工作就完成了，随后进行 char-level 的 padding 以及排序工作

首先将原本的 chars 列表进行 padding ：

-   chars list 中的chars[3] 如下图所示

![image-20190719141036595](imgs/image-20190719141036595.png)

-   可以看到 chars[3] 中的为一个 sequence ，其中每一项均为 sequence 中的某一个单词，如 chars[3] 的第 8 项即为一个长度为 2 的单词，由字符 15 和 40 组成
-   先对每个 sequence 进行 padding，保证 sequence 的长度均为 max_seq_len
-   而后对本 batch 内的所有 sequence 中的所有 word 统计得到最长单词长度 max_word_len
-   初始化 size 为 `batch_size, max_seq_len, max_word_len` 的 zeros Tensor `char_seq_tensor` ，统计当前 chars 中所有单词的长度获得 size 为 `batch_size, max_seq_len` 的 char_seq_lengths
-   将每个 sequence 的每个 word 的每个 char_Id 填入 `char_seq_tensor` 中这样就完成了  char-level 的 padding 工作

接着我们我们对数据做进一步的处理

-   首先通过 word_perm_idx 将序列的顺序调整的与 word 和 label 的顺序一致，然后是对 char_seq_tensor, char_seq_lengths 进行压实操作，将其 size 调整为 `(batch_size * max_seq_len, max_word_len)` 和 `(batch_size * max_seq_len,)` 
-   接着进行同样的排序与调序操作，将 word 的顺序调整为按照 word_len 降序排列
    -   这是我们应该注意到，单词的顺序在 char sequence layer 中被打乱了，我们在经过charCNN / charRNN 的处理后，得到的输出需要输入到 word sequence layer 中，顺序是不一致的，需要在输出时将顺序调整回去

-   回到之前的例子：
    -   原序列 word_seq_lengths = [9, 7, 11]，那么排序后的返回值为 word_seq_lengths = [11, 9, 7] ，word_perm_idx = [2, 0, 1]
    -   我们对 word_perm_idx 进行升序排列，得到返回值为 [0, 1, 2] 和 [1, 2, 0]，这时候令 `word_seq_lengths = word_seq_lengths[1, 2, 0]` ，我们就会发现 word_seq_lengths 变回了原序列
    -   当 char sequence layer 将其获得的每个 word 的 embedding 输出至 word sequence layer 时，需要使用之前保存的 char_seq_recover 将 word 的顺序还原

至此，各层所需的输入数据都准备完毕。

![image-20190719145901069](imgs/image-20190719145901069.png)

运行时，模型首先需要完成对 char sequence layer 的运算，获得 char-level 的 word embedding，然后与 word embedding 和 feature embedding 进行 concatenate，得到最终的 word embedding，再通过多层的 LSTM / GRU / CNN 得到最终的 size 为 `batch_size, seq_len, classes` 的特征向量

最后的 inference layer 可以有两种选择：CRF / Softmax ，这里我们先讨论 Softmax 的方式。对特征向量的最后一维做 log_softmax ，而后计算其 NLLLoss 并将其 最大值对应的 class 作为分类结果。

现在来看 **CRF** ，CRF 考虑的不是每次转移时的最优概率，考虑的整体序列的可能性。

Previous_to $\to$ current_from

>   CRF 的核心是它不像LSTM等模型，能够考虑长远的上下文信息，**它更多考虑的是整个句子的局部特征的线性加权组合（通过特征模版去扫描整个句子）**。关键的一点是，CRF的模型为p(y | x, w)，注意这里y和x都是序列，它有点像list wise，优化的是一个序列y = (y1, y2, …, yn)，而不是某个时刻的yt，即找到一个概率最高的序列y = (y1, y2, …, yn)使得p(y1, y2, …, yn| x, w)最高，它计算的是一种联合概率，优化的是整个序列（最终目标），而不是将每个时刻的最优拼接起来，在这一点上CRF要优于LSTM。

转移矩阵的 $[i, j]$ 代表的是由状态 $i$ 转移到状态 $j$ 的可能性

首先我们将获得的 size 为 `batch_size, seq_len, tag_size` 的特征向量调整为`seq_len, batch_size, tag_size` ，方便随后依时间步访问序列，而后再调整为`seq_len * batch_size, 1, tag_size`，再扩展成为 `seq_len * batch_size, tag_size, tag_size` 与转移矩阵相加得到 scores 。可以理解成是将通过 model 获得到的每个时间步的所有的 tag 的可能性都加到转移矩阵之上，即当我们按时间步遍历每个时间步上的 size 为`batch_size, tag_size, tag_size` 的 cur_values 矩阵时，这里的矩阵是由原始的转移矩阵 + 之前 model 得到的每一时间步上的 word 的每种 tag 的可能性。这里我们将特征向量 feature 从一个行向量(不看 batch_size )扩展为一个矩阵，其实就是不管 start 状态是什么，转移到 j 状态的可能性都会加上 扩展前的 feature[j]。

而在每次随时间步的迭代中，我们都会将前一时间步传来的 size 为 `batch_size, tag_size, 1` 的 partition 数组，扩展为 `batch_size, tag_size, tag_size` 并加上。

-   第一次迭代中，partition 是由 `inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1` 得到的，其含义是 start_tag 之后的下一个 tag 的可能性，也就是当前时间步对应的 word 的 tag 的可能性。将这一列向量扩展后并与 cur_values 相加后，相当于每一行都加上同样的值，也就是从状态 i 到其他任何状态都加上了 inivalues[START_TAG, i] (不看 batch_size )。这是因为这一数值的含义是从 start_tag 到状态 i 的可能性，那么 cur_values 需要将由状态 i 出发的所有状态 j 的可能性都增加这一数值，即cur_values[i] = inivalues[START_TAG, i].view(1, tag_size) + cur_values[i]。经过这样的 tag_size 次运算，我们就可以得到一个新的、考虑到前一状态转移矩阵的、新的状态转移矩阵。
-   接下来我们需要对这一矩阵进行处理，得到新的 partition 传递给下一次的迭代。我们先计算矩阵每一列的最大值，构成一个行向量 max_value ，max_value[j] 含义是下一状态为 j 的最大转移可能性， 将其拓展为和输入的 partition 一样的 size 后用 partition - max_value，矩阵的所有值都是负数，逐元素作用 exp 函数将其按列 sum ，逐元素作用 log 函数，最终得到的新的 partition 是一个行向量(不看 batch_size )，partition[j] 代表的是由转移到状态 j 的可能性之和。
-   遍历完序列后，得到 `final_partition = cur_partition[:, STOP_TAG] ` ，即各个状态转移到 stop_tag 的可能性，求得其 sum 并返回 sum 与 scores
-   需要注意的是，上述过程未提及 mask 步骤，实际操作中需要使用 mask 操作完成对 partition 的更新

而今我们已经获得了 `forward_score, scores` ，接下来继续计算 gold_score

-   首先获取到真实 tags 中 length_mask-1 位置的真实的 end_ids，再从转移矩阵的 end_transition 中取出对应的值，得到 end_energy 。其中，end_transition指的是 transitions[:, STOP_TAG] 扩展为 `(batch_size, tag_size)` 后的结果。end_energy 中的每个值是每种状态转化为 stop_tag 的可能性，即 end_id to stop_tag  的可能性

-   接着，从 scores 中取出各个 tag 的 score，构成 tg_energy
    -   这里对 tag 进行了处理，tag 和 scores 进行了压缩并且可以通过 tag 的 index 值，找到从 i 到 j 的转移概率值
-   最后求和得到 gold_score

现在我们来看 viterbi_decode 

-   首先和之前的计算过程类似，得到 scores 矩阵后进行遍历。不同之处在于，需要记录所有的 partition 以及 cur_bp 分别保存在 partition_history 和 back_points ，并且计算方式为 `partition, cur_bp = torch.max(cur_values, 1)` 
    -   partition 为每一时间步上的各个 to_target 的最大可能性
    -   cur_bp 为 partition 的每个值的 from_target
-   接着，取出 mask 处理后的真实的 last_partition，再与转移矩阵 transitions 相加，得到从最后一个 tag 转移到状态 j 的转移矩阵，而后求得其每列的最大值对应的 index ，再取出 STOP_TAG 对应的列，就获得了pointer ，即最有可能转移至 STOP_TAG 的 from_target 。再将这一 from_target 覆盖到 back_points 的对应位置中
-   最后计算 decode_idx，用于在 decode 阶段解析得到 decoded sequence
    -   pointer 是 decode_idx 的最后一项，因为其保存的是最有可能转移至 STOP_TAG 的 from_target ，即 end_id
    -   倒序解码时，前一时间步的 pointer 就变成了当前时间步的 to_target 了，所以对应从 back_points 中取得其 from_target 并保存在 decode_idx 中