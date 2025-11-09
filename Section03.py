# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月06日
8.3 语言模型和数据集

假设长度为T的文本序列中的词元依次为x1, x2, . . . , xT。于是，xt（1 ≤ t ≤ T）可以被认为是文
本序列在时间步t处的观测或标签。在给定这样的文本序列时，语言模型（language model）的目标是估计序
列的联合概率P(x1, x2, . . . , xT ).
'''

import Section02
import random
import torch
import matplotlib.pyplot as plt

# 下面的代码每次可以从数据中随机生成一个小批量。
# 在这里，参数batch_size指定了每个小批量中子序列样本的数目，
# 参数num_steps是每个子序列中预定义的时间步数。
# num_steps：每个训练样本（子序列）包含的词元数量
# 也称为：序列长度、时间步数、上下文长度
# num_steps 就是每个子序列的最大长度，也就是模型在一次前向传播中处理的词元序列长度。
"""
假设文本："深度学习是人工智能的一个重要分支"
分词后：["深度", "学习", "是", "人工", "智能", "的", "一个", "重要", "分支"]，
那么num_steps =2吗？因为分词后序列长度最大是2啊。
不对。
分词后的序列长度 = 整个文本分词后的词元总数，上述分词后是9
设置 num_steps = 4，意思是：把长度为9的长序列切成多个长度为4的子序列。

如果 num_steps = 2：得到4个训练样本
如果 num_steps = 4：得到2个训练样本
如果 num_steps = 8：得到1个训练样本
如果 num_steps = 9：得到1个训练样本（但会浪费，因为需要留一个位置给标签）
通常 num_steps 会设得比文本长度小，这样才能从单个文本中切出多个训练样本。
# 实际应用中根据情况选择
num_steps = 32   # 小型模型
num_steps = 512  # BERT等模型
num_steps = 2048 # GPT系列
研究任务：需要长上下文理解（如文档生成）→ 选较大的num_steps
计算资源：资源有限 → 选较小的num_steps
数据量：数据少 → 可以选大一些的num_steps
"""
"""
训练阶段最核心的公式
总需要Token数 = num_steps + 1
这是所有计算的基石！因为：
输入需要：num_steps 个token
标签需要：num_steps 个token
但它们重叠了 num_steps - 1 个token
所以实际需要：num_steps + 1 个原始token

训练阶段（有这个公式）：
需要：输入序列 + 标签序列
目标：让模型学习"给定前文预测下一个词"
必须：有完整的输入-标签对
公式：总需要Token数 = num_steps + 1
预测阶段（没有这个公式）：
只需要：输入序列
目标：生成后续文本
可以：只有输入，没有标签
公式：只需要 num_steps 个token就能开始预测
"""
def seq_data_iter_random(corpus, batch_size, num_steps):
    """使用随机抽样生成一个小批量子序列"""
    # 从随机偏移量开始对序列进行分区，随机范围包括num_steps-1
    # 如果偏移量是 num_steps，相当于丢弃了整整一个子序列的长度，这过于激进。
    # 但是每个训练样本需要 num_steps + 1 个token，而不是 num_steps 个
    # 为了避免无法产生标签的可能性
    corpus = corpus[random.randint(0, num_steps - 1):]
    # 减去1，是因为我们需要考虑标签
    # num_subseqs：能切出多少完整的子序列
    num_subseqs = (len(corpus) - 1) // num_steps
    # 长度为num_steps的子序列的起始索引
    """
    range(start, stop, step) 三个参数：   
    start：起始值（包含）    
    stop：结束值（不包含）    
    step：步长
    """
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    # 在随机抽样的迭代过程中来自两个相邻的、随机的、小批量中的子序列不一定在原始序列上相邻
    random.shuffle(initial_indices)

    def data(pos):
        # 返回从pos位置开始的长度为num_steps的序列
        return corpus[pos: pos + num_steps]

    """
    num_subseqs：总共能切出多少完整子序列
    batch_size：每个批量包含多少个子序列    
    num_batches：总共多少批量 = num_subseqs // batch_size   
    num_steps：每个子序列的长度
    例如：num_subseqs = 10    # 总共10个子序列
    batch_size = 3      # 每批3个子序列
    num_steps = 5       # 每个子序列长度5
    num_batches = 10 // 3 = 3  # 共3个完整批量
    剩余样本数 = 10 % 3 = 1     # 第4个批量不完整，通常丢弃
    一个批量包含多个子序列。
    """
    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        # 在这里，initial_indices包含子序列的随机起始索引
        initial_indices_per_batch = initial_indices[i: i + batch_size]
        X = [data(j) for j in initial_indices_per_batch]
        Y = [data(j + 1) for j in initial_indices_per_batch]
        yield torch.tensor(X), torch.tensor(Y)

# 在迭代过程中，除了对原始序列可以随机抽样外，我们还可以保证两个相邻的小批量中的子序列在原始序列
# 上也是相邻的。这种策略在基于小批量的迭代过程中保留了拆分的子序列的顺序，因此称为顺序分区。
"""
词元 (Token): 这是最基本的单位。您可以将其理解为一个词（或一个字，或一个标点符号）。
在代码中，corpus 里的每一个数字（如 corpus[i]）就代表一个词元。

序列 (Sequence): 这是由多个词元组成的。代码中的 num_steps（时间步长）定义了我们一次处理的序列有多长。
例如，如果 num_steps=35，那么一个序列就是由35个词元组成的。

批次 (Batch): 这是由多个序列组成的。代码中的 batch_size 定义了一个批次里包含多少个序列。
"""
def seq_data_iter_sequential(corpus, batch_size, num_steps):
    """使用顺序分区生成一个小批量子序列"""
    # 从随机偏移量开始划分序列
    # random.randint生成一个指定范围内的随机整数
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = torch.tensor(corpus[offset: offset + num_tokens])
    Ys = torch.tensor(corpus[offset + 1: offset + 1 + num_tokens])
    Xs, Ys = Xs.reshape(batch_size, -1), Ys.reshape(batch_size, -1)
    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_steps * num_batches, num_steps):
        X = Xs[:, i: i + num_steps]
        Y = Ys[:, i: i + num_steps]
        # 产出（“返回”）当前的小批量 X 和 Y。函数会在这里暂停，
        # 直到下一次迭代（例如 for 循环中的 next()）被调用。
        yield X, Y
# 将上面的两个采样函数包装到一个类中，以便稍后可以将其用作数据迭代器。
class SeqDataLoader:
    """加载序列数据的迭代器"""
    def __init__(self, batch_size, num_steps, use_random_iter, max_tokens):
        if use_random_iter:
            self.data_iter_fn = seq_data_iter_random
        else:
            self.data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = Section02.load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self.data_iter_fn(self.corpus, self.batch_size, self.num_steps)


# 定义一个函数load_data_time_machine，它同时返回数据迭代器和词表
def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    """返回时光机器数据集的迭代器和词表"""
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section03.py 的主函数")
    # print(Section02.DATA_HUB)
    # print(Section02.DATA_URL)
    # 根据8.2节中介绍的时光机器数据集构建词表，并打印前10个最常用的（频率最高的）单词。
    # 文本转换成词元
    tokens = Section02.tokenize(Section02.read_time_machine())
    # 因为每个文本行不一定是一个句子或一个段落，因此我们把所有文本行拼接到一起
    """
    下面一行替代形式：
    corpus = []
    for line in tokens:
        for token in line:
            corpus.append(token)
    """
    corpus = [token for line in tokens for token in line]
    # 将词元转换成词汇表对象，这是一个类的对象
    # 构建 token_to_idx 和 idx_to_token 映射
    vocab = Section02.Vocab(corpus)
    print(vocab.token_freqs[:10])
    freqs = [freq for token, freq in vocab.token_freqs]
    """
    图中显示，词频以一种明确的方式迅速衰减。这意味着单词的频率满足齐普夫定律（Zipf’s law），log ni = −α log i + c,
    这告诉我们想要通过计数统计和平滑来建模单词是不可行的，因为这样
    建模的结果会大大高估尾部单词的频率，也就是所谓的不常用单词。
    """
    # 设置图形
    plt.figure(figsize=(6, 3))
    # 绘制频率分布图
    plt.plot(freqs)
    # 设置坐标轴标签
    plt.xlabel('token: x')
    plt.ylabel('frequency: n(x)')
    # 设置对数坐标轴
    plt.xscale('log')
    plt.yscale('log')
    # 添加网格和标题
    plt.grid(True, alpha=0.3)
    plt.title('Token Frequency Distribution (Log-Log Scale)')
    plt.tight_layout()
    plt.show()
    # 查看二元语法的频率是否与一元语法的频率表现出相同的行为方式
    bigram_tokens = [pair for pair in zip(corpus[:-1], corpus[1:])]
    bigram_vocab = Section02.Vocab(bigram_tokens)
    print(bigram_vocab.token_freqs[:10])

    """
    corpus[:-2]：从开始到倒数第3个元素
    corpus[1:-1]：从第2个到倒数第2个元素   
    corpus[2:]：从第3个到最后一个元素
    下面一行在词元基础上操作，实现的是：
    # 三个错位列表
    列表A: [1, 2, 3, 4]    ← corpus[:-2]
    列表B: [2, 3, 4, 5]    ← corpus[1:-1]  
    列表C: [3, 4, 5, 6]    ← corpus[2:]
    
    # zip并行取元素
    第一次取：A[0]=1, B[0]=2, C[0]=3 → triple=(1,2,3)
    第二次取：A[1]=2, B[1]=3, C[1]=4 → triple=(2,3,4)
    第三次取：A[2]=3, B[2]=4, C[2]=5 → triple=(3,4,5)
    第四次取：A[3]=4, B[3]=5, C[3]=6 → triple=(4,5,6)
    结果：[(1,2,3), (2,3,4), (3,4,5), (4,5,6)]
    每个triple就是连续的三个词元。
    """
    trigram_tokens = [triple for triple in zip(corpus[:-2], corpus[1:-1], corpus[2:])]
    """
    Vocab 不关心词元类型，可以是字符、单词或三元组
    它把每个三元组 (1,2,3) 当作一个整体词元处理
    统计这些三元组的出现频率，构建 {三元组: 索引} 的映射
    下面一行可能实现的是：
    token_to_idx = {(1,2,3): 0, (2,3,4): 1, ...}
    idx_to_token = [(1,2,3), (2,3,4), ...]
    """
    trigram_vocab = Section02.Vocab(trigram_tokens)
    print(trigram_vocab.token_freqs[:10])

    # 直观地对比三种模型中的词元频率：一元语法、二元语法和三元语法
    bigram_freqs = [freq for token, freq in bigram_vocab.token_freqs]
    trigram_freqs = [freq for token, freq in trigram_vocab.token_freqs]
    # 设置图形
    plt.figure(figsize=(8, 5))
    # 绘制单元语法频率
    plt.plot(freqs, label='unigram')
    # 绘制二元语法频率
    plt.plot(bigram_freqs, label='bigram')
    # 绘制三元语法频率
    plt.plot(trigram_freqs, label='trigram')
    # 设置坐标轴标签和比例
    plt.xlabel('token: x')
    plt.ylabel('frequency: n(x)')
    plt.xscale('log')
    plt.yscale('log')
    # 设置图例和其他美化
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.title('N-gram Frequency Distribution')
    plt.tight_layout()
    plt.show()

    """
    token (词元)：是数据的基本单位（比如一个词或一个字）。
    num_steps：是序列的长度（length），这个长度的单位就是 token。
    
    假设语料库 corpus 是一句话："The quick brown fox jumps over the lazy dog"。

    Tokenization (词元化): 你先把这句话转换成 token 列表： ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
     （在代码中，这些词会被转换成数字ID，比如 [0, 1, 2, 3, 4, 5, 0, 6, 7]）

    num_steps 的作用:

    如果设置 num_steps = 3： 那么这个数据迭代器就会把数据切成长度为 3 的序列。比如，产出的一个小批量 X 中的一个序列可能是：
     ['The', 'quick', 'brown'] (即 [0, 1, 2]) 对应的 Y 序列就是： ['quick', 'brown', 'fox'] (即 [1, 2, 3])

    如果设置 num_steps = 5： 那么一个序列就会是： ['The', 'quick', 'brown', 'fox', 'jumps'] (即 [0, 1, 2, 3, 4]) 对应的 Y 序列就是：
     ['quick', 'brown', 'fox', 'jumps', 'over'] (即 [1, 2, 3, 4, 5])
    """
    # 从数据中随机生成一个小批量
    my_seq = list(range(35))
    for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

    # 通过顺序分区读取每个小批量的子序列的特征X和标签Y。
    for X, Y in seq_data_iter_sequential(my_seq, batch_size=2, num_steps=5):
        print('X: ', X, '\nY:', Y)

if __name__ == '__main__':
    main()
