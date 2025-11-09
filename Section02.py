# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月06日

8.2 文本预处理

这部分主要完成词元列表 idx_to_token（添加新词元）、
词元字典 token_to_idx（添加词元到索引的映射）的构建，即词表，
暂未涉及到词向量

文本的常见预处理步骤包括：
1. 将文本作为字符串加载到内存中。
2. 将字符串拆分为词元（如单词和字符）。
3. 建立一个词表，将拆分的词元映射到数字索引。
4. 将文本转换为数字索引序列，方便模型操作。
'''

"""
    模块	          主要用途	            在AI中的典型应用
collections	    增强的数据结构	           数据统计、分组、缓存
    re         正则表达式、文本处理	   数据清洗、特征提取、文本预处理
"""

import collections
import re
import hashlib
import os
import tarfile
import zipfile
import requests

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
DATA_HUB['time_machine'] = (DATA_URL + 'timemachine.txt',
                            '090b5e7e70c295757f55df93cb0a180b9691891a')
def download(name, cache_dir=os.path.join('.', 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)

def read_time_machine():
    """将时间机器数据集加载到文本行的列表中"""
    """
    lines = f.readlines()
    读取文件的所有行    
    返回：字符串列表，每个元素是一行文本    
    保留行尾的换行符 \n
    例如，文件内容：
    The Time Machine
    Chapter 1
    
    The Time Traveller was expounding...
    读取结果：
    lines = [
    "The Time Machine\n",
    "Chapter 1\n", 
    "\n",
    "The Time Traveller was expounding...\n"
    ]
    """
    with open(download('time_machine'), 'r') as f:
        lines = f.readlines()
    """
    re.sub('[^A-Za-z]+', ' ', line)
    [^A-Za-z]+：匹配非字母字符
    [^A-Za-z]：不是A-Z或a-z的字符
    +：一个或多个这样的字符
    替换为空格：将所有非字母字符序列替换为单个空格
    例如：输入： "Hello, world! 123"
         输出： "Hello  world   "
    去除首尾空格：.strip()
    转为小写：.lower()
    """
    return [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]


# 8.2.2 词元化
# 下面的tokenize函数将文本行列表（lines）作为输入，
# 列表中的每个元素是一个文本序列（如一条文本行）。
# 每个文本序列又被拆分成一个词元列表，词元（token）是文本的基本单位。
# 最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）。
def tokenize(lines, token='word'):
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        # split() 方法默认按空白字符（空格、制表符、换行符等）分割字符串：
        # 用于对文本中的每一行进行分词操作
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

# 8.2.3 词表
# 词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。
# 现在，让我们构建一个字典，通常也叫做词表（vocabulary），
# 用来将字符串类型的词元映射到从0开始的数字索引中。
# 语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元“<unk>”。
# 填充词元（“<pad>”）；序列开始词元（“<bos>”）；序列结束词元（“<eos>”）。
"""
这段代码正是在构建词表（Vocabulary）
词表就是文本中所有唯一单词（词元）的集合，并为每个单词分配一个唯一的数字索引。
其本质是一个映射系统：
单词（人类可读） ⇄ 索引（机器可处理）
"""
class Vocab:
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        """
        counter.items(): 将字典转换为元组，例如：
        counter = {'the': 3000, 'and': 1500, 'of': 1200, 'machine': 50}
        counter.items() = [('the', 3000), ('and', 1500), ('of', 1200), ('machine', 50)]
        
        key=lambda x: x[1]：按元组的第二个元素（词频）排序
        reverse=True：降序排列（从大到小）
        
        x 是 counter.items() 返回的每个元组。
        """
        # _token_freqs 不是字典，而是一个排序后的元组列表。
        # 将词频统计结果按频率从高到低排序
        # 词频信息存储在 _token_freqs 中
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],reverse=True)
        # 未知词元的索引为0，字符串列表，存储所有词元
        # 如，['<unk>', '<pad>', '<bos>', '<eos>']
        self.idx_to_token = ['<unk>'] + reserved_tokens
        # 给未知、保留的词元加上索引
        # 索引信息存储在 token_to_idx 中，字典{词元：索引}
        # 如，{'<unk>': 0, 'a': 1, 'b': 2, 'c': 3}
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        # 更新词元列表、词元字典
        # 更新词元列表 idx_to_token（添加新词元）
        # 更新词元字典 token_to_idx（添加词元到索引的映射）
        for token, freq in self._token_freqs:
            # 丢弃出现频率较低的词元
            if freq < min_freq:
                break
            if token not in self.token_to_idx:   # 避免重复添加
                self.idx_to_token.append(token)
                # 添加索引
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    # __getitem__ 是 Python 的魔法方法，当你使用 obj[key] 语法时会自动调用。
    # 返回索引而不是词频
    """
     索引 | 单词  | 词向量（128维）
    -----|-------|-------------------
    0    | <unk> | [0.1, 0.2, ..., 0.8]
    1    | <pad> | [0.0, 0.0, ..., 0.0]
    ...
    4    | the   | [0.3, -0.1, ..., 0.5]  ← 索引4指向这个词向量
    5    | cat   | [0.7, 0.2, ..., -0.3]  ← 索引5指向这个词向量
    ...
    """
    def __getitem__(self, tokens):
        """
        处理单个单词的情况,
        如果 tokens 不是列表或元组（即单个字符串）
        在 token_to_idx 字典中查找该单词的索引
        如果单词不存在，返回 self.unk（通常是 0，对应 <unk>）
        """
        if not isinstance(tokens, (list, tuple)):
            # get 是 Python 字典的一个内置方法，用于安全地获取字典中的值。
            # 用法：字典.get(键, 默认值)
            # 如果键存在于字典中 → 返回该键对应的值
            # 如果键不存在于字典中 → 返回指定的默认值
            return self.token_to_idx.get(tokens, self.unk)
        """
        处理单词列表的情况,
        如果 tokens 是列表或元组
        对列表中的每个单词递归调用 __getitem__    
        返回索引列表
        """
        return [self.__getitem__(token) for token in tokens]

    # 将索引转换回单词
    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        # self.idx_to_token 是一个列表，不是字典！
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    # len(tokens) == 0: 检查 tokens 是否为空列表
    # isinstance(tokens[0], list): 检查第一个元素是否是列表类型
    """
    [token for line in tokens for token in line]执行顺序等价于：
    result = []
    for line in tokens:      # 外层循环先执行
        for token in line:   # 内层循环后执行
            result.append(token)
    """
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    # collections.Counter(tokens) 的作用: 统计列表中每个元素的出现次数
    # 返回一个 Counter 对象（类似字典）
    return collections.Counter(tokens)

# 8.2.4 整合所有功能
def load_corpus_time_machine(max_tokens=-1):
    """返回时光机器数据集的词元索引列表和词表"""
    # 读取文本为一个列表变量lines
    lines = read_time_machine()
    # 将文本行拆分为字符词元？为啥不用单词词元？
    tokens = tokenize(lines, 'char')
    """
    字符词元: 将文本分割成单个字符
    text = "Hello world!"
    # 字符级分词
    tokens = ['H', 'e', 'l', 'l', 'o', ' ', 'w', 'o', 'r', 'l', 'd', '!']
    单词词元: 将文本分割成完整的单词
    text = "Hello world!"
    # 单词级分词
    tokens = ['Hello', 'world', '!']  # 或者 ['Hello', 'world!']
    """
    # 将词元转换为词表，即带索引的字符列表
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    # 将分词后的文本转换为数字索引序列
    # 这行代码将嵌套的分词结果展平为一个长的数字序列，即将所有文本数据平铺成一个长的索引序列。
    # 每个词元都被替换为它在词汇表中的索引，为后续的模型训练做好准备。
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    # 返回索引列表（整个文本的平铺数字序列）、词表对象（字符到索引的映射）
    # Vocab类对象内部包含：
    # token_to_idx 字典：{'a': 1, 'b': 2, ...}
    # idx_to_token 列表：['<unk>', 'a', 'b', ...]
    return corpus, vocab

def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section02.py 的主函数")
    # 8.2.1 读取数据集
    # 从H.G.Well的时光机器99中加载文本。这是一个相当小的语料库，有30000多个单词，
    # 只但足够小试牛刀，而现实中的文档集合可能会包含数十亿个单词。
    lines = read_time_machine()
    print(f'# 文本总行数: {len(lines)}')
    print(lines[0])
    print(lines[10])

    # 8.2.2 词元化
    # 实际上就是字符串处理
    tokens = tokenize(lines)
    # 返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（string）。
    for i in range(11):
        print(tokens[i])

    # 8.2.3 词表
    # 首先使用时光机器数据集作为语料库来构建词表，然后打印前几个高频词元及其索引。
    vocab = Vocab(tokens)
    print('*'*164)
    print(list(vocab.token_to_idx.items())[:10])
    # 将每一条文本行转换成一个数字索引列表
    for i in [0, 10]:
        print('文本:', tokens[i])
        print('索引:', vocab[tokens[i]])

    # 8.2.4 整合所有功能
    corpus, vocab = load_corpus_time_machine()
    print('len(corpus) = ',len(corpus), ', len(vocab) = ', len(vocab))


if __name__ == '__main__':
    main()
