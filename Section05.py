# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月06日
8.5 循环神经网络的从零开始实现

无隐状态的神经网络：没有记忆。它处理每个输入时，都像是第一次见到一样，完全独立。
有隐状态的循环神经网络：拥有记忆。它在处理当前输入时，会参考它上一步处理过的输入。
'''

import math
import torch
from torch import nn
from torch.nn import functional as F
import Section02
import Section03
import numpy as np
import time


# 8.5.2 初始化模型参数
# 初始化循环神经网络模型的模型参数。隐藏单元数num_hiddens是一个可调的超参数。
# 当训练语言模型时，输入和输出来自相同的词表。因此，它们具有相同的维度，即词表的大小。
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params

# 8.5.3 循环神经网络模型
def init_rnn_state(batch_size, num_hiddens, device):
    # 为什么是0？ 这是一种标准做法，表示在序列开始时，RNN没有任何“先验知识”或“记忆”。
    # 返回的是一个只包含一个元素的元组（Tuple）
    """
    对于简单RNN，它返回 (H_0,) —— 一个包含初始隐状态的1元组。
    对于LSTM，它会返回 (H_0, C_0) —— 一个包含两个初始状态的2元组。
    这样做的好处是，后续使用这个状态的 rnn 函数不需要改变。
    rnn 函数可以统一地从元组中解包它需要的状态，而不用去关心它到底是在处理简单RNN还是LSTM。这是一种非常灵活和可扩展的设计。
    """
    return (torch.zeros((batch_size, num_hiddens), device=device), )

# 下面的rnn函数定义了如何在一个时间步内计算隐状态和输出。
def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        # H 就是“隐状态” (Hidden State)，它是RNN的“记忆”核心。
        # 为了让神经网络能够有效地“融合新信息”和“旧记忆”。
        # RNN的目标是在每个时间步 t，根据当前输入 X_t 和上一步的记忆 H_{t-1}，来创造一个新的记忆 H_t。
        # 下面这个公式做的就是这个：
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)

# 定义了所有需要的函数之后，接下来创建一个类来包装这些函数，
# 并存储从零开始实现的循环神经网络模型的参数。
class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""
    def __init__(self, vocab_size, num_hiddens, device,get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        # 调用rnn(inputs, state, params)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        # 调用init_rnn_state(batch_size, num_hiddens, device)
        return self.init_state(batch_size, self.num_hiddens, device)

def try_gpu(i=0):
    if torch.cuda.device_count()>=i+1:
        # PyTorch 的设备字符串有严格的格式要求：
        # f'cuda:{i}'，冒号后面没有空格，正确
        # f'cuda: {i}'，冒号后面没有空格，错误
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

# 8.5.4 预测
# 首先定义预测函数来生成prefix之后的新字符，其中的prefix是一个用户提供的包含多个字符的字符串。
# 在循环遍历prefix中的开始字符时，我们不断地将隐状态传递到下一个时间步，但是不生成任何输出。
# 这被称为预热（warm‐up）期，因为在此期间模型会自我更新（例如，更新隐状态），但不会进行预测。
# 预热期结束后，隐状态的值通常比刚开始的初始值更适合预测，从而预测字符并输出它们。
def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    # 获取outputs列表中的最后一个字符索引，并将其转换为 (1, 1) 形状的张量（即 batch_size=1, num_steps=1）
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]: # 预热期
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds): # 预测num_preds步
        y, state = net(get_input(), state)
        # y.argmax(...) 找出概率最高的那个字符的索引（即模型认为“最可能”的下一个字符）。
        # 将这个预测出的新字符索引添加到 outputs 列表。
        # 这个新添加的字符将成为下一次循环中 get_input() 的来源。这个过程（用自己的输出作为下一步的输入）被称为自回归 (Autoregressive)。
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    # outputs 列表中现在包含了prefix的全部索引和所有num_preds个新预测的索引。
    return ''.join([vocab.idx_to_token[i] for i in outputs])

# 8.5.5 梯度裁剪
# 对于长度为T的序列，我们在迭代中计算这T个时间步上的梯度，将会在反向传播过程中产生长度为O(T)的
# 矩阵乘法链。如 4.8节所述，当T较大时，它可能导致数值不稳定，例如可能导致梯度爆炸或梯度消失。
# 因此，循环神经网络模型往往需要额外的方式来支持稳定训练。
def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    def start(self):
        """启动计时器"""
        self.tik = time.time()
    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time()-self.tik)
        return self.times[-1]
    def sum(self):
        """返回时间总和"""
        return sum(self.times)
    def cumsum(self):
        """返回累计时间"""
        """
        np.array(self.times) - 将列表转换为NumPy数组
        .cumsum() - 计算累积和
        .tolist() - 转换回Python列表
        """
        return np.array(self.times).cumsum().tolist()

# 8.5.6 训练
def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, Timer()
    # 训练损失之和,词元数量
    metric = [0.0] * 2
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            # 这个 if 条件只在 net 是一个PyTorch官方模块，并且它返回的 state 是一个单独的张量（而不是元组）时，才会为 True。
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                # .detach_() 是实现“截断反向传播” (Truncated BPTT) 的关键。
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        # .reshape(-1) (展平)
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        # 你的 state 变量就像一个单独的“记忆插槽”，它只持有最近一次的隐状态。
        y_hat, state = net(X, state)
        # y.long() 的意思是：将 y 这个张量（Tensor）的数据类型转换为 torch.long（即64位整型）。
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()   # 计算梯度
            grad_clipping(net, 1)   # 对参数的“梯度” (Gradients) 进行截断
            updater.step()   # 使用这两样东西：裁剪后的梯度 (Clipped Gradients)，上一步的参数值 (Old Parameter Values)来执行一次参数更新
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)

        metric=[a+float(b) for a,b in zip(metric,[l * y.numel(), y.numel()])]
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()

def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 循环神经网络模型的训练函数既支持从零开始实现，也可以使用高级API来实现。
def train_ch8(net, train_iter, vocab, lr, num_epochs, device, use_random_iter=False):
    """训练模型（定义见第8章）"""
    loss = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
    # legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            # animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))

def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section05.py 的主函数")
    # num_steps包含多个词元（token）
    batch_size, num_steps = 32, 35
    # 词汇表对象vocab，这是一个类的对象
    # 构建 token_to_idx 和 idx_to_token 映射
    train_iter, vocab = Section03.load_data_time_machine(batch_size, num_steps)

    # 8.5.1 独热编码
    print('独热编码: ', F.one_hot(torch.tensor([0, 2]), len(vocab)))
    X = torch.arange(10).reshape((2, 5))
    """
    .T 是转置 (Transpose) 操作，它会把 X 的维度（行和列）交换。
    遍历 X.T (形状 [5, 2]) 中的每一个整数，并将其转换为一个长度为 28 的向量。
    例如，0 会变成 [1, 0, 0, ..., 0] (长度28)
    例如，5 会变成 [0, 0, 0, 0, 0, 1, ..., 0] (长度28)
    """
    print(F.one_hot(X.T, 28).shape)
    # 8.5.2 初始化模型参数

    # 检查输出是否具有正确的形状。例如，隐状态的维数是否保持不变。
    num_hiddens = 512
    """
    没有在创建 net 时 调用 get_params。只是把 get_params 这个函数本身（像一个“说明书”或“配方”）作为参数传递给了 RNNModelScratch 的构造函数 (__init__)。
    RNNModelScratch 的构造函数内部会接收这个“配方”，然后由它自己来调用这个函数。
    
    当创建 net 时，RNNModelScratch 类的 __init__ 方法（构造函数）被调用。
    RNNModelScratch 的 __init__ 方法在执行到内部的 self.params = ... 这一行时，
    它使用它已经拥有的 self.vocab_size、self.num_hiddens 和 self.device 作为参数，
    来调用它刚刚接收到的 get_params_function（也就是传进去的 get_params）
    
    这种“将函数作为参数传递，以便稍后在内部调用”的模式非常常见，它是一种解耦（Decoupling）的设计：
    RNNModelScratch 类不需要知道参数是如何被创建的（是用 normal 还是 zeros 还是其他方法），
    它只关心一件事：“请给我一个函数，这个函数能接收 (vocab_size, hiddens, device) 并返回我需要的 params 列表就行”。
    """
    net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params, init_rnn_state, rnn)
    state = net.begin_state(X.shape[0], try_gpu())
    Y, new_state = net(X.to(try_gpu()), state)
    print(Y.shape, len(new_state), new_state[0].shape)

    # 测试predict_ch8函数。我们将前缀指定为time traveller，并基于这个前缀生成10个后续字符。
    # 鉴于我们还没有训练网络，它会生成荒谬的预测结果。
    """
    “预热”（Warm-up）是如何使用RNN（循环神经网络）来生成文本这个过程中非常关键的一步，它的意思是：
    在让模型“预测”新字符之前，先强迫它“阅读并理解”你给它的上下文（prefix）。
    想象一下RNN是一个人，它的**“隐状态” (Hidden State)** 就是它的**“短期记忆”**。
    在这个过程中，我们只关心更新它的“记忆”（隐状态），而不在乎它在阅读每个字时“想”了什么（即，我们丢弃它在预热期间的所有输出）。
    为了让RNN生成合理连贯的文本，你不能让它“凭空想象”。
    “预热”在干什么：它通过处理 prefix 字符串，将RNN的**隐状态（记忆）从一个“无信息”的初始状态，转变为一个“充满上下文”**的就绪状态。
    为什么这么做：只有当RNN的“记忆”里有了 prefix 的上下文时，它接下来的预测（生成的新字符）才能与 prefix 逻辑上衔接起来。
    """
    # 参数：字符串，预测步，网络，词表，设备
    print(predict_ch8('time traveller ', 10, net, vocab, try_gpu()))

    # 训练循环神经网络模型。因为我们在数据集中只使用了10000个词元，
    # 所以模型需要更多的迭代周期来更好地收敛。
    num_epochs, lr = 500, 1
    train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu())

    # 检查一下使用随机抽样方法的结果。
    net = RNNModelScratch(len(vocab), num_hiddens, try_gpu(), get_params, init_rnn_state, rnn)
    train_ch8(net, train_iter, vocab, lr, num_epochs, try_gpu(), use_random_iter=True)

if __name__ == '__main__':
    main()
