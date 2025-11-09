# -*- coding:utf-8 -*-
'''
作者：cy
日期：2025年11月07日
8.6 循环神经网络的简洁实现
'''

import torch
from torch import nn
from torch.nn import functional as F
import Section02
import Section03
import Section05

# 为一个完整的循环神经网络模型定义了一个RNNModel类。
# 注意，rnn_layer只包含隐藏的循环层，我们还需要创建一个单独的输出层。
class RNNModel(nn.Module):
    """循环神经网络模型"""
    # 这个初始化代码实际上就是一个构造函数
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.num_hiddens = self.rnn.hidden_size
        # 如果RNN是双向的（之后将介绍），num_directions应该是2，否则应该是1
        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.vocab_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.vocab_size)

    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.to(torch.float32)
        Y, state = self.rnn(X, state)
        # 全连接层首先将Y的形状改为(时间步数*批量大小,隐藏单元数)
        # 它的输出形状是(时间步数*批量大小,词表大小)。
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        # 简单RNN (nn.RNN) 或 GRU (nn.GRU)：它们只需要一个隐状态。
        if not isinstance(self.rnn, nn.LSTM):
            # nn.GRU以张量作为隐状态
            return torch.zeros((self.num_directions * self.rnn.num_layers,
                                batch_size, self.num_hiddens),
                               device=device)
        # LSTM (nn.LSTM)：它需要两个状态（隐状态和细胞状态）。
        else:
            # nn.LSTM以元组作为隐状态
            return (torch.zeros((
                self.num_directions * self.rnn.num_layers,
                batch_size, self.num_hiddens), device=device),
                    torch.zeros((
                        self.num_directions * self.rnn.num_layers,
                        batch_size, self.num_hiddens), device=device))

def main():
    """主函数，包含所有需要执行的代码"""
    print("🚀 Section06.py 的主函数")
    # 8.6.1 定义模型
    batch_size, num_steps = 32, 35
    train_iter, vocab = Section03.load_data_time_machine(batch_size, num_steps)
    num_hiddens = 256
    rnn_layer = nn.RNN(len(vocab), num_hiddens)
    # 使用张量来初始化隐状态，它的形状是（隐藏层数，批量大小，隐藏单元数）。
    state = torch.zeros((1, batch_size, num_hiddens))
    print(state.shape)
    # 通过一个隐状态和一个输入，我们就可以用更新后的隐状态计算输出。
    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    Y, state_new = rnn_layer(X, state)
    print(Y.shape, state_new.shape)

    # 8.6.2 训练与预测
    # 在训练模型之前，基于一个具有随机权重的模型进行预测。
    device = Section05.try_gpu()
    net = RNNModel(rnn_layer, vocab_size=len(vocab))
    net = net.to(device)
    # 很明显，这种模型根本不能输出好的结果。
    print(Section05.predict_ch8('time traveller', 10, net, vocab, device))
    # 与上一节相比，由于深度学习框架的高级API对代码进行了更多的优化，
    # 该模型在较短的时间内达到了较低的困惑度。
    num_epochs, lr = 500, 1
    Section05.train_ch8(net, train_iter, vocab, lr, num_epochs, device)

if __name__ == '__main__':
    main()
