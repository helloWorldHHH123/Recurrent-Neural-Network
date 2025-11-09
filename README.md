# Recurrent-Neural-Network
Python | Deep Learning | Chapter 08 Recurrent Neural Network (RNN)
# 现代卷积神经网络
- 卷积神经网络可以有效地处理空间信息，那么本章的循环神经网络（recurrent neural network，
RNN）则可以更好地处理序列信息。循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可
以确定当前的输出。
- 《动手学深度学习》这本书的 第8章 “循环神经网络” 实现以及代码解析。


## 简介

- 许多使用循环网络的例子都是基于文本数据的，因此我们将在本章中重点介绍语言模型。在对序列数据进行
更详细的回顾之后，我们将介绍文本预处理的实用技术。然后，我们将讨论语言模型的基本概念，并将此讨
论作为循环神经网络设计的灵感。最后，我们描述了循环神经网络的梯度计算方法，以探讨训练此类网络时
可能遇到的问题。
- 仅限于以后方便查看和学习。

### 核心技术
- **深度学习**: PyTorch
- **数据处理**: Pandas, NumPy
- **可视化**: Matplotlib

## 🛠 环境配置

### 系统要求
- **操作系统**: Windows 10/11
- **内存**: 最低8GB，推荐16GB
- **存储**: 至少2GB可用空间

### Python环境
- **Python完整版本**: 3.9.23

### 深度学习框架
- **PyTorch**: 1.11.0+cu113
- **Torchvision**: 0.12.0+cu113
- **CUDA**: 11.3
- **当前GPU**: NVIDIA GeForce RTX 2060 with Max-Q Design

### 数据科学库
- **Pandas**: 2.0.3
- **NumPy**: 2.0.3
- **Scikit-learn**: 1.6.1
- **Matplotlib**: 3.7.2
