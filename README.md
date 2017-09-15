
# clnet: ***OpenCL for Nets***
A Deep Learning Framework based on OpenCL, written by C++. Supports popular MLP, RNN(LSTM), CNN neural networks. 
基于OpenCL的深度学习计算框架，C++开发，支持多层感知器，长短时记忆模型，卷积神经网络。

Current Status: ***pending release***.
-
Progress: Currently clnet can successfully run fully connected neural networks (MLP), CharRNN (LSTM) which uses dynamic computing graph to deal with loops, CNN (LeNet5 on MNIST dataset).
MLP tested on Intel, AMD, and Nvidia GPU. RNN and CNN successfully run on Nvidia GPU. AMD GPU is still in debugging for RNN and CNN. Intel integrated GPU is waiting for performance tuning.

当前状态：尚未正式发布，即将发布。
-
已完成进度：
可成功运行MLP全连接多层感知器，CharRNN（LSTM，基于动态计算图的循环实现），CNN（LeNet5，MNIST）。全连接MLP模型可以在Intel，AMD，NVIDIA显卡上运行。
RNN和CNN模型只在NVIDIA显卡上通过测试，AMD显卡运行结果异常，仍在调试中。Intel核显性能差距较大未测试。
多显卡支持在代码重构后尚未启用，有待调试。分布式有待开发。

I'm working hard for ***CLNet*** official release!
-
