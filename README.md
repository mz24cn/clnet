
# clNET: **OpenCL for Nets**
A Deep Learning Framework based on OpenCL, written by C++. Supports popular MLP, RNN(LSTM), CNN neural networks. 
基于OpenCL的深度学习计算框架，C++开发，支持多层感知器，长短时记忆模型，卷积神经网络。

Current Status: **pending release**.
-
Progress: Currently clnet can successfully run fully connected neural networks (MLP), CharRNN (LSTM) which uses dynamic computing graph to deal with loops, CNN (LeNet5 on MNIST dataset).  
Tested on Nvidia GTX1080, AMD R9 295X2, Intel HD Graphics 630 GPU.  
Next I will restore the support for multi-GPUs training and revise the kernels for performance tuning.

当前状态：尚未正式发布，即将发布。
-
已完成进度：
可成功运行MLP全连接多层感知器，CharRNN（LSTM，基于动态计算图的循环实现），CNN（LeNet5，MNIST）。  
三种模型均在Nvidia GTX1080, AMD R9 295X2, Intel HD Graphics 630以及Intel CPU，AMD CPU/APU上测试通过。性能尚待进一步优化。  
多显卡支持在代码重构后尚未启用，有待调试。分布式有待开发。

演示例子运行命令行：  
全连接MLP：  
.\OpenCLNet.exe MLP /ds /[0] /nf  
charRNN：  
.\OpenCLNet.exe charRNN /nf /ds /[0] :corpus\_file D:\DataSets\charRNN\obama.txt :index\_file D:\DataSets\charRNN\obama.index  
obama.txt可从[http://data.mxnet.io/mxnet/data/char_lstm.zip](http://data.mxnet.io/mxnet/data/char_lstm.zip)下载。  
MNIST CNN：  
.\OpenCLNet.exe MNIST\_CNN /ds /nf /[0]  :mnist\_folder D:\DataSets\MNIST\  
D:/DataSets/下需包含MNIST数据集文件train-images.idx3-ubyte，train-labels.idx1-ubyte，t10k-images.idx3-ubyte，t10k-labels.idx1-ubyte。可从[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)下载

I'm working hard for **clNET** official release!
-