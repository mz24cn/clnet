
# clNET: **OpenCL for Nets**
A Deep Learning Framework based on OpenCL, written by C++. Supports popular MLP, RNN(LSTM), CNN neural networks. 
基于OpenCL的深度学习计算框架，C++开发，支持多层感知器，长短时记忆模型，卷积神经网络。

Progress: Currently clnet can successfully run fully connected neural networks (MLP), CharRNN (LSTM) which uses dynamic computing graph to deal with loops, CNN (LeNet5 on MNIST dataset).  
Tested on Nvidia GTX1080, AMD R9 295X2, Intel HD Graphics 630 GPU.  
Support multiple devices training.

已完成进度：
可成功运行MLP全连接多层感知器，CharRNN（LSTM，基于动态计算图的循环实现），CNN（LeNet5，MNIST）的训练及推断。  
三种模型均在Nvidia GTX1080, AMD R9 295X2, Intel HD Graphics 630以及Intel CPU，AMD CPU/APU上测试通过。  
测试通过的编译环境：  
Windows 10，MSVS2015；  
Linux CentOS 7，g++ 4.8.5，Makefile；  
eclipse/CDT，MinGW64 6；  
eclipse/CDT，CrossGCC: CodeBench Lite 2014/05（gcc 4.8.3）；  
Kernel性能尚待进一步优化（矩阵乘法gemm及小卷积核FFT优化算法）。   
支持多显卡训练。分布式有待开发。

演示例子运行命令行：  
全连接MLP：  
```
.\Release\OpenCLNet.exe MLP /ds
```  
charRNN：
```  
.\Release\OpenCLNet.exe charRNN /ds :corpus_file D:\DataSets\charRNN\obama.txt :index_file D:\DataSets\charRNN\obama.index
p
save D:\DataSets\charRNN\epoch520_91%.clnetparams
``` 
charRNN推断：
``` 
.\Release\OpenCLNet.exe charRNN /p :index_file D:\DataSets\charRNN\obama.index :params_file D:\DataSets\charRNN\epoch520_91%.clnetparams :sample "Now it's time"
``` 
obama.txt可从[http://data.mxnet.io/mxnet/data/char_lstm.zip](http://data.mxnet.io/mxnet/data/char_lstm.zip)下载。  
MNIST CNN：  
```
.\Release\OpenCLNet.exe MNIST_CNN /ds :mnist_folder D:\DataSets\MNIST\
```  
D:/DataSets/下需包含MNIST数据集文件train-images.idx3-ubyte，train-labels.idx1-ubyte，t10k-images.idx3-ubyte，t10k-labels.idx1-ubyte。可从[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)下载。目录名末尾请加上路径分隔符。

如何调试：  
“/ds”生成执行树，“/ss”执行到第一个Tensor，停留，等待交互命令：  
```
.\Release\OpenCLNet.exe MLP /ss /ds /0  
```
<pre>
clnet::type::XavierNormalDistributionInitializer                XavierNormalDistributionInitializer
-       clnet::type::Weight             l0_weight[2,4096]
        clnet::type::Bias               l0_bias[4096]
        clnet::type::Weight             l1_weight[4096,1]
        clnet::type::Bias               l1_bias[1]
clnet::type::IterativeOptimizer         IterativeOptimizer[4]
        clnet::InstantTensor            data_generator
        clnet::type::Data               X[128,2]
        clnet::type::Weight             l0_weight[2,4096]
        clnet::type::Bias               l0_bias[4096]
        clnet::type::FullyConnectedLayer                FCLayer_0=sigmoid(l0_weight*X+l0_bias)
        clnet::type::Output             FCLayer_0[128,4096]
        clnet::type::Weight             l1_weight[4096,1]
        clnet::type::Bias               l1_bias[1]
        clnet::type::FullyConnectedLayer                FCLayer_1=softrelu(l1_weight*FCLayer_0+l1_bias)
        clnet::type::Output             FCLayer_1[128,1]
        clnet::type::Data               Y[128]
        clnet::back::Loss               linear_regression(FCLayer_1,Y)
        clnet::back::Gradient           gradient(FCLayer_1)[128,1]
        clnet::back::FullyConnectedLayer                back:FCLayer_1=softrelu(l1_weight*FCLayer_0+l1_bias)
        clnet::back::Gradient           gradient(FCLayer_0)[128,4096]
        clnet::back::FullyConnectedLayer                back:FCLayer_0=sigmoid(l0_weight*X+l0_bias)
        clnet::back::Gradient           gradient(l0_weight)[2,4096]
        clnet::back::Gradient           gradient(l0_bias)[4096]
        clnet::back::Gradient           gradient(l1_weight)[4096,1]
        clnet::back::Gradient           gradient(l1_bias)[1]
        clnet::type::StochasticGradientDescentUpdater           SGD
-               clnet::type::Weight             l0_weight[2,4096]
                clnet::type::Bias               l0_bias[4096]
                clnet::type::Weight             l1_weight[4096,1]
                clnet::type::Bias               l1_bias[1]
-       clnet::InstantTensor            MLPMonitor

[1,@2018-06-30 16:24:29] GeForce GTX 1050 Ti (kernels build: 119ms)
[debugger] interactive thread started on device 1.
[debugger] device 1 break on IterativeOptimizer: clnet::type::IterativeOptimizer 
</pre>
执行到SGD（别名为SGD的Tensor）：  
```
g SGD
```  
<pre>
[debugger] device 1 continue to run.  
[debugger] device 1 break on SGD: clnet::type::StochasticGradientDescentUpdater 
</pre> 
观察输入样本X（别名为X的Tensor）：  
```
d X  
```
<pre>
        this:                   0xfa1e60  
        type:                   clnet::type::Data  
        alias:                  X  
        volume:                 256  
        dimensions:             [128,2]  
        size:                   1024 bytes  
        pointer:                0xe7aac0  
        gradient:               NULL  
        inputs:  
                data_generator[]: clnet::InstantTensor  
        peers:  
                FCLayer_0=sigmoid(l0_weight*X+l0_bias)[]: clnet::type::FullyConnectedLayer  
</pre>
```
X  
```
<pre>
X[128,2]: clnet::type::Data  
0  
0:      1.00375,2.69076  
1:      1.57991,3.42622  
2:      2.75503,2.43962  
3:      2.05087,3.68789  
4:      3.46852,3.23981  
5:      1.52232,3.57683  
6:      3.1315,2.5406  
7:      1.91198,1.04495  
8:      1.27421,2.09336  
9:      1.44194,1.4977  
 ...  
</pre>
观察梯度值：  
```
d l0_weight
```
<pre>
        this:                   0x2b227b0
        type:                   clnet::type::Weight
        alias:                  l0_weight
        volume:                 8192
        dimensions:             [2,4096]
        size:                   32768 bytes
        pointer:                0x2b2c900
        gradient:               gradient(l0_weight)[2,4096]: clnet::back::Gradient
        inputs:
        peers:
                FCLayer_0=sigmoid(l0_weight*X+l0_bias)[]: clnet::type::FullyConnectedLayer
</pre>
```
gradient(l0_weight)
```
<pre>
gradient(l0_weight)[2,4096]: clnet::back::Gradient
0
0:      0.0932272,-0.00103467,0.616816,0.0487299,0.108153,0.453982,-0.168111,0.00612603,0.0466066,0.0776809,0.480914,0.00167271,-0.0579107,-0.171267,-0.00544866,0.0305377,0.396773,-0.0364095,-0.0105135,-0.244325,0.0070936,-0.0271294,0.0982886,0.000907668,0.0083473,0.000168261,0.038511,-0.00443278,-0.141771,-0.000452508,0.0574187,0.59741,-0.0461692,0.0273872,0.0211383,0.0937608,-0.0543251,-0.0177396,0.0404992,0.244961 ...
1:      0.043596,-0.105236,0.252182,0.0135588,0.0468406,0.208793,-0.0282288,0.0436221,0.0046685,0.0364535,0.231056,0.0131293,-0.0219158,-0.0984129,-0.000470661,0.010817,0.0848113,-0.00210151,-0.00500153,-0.113508,0.00290996,-0.00091675,-0.0437556,0.000426235,0.0348718,6.88916e-005,0.011789,-0.0166271,-0.046225,-0.000272511,0.0210079,0.22276,-0.0209225,0.0109369,0.00923857,0.0413359,0.0153701,0.0267138,0.0193877,0.177686 ...  
</pre>
只看 部分数据：  
```
gradient(l0_weight)[:,0:8]
```
<pre>
data[0:2/2,0:8/4096] for gradient(l0_weight)[2,4096]: clnet::back::Gradient
0:      0.0932272,-0.00103467,0.616816,0.0487299,0.108153,0.453982,-0.168111,0.00612603
1:      0.043596,-0.105236,0.252182,0.0135588,0.0468406,0.208793,-0.0282288,0.0436221  
</pre>
```
l0_weight[:,0:8]
```
<pre>
data[0:2/2,0:8/4096] for l0_weight[2,4096]: clnet::type::Weight  
0:      -0.280252,-1.62137,0.129004,0.495599,0.42723,0.0478061,-0.688217,1.87265  
1:      1.73239,0.18904,-0.326688,0.204418,-2.56337,-0.718758,-0.185233,-0.827314  
</pre>
单步模式，执行完SGD，观察参数的变化：  
```
s
```
<pre>
[debugger] step into mode activated.
</pre>  
```
c
```
<pre>
[debugger] device 1 continue to run.  
[debugger] device 1 break on MLPMonitor: clnet::InstantTensor  
</pre>
```
l0_weight[:,0:8]
```
<pre>
data[0:2/2,0:8/4096] for l0_weight[2,4096]: clnet::type::Weight  
0:      -0.280253,-1.62137,0.128998,0.495598,0.427229,0.0478016,-0.688215,1.87265  
1:      1.73239,0.189041,-0.326691,0.204418,-2.56337,-0.71876,-0.185233,-0.827315  
</pre>
修改超参数：  
```
SGD.learning_rate
```
<pre>
[debugger] SGD.learning_rate = 1e-005
</pre>
```
SGD.learning_rate *= 0.5
```
<pre>
[debugger] SGD.learning_rate = 1e-005  
[debugger] SGD.learning_rate *= 0.5  
[debugger] SGD.learning_rate = 5e-006  
</pre>
执行profile，性能调优：  
```
pf
```
<pre>
[debugger] profile mode activated.  
</pre>
```
g
```
<pre>
[debugger] breakpoint removed.  
</pre>
```
c
```
<pre>
[debugger] device 1 continue to run.  
[1,0,4ms] error rate: 0.331467  
[1,2000,39006/s] error rate: 0.00325364  
[1,4000,39072/s] error rate: 0.00251041  
</pre>
```
p
```
<pre>
[debugger] breakpoint added on SGD.  
[debugger] device 1 break on SGD: clnet::type::StochasticGradientDescentUpdater  
</pre>
```
pf list
```
<pre>
back:FCLayer_1=softrelu(l1_weight*FCLayer_0+l1_bias): clnet::back::FullyConnectedLayer:              3s.271ms/20%  
FCLayer_1=softrelu(l1_weight*FCLayer_0+l1_bias): clnet::type::FullyConnectedLayer:              3s.87ms/19%  
back:FCLayer_0=sigmoid(l0_weight*X+l0_bias): clnet::back::FullyConnectedLayer:          923ms/5%  
SGD: clnet::type::StochasticGradientDescentUpdater:             872ms/5%  
FCLayer_0=sigmoid(l0_weight*X+l0_bias): clnet::type::FullyConnectedLayer:               854ms/5%  
X: clnet::type::Data:           805ms/4%  
linear_regression(FCLayer_1,Y): clnet::back::Loss:              641ms/3%  
Y: clnet::type::Data:           593ms/3%  
data_generator: clnet::InstantTensor:           507ms/3%  
MLPMonitor: clnet::InstantTensor:               455ms/2%  
gradient(FCLayer_0): clnet::back::Gradient:             440ms/2%  
l0_bias: clnet::type::Bias:             397ms/2%  
l0_weight: clnet::type::Weight:                 395ms/2%  
gradient(FCLayer_1): clnet::back::Gradient:             363ms/2%  
FCLayer_1: clnet::type::Output:                 355ms/2%  
l1_bias: clnet::type::Bias:             347ms/2%  
gradient(l0_weight): clnet::back::Gradient:             335ms/2%  
FCLayer_0: clnet::type::Output:                 334ms/2%  
gradient(l0_bias): clnet::back::Gradient:               324ms/2%  
l1_weight: clnet::type::Weight:                 289ms/1%  
gradient(l1_bias): clnet::back::Gradient:               287ms/1%  
gradient(l1_weight): clnet::back::Gradient:             278ms/1%  
</pre>
一旦找到瓶颈，可以通过修改内置的kernels.cl或者修改Tensor.generate_source_code()加载的其他来源的OpenCL源码，实时重载kernels，测试提升效果：  
```
rk
```
<pre>
[debugger] waiting ...
[debugger] kernels reloaded.
</pre>
使用动态执行图，在执行“不等长”的数据如RNN-LSTM上，有性能优势：  
```
.\Release\OpenCLNet.exe charRNN /ds /0 :corpus_file D:\DataSets\charRNN\obama.txt :index_file D:\DataSets\charRNN\obama.index
```  
<pre>
clnet::type::XavierNormalDistributionInitializer                XavierNormalDistributionInitializer
-       clnet::type::Weight             embedding_matrix[84,256]
        clnet::type::Weight             lstm_weight_h0[256,1024]
        clnet::type::Weight             lstm_weight_x0[256,1024]
        clnet::type::Bias               lstm_bias0[1024]
        clnet::type::Weight             lstm_weight_h1[256,1024]
        clnet::type::Weight             lstm_weight_x1[256,1024]
        clnet::type::Bias               lstm_bias1[1024]
        clnet::type::Weight             lstm_weight_h2[256,1024]
        clnet::type::Weight             lstm_weight_x2[256,1024]
        clnet::type::Bias               lstm_bias2[1024]
        clnet::type::Weight             class_weight[256,84]
        clnet::type::Bias               class_bias[84]
clnet::type::IterativeOptimizer         IterativeOptimizer[4]
        SentenceIterator                [8289]
        clnet::type::Data               data[32,129]
        clnet::type::Weight             embedding_matrix[84,256]
        clnet::type::Embedding          Embedding(data)
        clnet::type::Output             embedding[32,129,256]
        clnet::type::LSTMInitializer            lstm_initializer
-               clnet::type::Output             lstm_cell_state0[32,256]
                clnet::type::Output             lstm_hidden0[32,256]
                clnet::type::Output             lstm_cell_state1[32,256]
                clnet::type::Output             lstm_hidden1[32,256]
                clnet::type::Output             lstm_cell_state2[32,256]
                clnet::type::Output             lstm_hidden2[32,256]
        clnet::type::LSTM               LSTM(embedding)
                clnet::type::Weight             lstm_weight_h2[256,1024]
                clnet::type::FullyConnectedLayer                lstm_cell2_FC_hidden=lstm_weight_h2*lstm_hidden2
                clnet::type::Weight             lstm_weight_h1[256,1024]
                clnet::type::FullyConnectedLayer                lstm_cell1_FC_hidden=lstm_weight_h1*lstm_hidden1
                clnet::type::Weight             lstm_weight_h0[256,1024]
                clnet::type::FullyConnectedLayer                lstm_cell0_FC_hidden=lstm_weight_h0*lstm_hidden0
                clnet::type::Output             lstm_input_timestep[32,256]
                clnet::type::Weight             lstm_weight_x0[256,1024]
                clnet::type::Bias               lstm_bias0[1024]
                clnet::type::FullyConnectedLayer                lstm_cell0_FC_input=lstm_weight_x0*lstm_input_timestep+lstm_bias0
                clnet::type::Output             lstm_cell0_FC_input[32,1024]
                clnet::type::BinaryOperator             lstm_cell0_FC_hidden+=lstm_cell0_FC_input
                clnet::type::Output             lstm_cell0_FC_hidden[32,1024]
                clnet::type::LSTMCell           lstm_cell0
                clnet::Tensor           lstm_dropout0_mask[32,256]
                clnet::type::DropOut            lstm_dropout0
                clnet::type::Output             lstm_hidden0[32,256]
                clnet::type::Weight             lstm_weight_x1[256,1024]
                clnet::type::Bias               lstm_bias1[1024]
                clnet::type::FullyConnectedLayer                lstm_cell1_FC_input=lstm_weight_x1*lstm_hidden0+lstm_bias1
                clnet::type::Output             lstm_cell1_FC_input[32,1024]
                clnet::type::BinaryOperator             lstm_cell1_FC_hidden+=lstm_cell1_FC_input
                clnet::type::Output             lstm_cell1_FC_hidden[32,1024]
                clnet::type::LSTMCell           lstm_cell1
                clnet::Tensor           lstm_dropout1_mask[32,256]
                clnet::type::DropOut            lstm_dropout1
                clnet::type::Output             lstm_hidden1[32,256]
                clnet::type::Weight             lstm_weight_x2[256,1024]
                clnet::type::Bias               lstm_bias2[1024]
                clnet::type::FullyConnectedLayer                lstm_cell2_FC_input=lstm_weight_x2*lstm_hidden1+lstm_bias2
                clnet::type::Output             lstm_cell2_FC_input[32,1024]
                clnet::type::BinaryOperator             lstm_cell2_FC_hidden+=lstm_cell2_FC_input
                clnet::type::Output             lstm_cell2_FC_hidden[32,1024]
                clnet::type::LSTMCell           lstm_cell2
                clnet::Tensor           lstm_dropout2_mask[32,256]
                clnet::type::DropOut            lstm_dropout2
                clnet::type::Output             lstm_hidden2[32,256]
-               clnet::Tensor           lstm_runtime_cell_no[3]
        clnet::type::Output             lstm[32,129,256]
        clnet::type::Weight             class_weight[256,84]
        clnet::type::Bias               class_bias[84]
        clnet::type::FullyConnectedLayer                FC=class_weight*lstm+class_bias
        clnet::type::Output             FC[4128,84]
        clnet::type::Data               label[32,129]
        clnet::back::Loss               negative_log_likelihood(softmax(FC),label)
        clnet::back::Gradient           gradient(FC)[4128,84]
        clnet::back::FullyConnectedLayer                back:FC=class_weight*lstm+class_bias
        clnet::back::Gradient           gradient(lstm)[32,129,256]
        clnet::type::LSTMInitializer            LSTM(embedding)_gradient_initializer
-               clnet::back::Gradient           gradient(lstm_cell_state0)[32,256]
                clnet::back::Gradient           gradient(lstm_hidden0)[32,256]
                clnet::back::Gradient           gradient(lstm_cell_state1)[32,256]
                clnet::back::Gradient           gradient(lstm_hidden1)[32,256]
                clnet::back::Gradient           gradient(lstm_cell_state2)[32,256]
                clnet::back::Gradient           gradient(lstm_hidden2)[32,256]
        clnet::back::LSTM               back:LSTM(embedding)
                clnet::back::DropOut            back:lstm_dropout2
                clnet::back::Gradient           gradient(lstm_hidden2)[32,256]
                clnet::back::LSTMCell           back:lstm_cell2
                clnet::back::Gradient           gradient(lstm_cell2_FC_hidden)[32,1024]
                clnet::back::BinaryOperator             back:lstm_cell2_FC_hidden+=lstm_cell2_FC_input
                clnet::back::Gradient           gradient(lstm_cell2_FC_input)[32,1024]
                clnet::back::FullyConnectedLayer                back:lstm_cell2_FC_input=lstm_weight_x2*lstm_hidden1+lstm_bias2
                clnet::back::DropOut            back:lstm_dropout1
                clnet::back::Gradient           gradient(lstm_hidden1)[32,256]
                clnet::back::LSTMCell           back:lstm_cell1
                clnet::back::Gradient           gradient(lstm_cell1_FC_hidden)[32,1024]
                clnet::back::BinaryOperator             back:lstm_cell1_FC_hidden+=lstm_cell1_FC_input
                clnet::back::Gradient           gradient(lstm_cell1_FC_input)[32,1024]
                clnet::back::FullyConnectedLayer                back:lstm_cell1_FC_input=lstm_weight_x1*lstm_hidden0+lstm_bias1
                clnet::back::DropOut            back:lstm_dropout0
                clnet::back::Gradient           gradient(lstm_hidden0)[32,256]
                clnet::back::LSTMCell           back:lstm_cell0
                clnet::back::Gradient           gradient(lstm_cell0_FC_hidden)[32,1024]
                clnet::back::BinaryOperator             back:lstm_cell0_FC_hidden+=lstm_cell0_FC_input
                clnet::back::Gradient           gradient(lstm_cell0_FC_input)[32,1024]
                clnet::back::FullyConnectedLayer                back:lstm_cell0_FC_input=lstm_weight_x0*lstm_input_timestep+lstm_bias0
                clnet::back::FullyConnectedLayer                back:lstm_cell2_FC_hidden=lstm_weight_h2*lstm_hidden2
                clnet::back::Gradient           gradient(lstm_weight_h2)[256,1024]
                clnet::back::FullyConnectedLayer                back:lstm_cell1_FC_hidden=lstm_weight_h1*lstm_hidden1
                clnet::back::Gradient           gradient(lstm_weight_h1)[256,1024]
                clnet::back::FullyConnectedLayer                back:lstm_cell0_FC_hidden=lstm_weight_h0*lstm_hidden0
                clnet::back::Gradient           gradient(lstm_weight_h0)[256,1024]
                clnet::back::Gradient           gradient(lstm_weight_x0)[256,1024]
                clnet::back::Gradient           gradient(lstm_bias0)[1024]
                clnet::back::Gradient           gradient(lstm_weight_x1)[256,1024]
                clnet::back::Gradient           gradient(lstm_bias1)[1024]
                clnet::back::Gradient           gradient(lstm_weight_x2)[256,1024]
                clnet::back::Gradient           gradient(lstm_bias2)[1024]
                clnet::back::Gradient           gradient(lstm_input_timestep)[32,256]
-               clnet::back::Gradient           gradient(embedding)[32,129,256]
                clnet::Tensor           lstm_runtime_cell_no[3]
        clnet::back::Gradient           gradient(embedding)[32,129,256]
        clnet::back::Embedding          back:Embedding(data)
        clnet::back::Gradient           gradient(embedding_matrix)[84,256]
        clnet::back::Gradient           gradient(class_weight)[256,84]
        clnet::back::Gradient           gradient(class_bias)[84]
        clnet::type::StochasticGradientDescentUpdater           SGD
-               clnet::type::Weight             embedding_matrix[84,256]
                clnet::type::Weight             lstm_weight_h0[256,1024]
                clnet::type::Weight             lstm_weight_x0[256,1024]
                clnet::type::Bias               lstm_bias0[1024]
                clnet::type::Weight             lstm_weight_h1[256,1024]
                clnet::type::Weight             lstm_weight_x1[256,1024]
                clnet::type::Bias               lstm_bias1[1024]
                clnet::type::Weight             lstm_weight_h2[256,1024]
                clnet::type::Weight             lstm_weight_x2[256,1024]
                clnet::type::Bias               lstm_bias2[1024]
                clnet::type::Weight             class_weight[256,84]
                clnet::type::Bias               class_bias[84]
-       clnet::InstantTensor            charRNN_monitor

[1,@2018-06-30 16:29:48] GeForce GTX 1050 Ti (kernels build: 297ms)
[debugger] interactive thread started on device 1.
[debugger] device 1 break on IterativeOptimizer: clnet::type::IterativeOptimizer
</pre>

使用0.0002的学习率，标准的SGD更新（无weight decay，无冲量），运行Lenet-5，可以在第一个epoch达到97%的测试集准确率。测试集准确率最高99.19%。  
<pre>
clnet::type::XavierNormalDistributionInitializer                XavierNormalDistributionInitializer
-       clnet::type::Weight             conv1_weight[20,5,5,1]
        clnet::type::Bias               conv1_bias[20]
        clnet::type::Weight             conv2_weight[50,5,5,20]
        clnet::type::Bias               conv2_bias[50]
        clnet::type::Weight             feature_weight[2450,480]
        clnet::type::Bias               feature_bias[2450]
        clnet::type::Weight             inference_weight[480,10]
        clnet::type::Bias               inference_bias[480]
clnet::type::IterativeOptimizer         IterativeOptimizer[4]
        MNISTImageIterator              [60001]
-               clnet::Tensor           train_images[60000,28,28]
                clnet::Tensor           train_labels[60000]
                clnet::Tensor           test_images[10016,28,28]
                clnet::Tensor           test_labels[10016]
        clnet::Tensor           train_images_data[32,28,28,1]
        clnet::type::Weight             conv1_weight[20,5,5,1]
        clnet::type::Bias               conv1_bias[20]
        clnet::type::ConvolutionKernel          conv1=Convolution:5x5(train_images_data,tanh)
        clnet::type::Output             conv1[32,28,28,20]
        clnet::type::Pooling            pool1=Pooling(conv1,max)
        clnet::type::Output             pool1[32,14,14,20]
        clnet::type::Weight             conv2_weight[50,5,5,20]
        clnet::type::Bias               conv2_bias[50]
        clnet::type::ConvolutionKernel          conv2=Convolution:5x5(pool1,tanh)
        clnet::type::Output             conv2[32,14,14,50]
        clnet::type::Pooling            pool2=Pooling(conv2,max)
        clnet::type::Output             pool2[32,7,7,50]
        clnet::type::Reshape            reshape[32,2450]
        clnet::type::Weight             feature_weight[2450,480]
        clnet::type::Bias               feature_bias[2450]
        clnet::type::FullyConnectedLayer                feature=tanh(feature_weight*reshape+feature_bias)
        clnet::type::Output             feature[32,480]
        clnet::type::Weight             inference_weight[480,10]
        clnet::type::Bias               inference_bias[480]
        clnet::type::FullyConnectedLayer                inference=inference_weight*feature+inference_bias
        clnet::type::Output             inference[32,10]
        clnet::Tensor           train_images_label[32]
        clnet::back::Loss               negative_log_likelihood(softmax(inference),train_images_label)
        clnet::back::Gradient           gradient(inference)[32,10]
        clnet::back::FullyConnectedLayer                back:inference=inference_weight*feature+inference_bias
        clnet::back::Gradient           gradient(feature)[32,480]
        clnet::back::FullyConnectedLayer                back:feature=tanh(feature_weight*reshape+feature_bias)
        clnet::back::Reshape            gradient(reshape)[32,2450]
        clnet::back::Gradient           gradient(pool2)[32,7,7,50]
        clnet::back::Pooling            back:pool2=Pooling(conv2,max)
        clnet::back::Gradient           gradient(conv2)[32,14,14,50]
        clnet::back::ConvolutionKernel          back:conv2=Convolution:5x5(pool1,tanh)
        clnet::back::Gradient           gradient(pool1)[32,14,14,20]
        clnet::back::Pooling            back:pool1=Pooling(conv1,max)
        clnet::back::Gradient           gradient(conv1)[32,28,28,20]
        clnet::back::ConvolutionKernel          back:conv1=Convolution:5x5(train_images_data,tanh)
        clnet::back::Gradient           gradient(conv1_weight)[20,5,5,1]
        clnet::back::Gradient           gradient(conv1_bias)[20]
        clnet::back::Gradient           gradient(conv2_weight)[50,5,5,20]
        clnet::back::Gradient           gradient(conv2_bias)[50]
        clnet::back::Gradient           gradient(feature_weight)[2450,480]
        clnet::back::Gradient           gradient(feature_bias)[2450]
        clnet::back::Gradient           gradient(inference_weight)[480,10]
        clnet::back::Gradient           gradient(inference_bias)[480]
        clnet::type::StochasticGradientDescentUpdater           SGD
-               clnet::type::Weight             conv1_weight[20,5,5,1]
                clnet::type::Bias               conv1_bias[20]
                clnet::type::Weight             conv2_weight[50,5,5,20]
                clnet::type::Bias               conv2_bias[50]
                clnet::type::Weight             feature_weight[2450,480]
                clnet::type::Bias               feature_bias[2450]
                clnet::type::Weight             inference_weight[480,10]
                clnet::type::Bias               inference_bias[480]
-       clnet::InstantTensor            MNIST_CNN_monitor
        clnet::InstantTensor            MNIST_CNN_validator
                MNISTImageIterator              [60001]
-                       clnet::Tensor           train_images[60000,28,28]
                        clnet::Tensor           train_labels[60000]
                        clnet::Tensor           test_images[10016,28,28]
                        clnet::Tensor           test_labels[10016]
                clnet::Tensor           train_images_data[32,28,28,1]
                clnet::type::Weight             conv1_weight[20,5,5,1]
                clnet::type::Bias               conv1_bias[20]
                clnet::type::ConvolutionKernel          conv1=Convolution:5x5(train_images_data,tanh)
                clnet::type::Output             conv1[32,28,28,20]
                clnet::type::Pooling            pool1=Pooling(conv1,max)
                clnet::type::Output             pool1[32,14,14,20]
                clnet::type::Weight             conv2_weight[50,5,5,20]
                clnet::type::Bias               conv2_bias[50]
                clnet::type::ConvolutionKernel          conv2=Convolution:5x5(pool1,tanh)
                clnet::type::Output             conv2[32,14,14,50]
                clnet::type::Pooling            pool2=Pooling(conv2,max)
                clnet::type::Output             pool2[32,7,7,50]
                clnet::type::Reshape            reshape[32,2450]
                clnet::type::Weight             feature_weight[2450,480]
                clnet::type::Bias               feature_bias[2450]
                clnet::type::FullyConnectedLayer                feature=tanh(feature_weight*reshape+feature_bias)
                clnet::type::Output             feature[32,480]
                clnet::type::Weight             inference_weight[480,10]
                clnet::type::Bias               inference_bias[480]
                clnet::type::FullyConnectedLayer                inference=inference_weight*feature+inference_bias
                clnet::type::Output             inference[32,10]

[0,@2018-06-30 20:12:44] GeForce GTX 1080 Ti (kernels build: 190ms)
[debugger] interactive thread started on device 0.
[0,0,28153ms] train accuracy: 0.976492  test set accuracy: 97.08%
[0,1,1999.400146/s] train accuracy: 0.961088    test set accuracy: 97.79%
[0,2,1977.066040/s] train accuracy: 0.969694    test set accuracy: 98.26%
[0,3,1977.131226/s] train accuracy: 0.975035    test set accuracy: 98.5%
[0,4,1975.308594/s] train accuracy: 0.996367    test set accuracy: 98.62%
[0,5,1974.333618/s] train accuracy: 0.836167    test set accuracy: 98.43%
[0,6,1973.879028/s] train accuracy: 0.885966    test set accuracy: 98.78%
[0,7,1973.359619/s] train accuracy: 0.87724     test set accuracy: 98.66%
[0,8,1973.943970/s] train accuracy: 0.994898    test set accuracy: 98.71%
[0,9,1973.489502/s] train accuracy: 0.975384    test set accuracy: 98.73%
[0,10,1973.554321/s] train accuracy: 0.987039   test set accuracy: 98.87%
[0,11,1973.424561/s] train accuracy: 0.989917   test set accuracy: 98.8%
[0,12,1969.925781/s] train accuracy: 0.971296   test set accuracy: 98.82%
[0,13,1965.280029/s] train accuracy: 0.97529    test set accuracy: 98.98%
[0,14,1965.280029/s] train accuracy: 0.996434   test set accuracy: 99.01%
[0,15,1965.215698/s] train accuracy: 0.989955   test set accuracy: 98.93%
[0,16,1965.022583/s] train accuracy: 0.994556   test set accuracy: 99.05%
[0,17,1965.151367/s] train accuracy: 0.995275   test set accuracy: 99.03%
[0,18,1965.022583/s] train accuracy: 0.977169   test set accuracy: 98.96%
[0,19,1964.765259/s] train accuracy: 0.992683   test set accuracy: 99.05%
[0,20,1965.086914/s] train accuracy: 0.994362   test set accuracy: 98.93%
[0,21,1964.700928/s] train accuracy: 0.959518   test set accuracy: 98.96%
[0,22,1964.893921/s] train accuracy: 0.993861   test set accuracy: 98.92%
[0,23,1965.151367/s] train accuracy: 0.970596   test set accuracy: 99.06%
[0,24,1964.958252/s] train accuracy: 0.981751   test set accuracy: 98.95%
[0,25,1964.636597/s] train accuracy: 0.997321   test set accuracy: 99%
[0,26,1964.829590/s] train accuracy: 0.995127   test set accuracy: 99.11%
[0,27,1964.572266/s] train accuracy: 0.998763   test set accuracy: 99.15%
</pre>