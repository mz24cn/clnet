
# clNET: **OpenCL for Nets**
A Deep Learning Framework based on OpenCL, written by C++. Supports popular MLP, RNN(LSTM), CNN(ResNet) neural networks. 
基于OpenCL的深度学习计算框架，C++开发，支持多层感知器，长短时记忆模型，卷积神经网络。

Progress: Currently clnet can successfully run fully connected neural networks (MLP), CharRNN (LSTM) which uses dynamic computing graph to deal with loops, CNN (LeNet5 on MNIST dataset), WRN (Wide Residual Networks, CIFAR).  
Tested on Nvidia GTX1080, AMD R9 295X2, Intel HD Graphics 630 GPU.  
Support multiple devices training.

已完成进度：
可成功运行MLP全连接多层感知器，CharRNN（LSTM，基于动态计算图的循环实现），CNN（LeNet5，MNIST），WRN（CIFAR）的训练及推断。  
三种模型均在Nvidia GTX1080, AMD R9 295X2, Intel HD Graphics 630以及Intel CPU，AMD CPU/APU上测试通过。  
测试通过的编译环境：  
Windows 10，MSVS2015；  
Linux CentOS 7，g++ 4.8.5，Makefile；  
eclipse/CDT，MinGW64 6；  
eclipse/CDT，CrossGCC: CodeBench Lite 2014/05（gcc 4.8.3）；  
支持多显卡训练。
TODO list：Kernel性能尚待进一步优化，分布式有待开发。

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
MNIST CNN（训练及预测命令行。预测图片需使用28*28大小的24位BMP格式，黑底白字）：  
```
.\Release\OpenCLNet.exe MNIST_CNN /ds :mnist_folder D:\DataSets\MNIST\
.\Release\OpenCLNet.exe MNIST_CNN /p :params_file D:\DataSets\MNIST_CNN.clnetparams :file D:\9.bmp
```    
D:/DataSets/下需包含MNIST数据集文件train-images.idx3-ubyte，train-labels.idx1-ubyte，t10k-images.idx3-ubyte，t10k-labels.idx1-ubyte。可从[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)下载。目录名末尾请加上路径分隔符。

如何调试：  
“/ds”生成执行树，“/ss”执行到第一个Tensor，停留，等待交互命令：  
```
.\Release\OpenCLNet.exe MLP /ss /ds /0  
```
<pre>
clnet::type::GeneralInitializer                GeneralInitializer
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
clnet::type::GeneralInitializer                GeneralInitializer
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
clnet::type::GeneralInitializer                GeneralInitializer
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
        clnet::type::ConvolutionLayer          conv1=Convolution:5x5(train_images_data,tanh)
        clnet::type::Output             conv1[32,28,28,20]
        clnet::type::Pooling            pool1=Pooling(conv1,max)
        clnet::type::Output             pool1[32,14,14,20]
        clnet::type::Weight             conv2_weight[50,5,5,20]
        clnet::type::Bias               conv2_bias[50]
        clnet::type::ConvolutionLayer          conv2=Convolution:5x5(pool1,tanh)
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
        clnet::back::ConvolutionLayer          back:conv2=Convolution:5x5(pool1,tanh)
        clnet::back::Gradient           gradient(pool1)[32,14,14,20]
        clnet::back::Pooling            back:pool1=Pooling(conv1,max)
        clnet::back::Gradient           gradient(conv1)[32,28,28,20]
        clnet::back::ConvolutionLayer          back:conv1=Convolution:5x5(train_images_data,tanh)
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
                clnet::type::ConvolutionLayer          conv1=Convolution:5x5(train_images_data,tanh)
                clnet::type::Output             conv1[32,28,28,20]
                clnet::type::Pooling            pool1=Pooling(conv1,max)
                clnet::type::Output             pool1[32,14,14,20]
                clnet::type::Weight             conv2_weight[50,5,5,20]
                clnet::type::Bias               conv2_bias[50]
                clnet::type::ConvolutionLayer          conv2=Convolution:5x5(pool1,tanh)
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

支持宽残差网络Wide residual networks ([http://arxiv.org/abs/1605.07146](http://arxiv.org/abs/1605.07146))。命令行中“/dso”生成仅包含运算Tensor的执行树，“/pp”打印参数清单（按大小倒序）：
```
./build/OpenCLNet CIFAR_WRN :cifar_folder /cifar-10-batches-bin/ :width 1 :N 1 :batch_size 64 /dso /pp /1
```  
<pre>
clnet::type::GeneralInitializer         GeneralInitializer
clnet::type::IterativeOptimizer         IterativeOptimizer[4]
        CIFARImageIterator              [50049]
-               clnet::Tensor           train_images[50048,32,32,3]
                clnet::Tensor           train_labels[50048,1]
                clnet::Tensor           test_images[10048,32,32,3]
                clnet::Tensor           test_labels[10048,1]
        clnet::Tensor           train_images_data[64,32,32,3]
        clnet::type::ConvolutionLayer           conv0=Convolution:3x3(train_images_data)
        clnet::type::BatchNormalizedLayer               group0_block0_bn0=group0_block0_bn0_gamma*normalize(conv0)+group0_block0_bn0_beta
        clnet::type::Activation         ReLU(group0_block0_bn0)
        clnet::type::ConvolutionLayer           group0_block0_conv0=Convolution:3x3(relu(group0_block0_bn0))
        clnet::type::BatchNormalizedLayer               group0_block0_bn1=group0_block0_bn1_gamma*normalize(group0_block0_conv0)+group0_block0_bn1_beta
        clnet::type::Activation         ReLU(group0_block0_bn1)
        clnet::type::ConvolutionLayer           group0_block0_conv1=Convolution:3x3(relu(group0_block0_bn1))
        clnet::type::BinaryOperator             group0_block0_conv1+conv0
        clnet::type::BatchNormalizedLayer               group1_block0_bn0=group1_block0_bn0_gamma*normalize((group0_block0_conv1+conv0))+group1_block0_bn0_beta
        clnet::type::Activation         ReLU(group1_block0_bn0)
        clnet::type::ConvolutionLayer           group1_block0_conv0=Convolution:3x3(relu(group1_block0_bn0))
        clnet::type::BatchNormalizedLayer               group1_block0_bn1=group1_block0_bn1_gamma*normalize(group1_block0_conv0)+group1_block0_bn1_beta
        clnet::type::Activation         ReLU(group1_block0_bn1)
        clnet::type::ConvolutionLayer           group1_block0_conv1=Convolution:3x3(relu(group1_block0_bn1))
        clnet::type::ConvolutionLayer           group1_block0_convdim=Convolution:1x1(relu(group1_block0_bn0))
        clnet::type::BinaryOperator             group1_block0_conv1+group1_block0_convdim
        clnet::type::BatchNormalizedLayer               group2_block0_bn0=group2_block0_bn0_gamma*normalize((group1_block0_conv1+group1_block0_convdim))+group2_block0_bn0_beta
        clnet::type::Activation         ReLU(group2_block0_bn0)
        clnet::type::ConvolutionLayer           group2_block0_conv0=Convolution:3x3(relu(group2_block0_bn0))
        clnet::type::BatchNormalizedLayer               group2_block0_bn1=group2_block0_bn1_gamma*normalize(group2_block0_conv0)+group2_block0_bn1_beta
        clnet::type::Activation         ReLU(group2_block0_bn1)
        clnet::type::ConvolutionLayer           group2_block0_conv1=Convolution:3x3(relu(group2_block0_bn1))
        clnet::type::ConvolutionLayer           group2_block0_convdim=Convolution:1x1(relu(group2_block0_bn0))
        clnet::type::BinaryOperator             group2_block0_conv1+group2_block0_convdim
        clnet::type::BatchNormalizedLayer               bn=bn_gamma*normalize((group2_block0_conv1+group2_block0_convdim))+bn_beta
        clnet::type::Activation         ReLU(bn)
        clnet::type::Pooling            pool=Pooling(relu(bn),average)
        clnet::type::Reshape            reshape[64,64]
        clnet::type::FullyConnectedLayer                inference=inference_weight*reshape+inference_bias
        clnet::Tensor           train_images_label[64]
        clnet::back::Loss               negative_log_likelihood(softmax(inference),train_images_label)
        clnet::back::FullyConnectedLayer                back:inference=inference_weight*reshape+inference_bias
        clnet::back::Reshape            gradient(reshape)[64,64]
        clnet::back::Pooling            back:pool=Pooling(relu(bn),average)
        clnet::back::Activation         gradient(ReLU(bn))
        clnet::back::BatchNormalizedLayer               back:bn=bn_gamma*normalize((group2_block0_conv1+group2_block0_convdim))+bn_beta
        clnet::back::BinaryOperator             back:group2_block0_conv1+group2_block0_convdim
        clnet::back::ConvolutionLayer           back:group2_block0_convdim=Convolution:1x1(relu(group2_block0_bn0))
        clnet::back::ConvolutionLayer           back:group2_block0_conv1=Convolution:3x3(relu(group2_block0_bn1))
        clnet::back::Activation         gradient(ReLU(group2_block0_bn1))
        clnet::back::BatchNormalizedLayer               back:group2_block0_bn1=group2_block0_bn1_gamma*normalize(group2_block0_conv0)+group2_block0_bn1_beta
        clnet::back::ConvolutionLayer           back:group2_block0_conv0=Convolution:3x3(relu(group2_block0_bn0))
        clnet::back::Activation         gradient(ReLU(group2_block0_bn0))
        clnet::back::BatchNormalizedLayer               back:group2_block0_bn0=group2_block0_bn0_gamma*normalize((group1_block0_conv1+group1_block0_convdim))+group2_block0_bn0_beta
        clnet::back::BinaryOperator             back:group1_block0_conv1+group1_block0_convdim
        clnet::back::ConvolutionLayer           back:group1_block0_convdim=Convolution:1x1(relu(group1_block0_bn0))
        clnet::back::ConvolutionLayer           back:group1_block0_conv1=Convolution:3x3(relu(group1_block0_bn1))
        clnet::back::Activation         gradient(ReLU(group1_block0_bn1))
        clnet::back::BatchNormalizedLayer               back:group1_block0_bn1=group1_block0_bn1_gamma*normalize(group1_block0_conv0)+group1_block0_bn1_beta
        clnet::back::ConvolutionLayer           back:group1_block0_conv0=Convolution:3x3(relu(group1_block0_bn0))
        clnet::back::Activation         gradient(ReLU(group1_block0_bn0))
        clnet::back::BatchNormalizedLayer               back:group1_block0_bn0=group1_block0_bn0_gamma*normalize((group0_block0_conv1+conv0))+group1_block0_bn0_beta
        clnet::back::BinaryOperator             back:group0_block0_conv1+conv0
        clnet::back::ConvolutionLayer           back:group0_block0_conv1=Convolution:3x3(relu(group0_block0_bn1))
        clnet::back::Activation         gradient(ReLU(group0_block0_bn1))
        clnet::back::BatchNormalizedLayer               back:group0_block0_bn1=group0_block0_bn1_gamma*normalize(group0_block0_conv0)+group0_block0_bn1_beta
        clnet::back::ConvolutionLayer           back:group0_block0_conv0=Convolution:3x3(relu(group0_block0_bn0))
        clnet::back::Activation         gradient(ReLU(group0_block0_bn0))
        clnet::back::BatchNormalizedLayer               back:group0_block0_bn0=group0_block0_bn0_gamma*normalize(conv0)+group0_block0_bn0_beta
        clnet::back::ConvolutionLayer           back:conv0=Convolution:3x3(train_images_data)
        clnet::type::StochasticGradientDescentUpdater           SGD
-       clnet::InstantTensor            CIFAR_WRN_monitor
        clnet::InstantTensor            CIFAR_WRN_validator
                CIFARImageIterator              [50049]
-                       clnet::Tensor           train_images[50048,32,32,3]
                        clnet::Tensor           train_labels[50048,1]
                        clnet::Tensor           test_images[10048,32,32,3]
                        clnet::Tensor           test_labels[10048,1]
                clnet::Tensor           train_images_data[64,32,32,3]
                clnet::type::ConvolutionLayer           conv0=Convolution:3x3(train_images_data)
                clnet::type::BatchNormalizedLayer               group0_block0_bn0=group0_block0_bn0_gamma*normalize(conv0)+group0_block0_bn0_beta
                clnet::type::Activation         ReLU(group0_block0_bn0)
                clnet::type::ConvolutionLayer           group0_block0_conv0=Convolution:3x3(relu(group0_block0_bn0))
                clnet::type::BatchNormalizedLayer               group0_block0_bn1=group0_block0_bn1_gamma*normalize(group0_block0_conv0)+group0_block0_bn1_beta
                clnet::type::Activation         ReLU(group0_block0_bn1)
                clnet::type::ConvolutionLayer           group0_block0_conv1=Convolution:3x3(relu(group0_block0_bn1))
                clnet::type::BinaryOperator             group0_block0_conv1+conv0
                clnet::type::BatchNormalizedLayer               group1_block0_bn0=group1_block0_bn0_gamma*normalize((group0_block0_conv1+conv0))+group1_block0_bn0_beta
                clnet::type::Activation         ReLU(group1_block0_bn0)
                clnet::type::ConvolutionLayer           group1_block0_conv0=Convolution:3x3(relu(group1_block0_bn0))
                clnet::type::BatchNormalizedLayer               group1_block0_bn1=group1_block0_bn1_gamma*normalize(group1_block0_conv0)+group1_block0_bn1_beta
                clnet::type::Activation         ReLU(group1_block0_bn1)
                clnet::type::ConvolutionLayer           group1_block0_conv1=Convolution:3x3(relu(group1_block0_bn1))
                clnet::type::ConvolutionLayer           group1_block0_convdim=Convolution:1x1(relu(group1_block0_bn0))
                clnet::type::BinaryOperator             group1_block0_conv1+group1_block0_convdim
                clnet::type::BatchNormalizedLayer               group2_block0_bn0=group2_block0_bn0_gamma*normalize((group1_block0_conv1+group1_block0_convdim))+group2_block0_bn0_beta
                clnet::type::Activation         ReLU(group2_block0_bn0)
                clnet::type::ConvolutionLayer           group2_block0_conv0=Convolution:3x3(relu(group2_block0_bn0))
                clnet::type::BatchNormalizedLayer               group2_block0_bn1=group2_block0_bn1_gamma*normalize(group2_block0_conv0)+group2_block0_bn1_beta
                clnet::type::Activation         ReLU(group2_block0_bn1)
                clnet::type::ConvolutionLayer           group2_block0_conv1=Convolution:3x3(relu(group2_block0_bn1))
                clnet::type::ConvolutionLayer           group2_block0_convdim=Convolution:1x1(relu(group2_block0_bn0))
                clnet::type::BinaryOperator             group2_block0_conv1+group2_block0_convdim
                clnet::type::BatchNormalizedLayer               bn=bn_gamma*normalize((group2_block0_conv1+group2_block0_convdim))+bn_beta
                clnet::type::Activation         ReLU(bn)
                clnet::type::Pooling            pool=Pooling(relu(bn),average)
                clnet::type::Reshape            reshape[64,64]
                clnet::type::FullyConnectedLayer                inference=inference_weight*reshape+inference_bias

Total number of parameters: 78,330, trainable: 77,850
group2_block0_conv1_weight[64,3,3,64]: clnet::type::Weight      36,864
group2_block0_conv0_weight[64,3,3,32]: clnet::type::Weight      18,432
group1_block0_conv1_weight[32,3,3,32]: clnet::type::Weight      9,216
group1_block0_conv0_weight[32,3,3,16]: clnet::type::Weight      4,608
group0_block0_conv0_weight[16,3,3,16]: clnet::type::Weight      2,304
group0_block0_conv1_weight[16,3,3,16]: clnet::type::Weight      2,304
group2_block0_convdim_weight[64,1,1,32]: clnet::type::Weight    2,048
inference_weight[64,10]: clnet::type::Weight    640
group1_block0_convdim_weight[32,1,1,16]: clnet::type::Weight    512
conv0_weight[16,3,3,3]: clnet::type::Weight     432
bn_beta[64]: clnet::type::Bias  64
bn_gamma[64]: clnet::type::Weight       64
bn_moving_mean[64]: clnet::type::Parameter      64      -
bn_moving_variance[64]: clnet::type::Parameter  64      -
group2_block0_bn1_beta[64]: clnet::type::Bias   64
group2_block0_bn1_gamma[64]: clnet::type::Weight        64
group2_block0_bn1_moving_mean[64]: clnet::type::Parameter       64      -
group2_block0_bn1_moving_variance[64]: clnet::type::Parameter   64      -
group1_block0_bn1_beta[32]: clnet::type::Bias   32
group1_block0_bn1_gamma[32]: clnet::type::Weight        32
group1_block0_bn1_moving_mean[32]: clnet::type::Parameter       32      -
group1_block0_bn1_moving_variance[32]: clnet::type::Parameter   32      -
group2_block0_bn0_beta[32]: clnet::type::Bias   32
group2_block0_bn0_gamma[32]: clnet::type::Weight        32
group2_block0_bn0_moving_mean[32]: clnet::type::Parameter       32      -
group2_block0_bn0_moving_variance[32]: clnet::type::Parameter   32      -
group0_block0_bn0_beta[16]: clnet::type::Bias   16
group0_block0_bn0_gamma[16]: clnet::type::Weight        16
group0_block0_bn0_moving_mean[16]: clnet::type::Parameter       16      -
group0_block0_bn0_moving_variance[16]: clnet::type::Parameter   16      -
group0_block0_bn1_beta[16]: clnet::type::Bias   16
group0_block0_bn1_gamma[16]: clnet::type::Weight        16
group0_block0_bn1_moving_mean[16]: clnet::type::Parameter       16      -
group0_block0_bn1_moving_variance[16]: clnet::type::Parameter   16      -
group1_block0_bn0_beta[16]: clnet::type::Bias   16
group1_block0_bn0_gamma[16]: clnet::type::Weight        16
group1_block0_bn0_moving_mean[16]: clnet::type::Parameter       16      -
group1_block0_bn0_moving_variance[16]: clnet::type::Parameter   16      -
inference_bias[10]: clnet::type::Bias   10
[0,@2019-05-06 23:02:55] GeForce GTX 1080 Ti (kernels build: 1s.84ms)
[debugger] interactive thread started on device 0.
[0,0,58902ms] train loss: 1.16239       test set accuracy: 57.65%
[0,1,777.589600/s] train loss: 0.818452 test set accuracy: 65.6%
[0,2,772.429138/s] train loss: 0.941452 test set accuracy: 65.57%
[0,3,772.381409/s] train loss: 0.914646 test set accuracy: 70.4%
[0,4,772.333740/s] train loss: 0.890248 test set accuracy: 71.93%
[0,5,772.286072/s] train loss: 0.868742 test set accuracy: 72.64%
[0,6,772.119263/s] train loss: 0.784145 test set accuracy: 73.25%
[0,7,771.952576/s] train loss: 0.50787  test set accuracy: 74.12%
[0,8,771.964478/s] train loss: 0.679059 test set accuracy: 75%
[0,9,771.964478/s] train loss: 0.619829 test set accuracy: 74.54%
[0,10,772.000183/s] train loss: 0.596677        test set accuracy: 76.02%
[0,11,772.000183/s] train loss: 0.847572        test set accuracy: 75.81%
[0,12,771.904907/s] train loss: 0.802736        test set accuracy: 75.48%
[0,13,772.000183/s] train loss: 0.7105  test set accuracy: 77.84%
[0,14,771.857300/s] train loss: 0.682924        test set accuracy: 77%
[0,15,771.833496/s] train loss: 0.487696        test set accuracy: 76.49%
[0,16,771.893005/s] train loss: 0.54608 test set accuracy: 77.13%
[0,17,771.797791/s] train loss: 0.651461        test set accuracy: 76.69%
[0,18,771.773987/s] train loss: 0.709687        test set accuracy: 76.91%
[0,19,771.940674/s] train loss: 0.57213 test set accuracy: 77.01%
[0,20,731.054626/s] train loss: 0.383843        test set accuracy: 78.44%
[0,21,771.643127/s] train loss: 0.45581 test set accuracy: 78.72%
[0,22,771.583618/s] train loss: 0.498909        test set accuracy: 77.12%
[0,23,763.951660/s] train loss: 0.493421        test set accuracy: 77.9%
[0,24,763.765137/s] train loss: 0.554525        test set accuracy: 77.41%
[0,25,763.776733/s] train loss: 0.500107        test set accuracy: 77.98%
[0,26,759.822693/s] train loss: 0.635151        test set accuracy: 78.26%
[0,27,759.476746/s] train loss: 0.598002        test set accuracy: 78.51%
[0,28,759.407593/s] train loss: 0.559696        test set accuracy: 76.91%
[0,29,759.361511/s] train loss: 0.80741 test set accuracy: 78.15%
[0,30,758.970032/s] train loss: 0.607492        test set accuracy: 78.55%
SGD.learning_rate = 0.005
[debugger] SGD.learning_rate = 0.064
[debugger] SGD.learning_rate = 0.005
[0,31,758.567383/s] train loss: 0.320696        test set accuracy: 82.93%
[0,32,758.555908/s] train loss: 0.218599        test set accuracy: 82.99%
</pre>