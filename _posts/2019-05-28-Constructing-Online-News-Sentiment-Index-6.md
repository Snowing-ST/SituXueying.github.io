---
title: Constructing Online News Sentiment Index 6
author: Situ
layout: post
categories: [big data]
tags: [deep learning, text classification]
---

<font face="仿宋" >基于深度学习文本分类的网络新闻情感指数编制（六）<br>深度学习文本分类模型之用CNN RNN做文本分类</font>
<style>
    body {font-family: "华文中宋"}
</style>

## Main step 6:<center>text classification with deep learning (CNN RNN)</center>
### description:
- classification model ：CNN、RNN
- best model ：CNN+word2vec, accuracy:84%
- reference:
    - [Kim Y. Convolutional Neural Networks for Sentence Classification[J]. Eprint Arxiv, 2014.](https://arxiv.org/abs/1408.5882)
    - [Convolutional Neural Network for Text Classification in Tensorflow](https://github.com/dennybritz/cnn-text-classification-tf)
    - [Implementing a CNN for Text Classification in TensorFlow](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)
    - [Ye Zhang, Byron Wallace. A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification[J]. Eprint Arxiv, 2015.](https://arxiv.org/abs/1510.03820v3)
    - [CNN-RNN中文文本分类，基于TensorFlow](https://github.com/gaussic/text-classification-cnn-rnn)


### code explanation:
#### 1. transform news data into 3D word2vec and generates a batch iterator

- please see [w2v_CNN_data_helpers.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/6%20word2vec%2BSVM_CNN/w2v_CNN_data_helpers.py)

#### 2. CNN text classification model

- please see [w2v_CNN.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/6%20word2vec%2BSVM_CNN/w2v_CNN.py)

#### 3. train the CNN by using word2vec 3D matrix
- input:```train.csv```
- output: model training process saved in ```.\runs\1546591413\summaries```
- please see [w2v_CNN_train.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/6%20word2vec%2BSVM_CNN/w2v_CNN_train.py)

#### 4. visualize the training process using tensorboard
- tensorboard 注意路径不能有中文！！！
- 用anaconda prompt打开，并切换到summaries的文件下，如：
```
cd .\runs\1546591413\summaries
tensorboard --logdir=run1:“train”,run2:"dev"
```
- 用浏览器打开 http://your_computer_name:6006/
- 每一次改动都要重启prompt

#### 5. tag unlabel ```test.csv``` with CNN model
- input: ```test.csv```,final model saved in ```code/runs/1546591413/checkpoints```
- output: ```data_test.csv```, a test data with predicted labels
- please see [w2v_CNN_eval.py](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/blob/master/6%20word2vec%2BSVM_CNN/w2v_CNN_eval.py)

#### 6. text classification with RNN
- please see [RNN.py](),[RNN_train.py](),which is different from CNN

For more information about this project, please visit my [github](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index).