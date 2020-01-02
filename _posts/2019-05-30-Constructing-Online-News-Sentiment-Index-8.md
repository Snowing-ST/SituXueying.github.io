---
title: 网络新闻情感指数的应用
author: Situ
layout: post
categories: [big data]
tags: [时间序列,文本分类,NLP,deep-learning,R语言,Master Thesis]
---

<font face="仿宋" >基于深度学习文本分类的网络新闻情感指数编制（七）<br>网络新闻情感指数的应用</font>

## Main step 8:<center> validation analysis of Online News Sentiment Index </center> 
### description:
- compare the sub-indexes with the macroeconomic indicators
- compare the Online News Sentiment Index with traditional consumer confidence indexes, Consumer Confidence Index(CCI) released by National Bureau of Statistics and China Consumer Confidence Index(CCCI) released by academic institutions

<img src="{{ 'assets/images/post_images/validation_analysis1.jpg'| relative_url }}" /> 

|相关系数	|CCI|	CCCI|
|---|---|---|
|网络新闻情感指数|	0.5930|	0.8634|

<img src="{{ 'assets/images/post_images/validation_analysis.jpg'| relative_url }}" /> 

## Main step 9:<center> explore the relationship between the Online News Sentiment Index and CCCI </center> 

### description:
- use Online News Sentiment Index to predict CCCI
- use six sub-indexes to predict CCCI
- Time Series Analysis method: co-integration, regression, ARIMAX, VAR, VARX

### code explanation:
- [Time_Series_Analysis.R](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index/tree/master/8%20Time_Series_Analysis.R)

## <center>Research Results</center>

The study shows that the correlation between the Online News Sentiment Index and the China Consumer Confidence Index (CCCI) is as high as 0.86, and has a certain leading effect. The correlation between the fitted index and CCCI is increased to 0.94. The index shows obvious similarity, preemptiveness or complementarity to relevant economic macro indicators. The above results reflect the effectiveness of the Online News Sentiment Index, indicating that online public opinion imposes a certain impact on consumer confidence, and consumer confidence changes can be reflected in news texts. At the same time, the results also show that the time-consuming and costly questionnaire method can be substituted by mining the emotional tendency of online news in a timely and automatic way through computer programs.
