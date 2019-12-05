---
title: 招聘信息文本聚类
author: Situ
layout: post
categories: [big data]
tags: [文本聚类]
---

使用文本聚类的方式提取每条实习信息中描述专业技能的句子


# 《基于文本聚类的招聘信息中技能要求提取与量化》

- 上篇 [实习僧爬虫与招聘信息文本预处理](Extracting-Skill-Imformation-from-recruitment-ADs-with-text-clustering-1.html)
- <font color="pink" >中篇 招聘信息文本聚类</font>
- 下篇 [量化薪资与技能的关系](Extracting-Skill-Imformation-from-recruitment-ADs-with-text-clustering-3.html)

## 文本聚类
本研究中文本向量化采用tf-idf，用稀疏方式储存词-文档矩阵。矩阵维度为t\*n，t代表句子个数，n代表词语个数。 本文预处理后的词汇有1793个，句子2817条，提取1000个tf-idf特征，得到 2817*1000的文档词频矩阵。下面将用三种聚类方法对“职位描述”中的句子进行聚类，根据聚类结果的解释性选择聚类数目。

#### 1.Kmeans聚类

Kmeans是一种基于相似度的聚类方法。在聚类之前，需要用户显式地定义一个相似度函数。聚类算法根据相似度的计算结果将相似的文本分在同一个组。在这种聚类模式下，每个文本只能属于一个组,这种聚类方法也叫“硬聚类”。 K-Means方法是MacQueen1967年提出的，原理是给定一个数据集合X和一个整数K（K\<n），K-Means方法将X分成K个聚类并使得在每个聚类中所有值与该聚类中心距离的总和最小。


```python
#kmeans tfidf 文本聚类---------------------------------------------------
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
from sklearn.cluster import KMeans,MiniBatchKMeans
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time



os.chdir("/Users/situ/Documents/EDA/final")
#os.chdir("E:/graduate/class/EDA/final")


def loadDataset(myfile):
    '''导入文本数据集'''
    f = open(myfile,'r',encoding = "utf-8")
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip())

    f.close()
    return dataset

def transform(dataset,n_features=1000):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2,use_idf=True)
    X = vectorizer.fit_transform(dataset)
    return X,vectorizer

def knn_train(X,vectorizer,true_k=10,minibatch = False,showLable = False):
    #使用采样数据还是原始数据训练k-means，    
    if minibatch:#数据多时用，如大于1万条样本
        km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
                             init_size=1000, batch_size=1000, verbose=False)
    else:
        km = KMeans(n_clusters=true_k, init='k-means++', max_iter=300, n_init=1,
                    verbose=False,random_state=1994)
    km.fit(X)    
    if showLable:
        print("Top terms per cluster:")
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print (vectorizer.get_stop_words())
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :20]:
                print(' %s' % terms[ind], end='')
            print()
    result = list(km.predict(X))
    print ('Cluster distribution:')
    print (dict([(i, result.count(i)) for i in result]))
    return km
    
def test():
    '''测试选择最优参数'''
    dataset = loadDataset('clean_text.txt')    
    print("%d documents" % len(dataset))
    X,vectorizer = transform(dataset,n_features=500)
    true_ks = []
    scores = []
    for i in range(3,15,1):        
        score = -knn_train(X,vectorizer,true_k=i).score(X)/len(dataset)
        print (i,score)
        true_ks.append(i)
        scores.append(score)
    plt.figure(figsize=(8,4))
    plt.plot(true_ks,scores,label="score",color="red",linewidth=1)
    plt.xlabel("n_features")
    plt.ylabel("score")
    plt.legend()
    plt.show()
    
def main():
    '''在最优参数下输出聚类结果'''
    dataset = loadDataset('clean_text.txt')
    X,vectorizer = transform(dataset,n_features=1000)
    km = knn_train(X,vectorizer,true_k=6,showLable=True)
    score = -km.score(X)/len(dataset)
    print (score)
    test()

if __name__ == '__main__':
    main()  

dataset = loadDataset('clean_text.txt')
X,vectorizer = transform(dataset,n_features=1000)    
start = time.time()
km = knn_train(X,vectorizer,true_k=6,showLable=True)  
end = time.time()
print("time cost:",end-start)  
km.labels_#第5类是技能要求

#关键词和权重
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
weights = np.sort(km.cluster_centers_)[:, ::-1]
#argsort()将每行数组的值从小到大排序后，并按照其相对应的索引值输出.
#argsort()[:, ::-1] 将每行数组的值从大到小排序后，并按照其相对应的索引值输出.
#np.sort(km.cluster_centers_)[:, ::-1] 把每行数组的值从大到小排序
terms = vectorizer.get_feature_names() #list

import csv
csvfile = open("term_weight_km.csv",'w',newline='',encoding='utf-8-sig') 
writer = csv.writer(csvfile)
for j in range(km.cluster_centers_.shape[0]):
    for i in range(500):
        writer.writerow([terms[order_centroids[j,i]],weights[j,i],j])
csvfile.close()
```

经过多次尝试不同的聚类个数，发现把聚类个数定为6类时，能取得较好的聚类效果，即各个类别的文本能表达清晰明确的共同含义。图1中六个词云图展示了kmeans聚类的每个聚类中心的关键词，关键词大小与kmeans输出的权重大小有关。可以看到，前最上方的两张词云图都是描述日常工作任务，权重大的词有“整理”、“研究”、“相关”、“用户”、“数据处理”，“完成”、“项目”、“整理”、“收集”。中间左图中，权重大的词有“接收”、“暑期实习”、“每周”、“四天”等，可以看出这个类别的句子是跟实习时间有关。

中间右图中出现了很多数据分析常用的软件，如“python”、“excel”、“sql”、“spss”、“sas”，说明这个类别的句子是描述专业技能的，从中可以看出，office软件仍然是最为基础的要求，同时也要求应聘者能熟练掌握sql语言、使用数据库，当python、R软件等编程软件兴起时，像sas、spss等传统的统计分析软件仍然占据半壁江山，另外还有些数据分析实习要求掌握大数据相关的软件如hadoop、hive等。

最下方左图中权重大词有“逻辑思维”、“沟通”、“责任心”、“团队精神”、“细心”等，说明这些品质是数据分析岗最为看重的。最下方右图则是对应聘者的学历、专业的要求描述，要求最多的专业是“统计学”、“数学”。

<img src="{{ 'assets/images/post_images/tagxedo.png'| relative_url }}" /> 


#### 2.GMM聚类
GMM聚类是一种基于模型的聚类方法，它并不要求每个文本只属于一个组，而是给出一个文本属于不同组的概率。这种聚类方法也叫“软聚类”。这类方法通常假设数据满足一定的概率分布，聚类的过程就是要尽力找到数据与模型之间的拟合点。GMM假设数据服从高斯混合分布（Gaussian Mixture Distribution），
GMM中的k个组件对应于k个族，所以GMM聚类的过程实际上是以似然函数作为评分函数，求使得似然函数最大化的k组$\omega$ ,$\mu$,$\Sigma$ 参数。

```python
#GMM 文本聚类——————————————————————————————————————————————————————————————
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
import os
import numpy as np

#os.chdir("E:/graduate/class/EDA/final")
"""
tfidf 提取1000维特征时，GMM输出的概率不是0就是1
不知道是否降维后GMM可以输出概率？？？
"""
def loadDataset(myfile):
    '''导入文本数据集'''
    f = open(myfile,'r',encoding = "utf-8")
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip())

    f.close()
    return dataset

def transform(dataset,n_features=1000):
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=n_features, min_df=2,use_idf=True)
    X = vectorizer.fit_transform(dataset)
    return X,vectorizer

def gmm_train(X,vectorizer,n_components,showLable = False):
 
    gmmModel = GaussianMixture(n_components=n_components, covariance_type='full', random_state=0)
    gmmModel.fit(X.toarray())
    if showLable:
        print("Top terms per cluster:")
        order_centroids =gmmModel.means_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print (vectorizer.get_stop_words())
        for i in range(n_components):
            print("Cluster %d:" % i, end='')
            for ind in order_centroids[i, :20]:
                print(' %s' % terms[ind], end='')
            print()
        result = list(gmmModel.predict(X.toarray())) #标签
        print ('Cluster distribution:')
        print (dict([(i, result.count(i)) for i in result]))
    return gmmModel
    
dataset = loadDataset('clean_text.txt')
X,vectorizer = transform(dataset,n_features=1000)   
start = time.time()
gmmModel = gmm_train(X,vectorizer,n_components=5,showLable=True)  #聚成7类以上是可以接受的结果 
end = time.time()
print("time cost:",end-start)  
gmm_labels = gmmModel.predict(X.toarray())#第3类是技能要求
#p = gmmModel.predict_proba(X.toarray())
#np.round(p,2)
#
#print(gmmModel.weights_)
#print(gmmModel.means_)
#print(gmmModel.covariances_)



#用jieba提取关键词
from jieba.analyse import extract_tags

def get_labels_kw(clean_text,labels,name, topK=15):
    def get_label_i(i,clean_text = clean_text,labels = labels): 
        """获取不同标签的文本"""
        j=0
        period_i=[]
        for j in list(range(len(clean_text))):
            if labels[j] == i:
                period_i.append(clean_text[j])
        print("标签%d有%d条文本"%(i,len(period_i)))
        return " ".join(period_i)

    def get_kw(text):
        return extract_tags(text, topK=topK, withWeight=False, allowPOS=())
    
    period_text = list(map(get_label_i,np.unique(labels)))
    news_kw = list(map(get_kw,period_text))

    for j in range(len(news_kw)):
        print("\n%s方法第%d个类别的关键词：\n"%(name,j+1))
        for i in range(len(news_kw[j])):
            print(news_kw[j][i])
 
get_labels_kw(dataset,gmm_labels,"gmm")    #第一类是技能要求
```

由于文本聚类的稀疏性，且本文所使用句子都有明显的特征，因此GMM聚类后给出的每个句子的概率都十分接近0或1，相当于kmeans的效果。经过多次尝试不同的聚类个数，发现把聚类个数定为8类时，能取得较好的聚类效果，各个类别的关键词能体现明确的文本摘要。

表5展示了每个聚类中心的关键词。从类别的样本分布可以看出，除类别3有1000多条样本外，其他类别样本分布均匀。说明职位描述中，篇幅最大的是日常工作任务描述。

从表中关键词可以看出，第1、6类是专业技能描述，第2、 3、 8类是任务描述，第4类是实习时间描述，第5类是专业、学历描述，第7类是通用技能、品质描述。另外，尽管第1、6类是专业技能描述，但却略有不同，第1类出现的软件为python、java、hive、hadoop，还出现了算法、数据挖掘、机器学习等词，说明此类职位描述的对编程、对分布式、算法要求更高些，而第6类则只是要求应聘者会office办公软件，以及传统的sas、spss统计分析软件，也要求python。这说明数据分析岗仍可以往下细分为普通统计分析和偏向算法工程师的数据分析。


|类别|	1|	2|	3|	4|	5|	6|	7|	8|
|---|---|---|---|---|---|---|---|---|
|样本数|	212|	179|	1163|	202|	220|	209|	301|	331|
|关键词|熟悉 了解 python sql 一定 工具 数据挖掘 熟练掌握 一种 算法 掌握 语言 经验 方法 hive 数据库 java hadoop 常用 机器学习|提供 支持 运营 产品 决策 日常 报告 部门 协助 报表 公司 提供数据 相关 团队 需求 包括 优化 指标 建议 用户|相关 完成 经验 项目 协助 公司 报告 整理 研究 参与 行业 部门 维护 系统 学习 管理 撰写 产品 问题 平台|实习 以上 每周 至少 接受 长期 时间 暑期实习 保证 一周 实习期 能够 天及 转正 经验 工作日 全职 期间 考虑 四天|数学 以上学历 相关专业 本科 计算机 统计学 专业  硕士 研究生 金融 经济学 在读 全日制  大学本科|使用 熟练 excel sql 软件 ppt 办公软件 python 工具 熟悉 office 操作 spss 熟练掌握 常用 掌握 精通 sas 运用|沟通 良好 具备 学习 责任心 逻辑思维 较强 团队协作 敏感 团队 协调 逻辑 合作精神 表达能力 优秀 敏感度 抗压 善于 精神|进行 需求 用户 提出 整理 协助 产品 报告 模型 相关 项目 理解 挖掘 行为 优化 收集 建议 业务部门 信息 研究|

#### 3.NMF聚类
NMF是一种非线性降维方法，降维后的矩阵相当于对原文档词频矩阵进行了特征提取，过滤噪声特征项，因此提取的特征更能反映样本的局部特征，聚类效果更好。NMF聚类的结果如表6所示，可以看到NMF仅聚成5个类别，即可使每个类别的文本有清晰明确的文本摘要。

```python
# NMF-based clustering-------------------------------------------------
#    ||X - UVT||
from sklearn.decomposition import NMF
from numpy.linalg import norm

def nmf_train(X,n_components):
    model = NMF(n_components=n_components, init='random', random_state=0)
    U = model.fit_transform(X.T)
    VT = model.components_    
    
    #1000个词，k = 10，文档数2772
    # 归一化
    V = VT.T
    
    nu,pu = U.shape
    nv,pv = V.shape
    
    for i in range(nv):
        for j in range(pv):
            V[i,j] = V[i,j]*norm(U[:,j])
    for j in range(pu):
        U[:,j] = U[:,j]/norm(U[:,j])
    #使用矩阵H来决定每个文档的归类。那个文档di的类标为m，当：m = argmaxj{vij}
    V.shape #(2773, 10)
    nmf_labels = list(map(np.argmax,V))
    return nmf_labels

from jieba.analyse import extract_tags

def get_labels_kw(clean_text,labels,name,topK=15):
    def get_label_i(i,clean_text = clean_text,labels = labels): 
        """获取不同标签的文本"""
        j=0
        period_i=[]
        for j in list(range(len(clean_text))):
            if labels[j] == i:
                period_i.append(clean_text[j])
        print("标签%d有%d条文本"%(i,len(period_i)))
        return " ".join(period_i)

    def get_kw(text):
        return extract_tags(text, topK=topK, withWeight=False, allowPOS=())
    
    period_text = list(map(get_label_i,np.unique(labels)))
    news_kw = list(map(get_kw,period_text))

    for j in range(len(news_kw)):
        print("\n%s方法第%d个类别的关键词：\n"%(name,j+1))
        for i in range(len(news_kw[j])):
            print(news_kw[j][i])
            
start = time.time()     
nmf_labels = nmf_train(X,n_components=5) #聚成5/6/7类的效果还行
get_labels_kw(dataset,nmf_labels,"nmf",20)   #第0类是技能要求  
end = time.time()
print("time cost:",end-start)  
```


|    类别      |    1                                                                                                                         |    2                                                                                                             |    3                                                                                                                 |    4                                                                                                                   |    5                                                                                                      |
|--------------|------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
|    样本数    |    275                                                                                                                       |    1355                                                                                                          |    389                                                                                                               |    480                                                                                                                 |    318                                                                                                    |
|    关键词    |    以上学历   专业   本科   数学   统计学   相关   计算机   统计   在读   研究生   专业本科   全日制   硕士   大三   金融    |    协助   相关   数据挖掘   进行   运营   报告   需求   用户   整理   项目   产品   完成   报表   建模   支持    |    excel   熟练   sql   python   使用   软件   熟悉   ppt   熟练掌握   工具   sas   office   办公   spss   数据库    |    沟通   良好   团队   具备   学习   责任心   逻辑思维   精神   具有   逻辑   合作   表达能力   较强   抗压   敏感    |    实习   每周   以上   暑期   至少   接受   时间   长期   实习期   转正   保证   一周   全职   实习生    |
|              |                                                                                                                              |                                                                                                                  |                                                                                                                      |                                                                                                                        |                                                                                                           |



## 聚类方法的比较
#### 1.聚类效果
从以上展示的聚类效果来看，NMF的聚类类别最少，仅5类就可以使得每个类别具有明确清晰的，符合预先设想的文本摘要。
从聚类算法运行速度来看，同样都是从python中sklearn调用的函数，NMF最快,仅需要0.46s；kmeans与之相差无几，需0.56s， GMM最慢，需要16.78s才能运行完毕。
#### 2.兰德指数
兰德指数是一种基于聚类相似度的评价指标，它通过观察一对样本点xi,yi在两种聚类方法中是否被分在同一个类别来判断两种聚类方法的相似性。从表7可以看出，GMM、kmeans与NMF的聚类效果相似度低，兰德指数少于0.3；kmeans和GMM的聚类效果较相似，两者的兰德指数为0.43，说明两者中有43%对样本点是分在了同一个类别或不在同一个类别。这可能是因为GMM是k-Means方法的概率变种，其基本算法框架和k-Means类似，都是通过多次迭代逐步改进聚类结果的质量。

```python
#聚类效果比较——————————————————————————————————————————————————————————
# 兰德指数：比较聚类效果的相似性
from sklearn.metrics import adjusted_rand_score  
adjusted_rand_score(km.labels_,gmm_labels)    #相似度低。。
adjusted_rand_score(km.labels_,nmf_labels)     
adjusted_rand_score(gmm_labels,nmf_labels)       
# 轮廓系数
from sklearn.metrics import silhouette_score
silhouette_score(X,km.labels_, metric='euclidean')
silhouette_score(X,gmm_labels, metric='euclidean')
silhouette_score(X,nmf_labels, metric='euclidean')

```

|    兰德指数    |    Kmeans    |       GMM    |    NMF    |
|----------------|:------------:|-------------:|-----------|
|    Kmeans      |              |              |           |
|    GMM         |    0.4310    |              |           |
|    NMF         |    0.2480    |    0.2862    |           |


## LDA聚类
#### 1. 主题数为6时各个主题的关键词能形成明确的文本摘要
聚类结果与kmeans类似

#### 2.主题数为10时技能主题分化
当LDA提取的主题数设为10时，如图2所示，出现了两个专业技能主题，却略有不同，一个主题中出现的软件为python、java、hive、hadoop，还出现了算法、数据挖掘、机器学习等词，说明此类职位描述的对编程、对分布式、算法要求更高些，而另一个主题则只是要求应聘者会office办公软件，以及传统的sas、spss统计分析软件，也要求python。这说明数据分析岗仍可以往下细分为普通统计分析和偏向算法工程师、大数据方面的数据分析。

```python
#LDA-------------------------------------------------------------------
from gensim import corpora
import os
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import LdaModel


os.chdir("/Users/situ/Documents/EDA/final")


def loadDataset(myfile):
    '''导入文本数据集'''
    f = open(myfile,'r',encoding = "utf-8")
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip().split())

    f.close()
    return dataset


clean_text4 = loadDataset("clean_text.txt")
TEMP_FOLDER = tempfile.gettempdir()
print('Folder "{}" will be used to save temporary dictionary and corpus.'.format(TEMP_FOLDER))
dictionary = corpora.Dictionary(clean_text4)
dictionary.save(os.path.join(TEMP_FOLDER, 'deerwester.dict'))  # store the dictionary, for future reference
print(dictionary)
print(dictionary.token2id)
corpus = [dictionary.doc2bow(text) for text in clean_text4]

corpora.MmCorpus.serialize(os.path.join(TEMP_FOLDER, 'deerwester.mm'), corpus)  # store to disk, for later use
len(corpus)

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))


# Set training parameters.
num_topics = 5
#10 8
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make a index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                       alpha='auto', eta='auto', \
                       iterations=iterations, num_topics=num_topics, \
                       passes=passes, eval_every=eval_every)

top_topics = model.top_topics(corpus,5)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)


model.print_topic(1,30)
model.print_topic(3,30)

#判断一个训练集文档属于哪个主题
for index, score in sorted(model[corpus[0]], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, model.print_topic(index, 10)))
    
 
#给训练集输出其属于不同主题概率   
for index, score in sorted(model[corpus[0]], key=lambda tup: -1*tup[1]):
    print(index, score)
    
    
    
    
#判断一个测试集文档属于哪个主题
#unseen_document = [" ".join(text_i) for text_i in clean_text4[130]]
#unseen_document = " ".join(unseen_document)
    
unseen_document = text[130]
"""
还要对文档进行之前的文本预处理
"""


bow_vector = dictionary.doc2bow(unseen_document.split())
for index, score in sorted(model[bow_vector], key=lambda tup: -1*tup[1]):
    print("Score: {}\t Topic: {}".format(score, model.print_topic(index, 10)))

#给每个完整的分词后的文档生成不同主题的得分
import pandas as pd
import numpy as np
data = pd.read_csv("data_with_wordseg.csv",encoding = "gbk")
data.head()
lda_score = np.zeros((data.shape[0],num_topics))
for i in range(len(data["word_seg"])):
    line = data["word_seg"][i]
    bow_vector = dictionary.doc2bow(line.split())
    for index, score in sorted(model[bow_vector], key=lambda tup: -1*tup[1]):
        lda_score[i,index] = score
lda_score[:5,:]
lda_score_df = pd.DataFrame(lda_score,columns = ["lda"+str(i) for i in range(num_topics)])
lda_score_df.head()
data = pd.concat([data, lda_score_df], axis=1)
data.to_csv("data_with_lda_score.csv",index = False,encoding = "gbk")


#LDA visualization---------------------------------------------------

import pyLDAvis
import pyLDAvis.gensim

vis_wrapper = pyLDAvis.gensim.prepare(model,corpus,dictionary)
pyLDAvis.display(vis_wrapper)
pyLDAvis.save_html(vis_wrapper,"lda%dtopics.html"%num_topics)


#pyLDAvis.enable_notebook()
#pyLDAvis.prepare(mds='tsne', **movies_model_data)
```

<img src="{{ 'assets/images/post_images/LDA.png'| relative_url }}" /> 



## 参考博客：
描述统计参考：

- [数据分析师挣多少钱？“黑”了招聘网站告诉你！](https://zhuanlan.zhihu.com/p/25704059)


模型参考：

- [利用python做LDA文本分析，该从哪里入手呢？](https://www.zhihu.com/question/39254526)
