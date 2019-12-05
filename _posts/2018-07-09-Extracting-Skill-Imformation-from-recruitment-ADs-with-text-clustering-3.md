---
title: 量化薪资与技能的关系
author: Situ
layout: post
categories: [big data]
tags: [regression,文本聚类]
---

对招聘信息中描述的专业技能进行量化，从而探究专业技能对薪资的影响。

# 《基于文本聚类的招聘信息中技能要求提取与量化》

- 上篇 [实习僧爬虫与招聘信息文本预处理](Extracting-Skill-Imformation-from-recruitment-ADs-with-text-clustering-1.html)
- 中篇 [招聘信息文本聚类](Extracting-Skill-Imformation-from-recruitment-ADs-with-text-clustering-2.html)
- <font color="pink">下篇 量化薪资与技能的关系 </font>

## LDA主题模型量化技能要求
#### 1. 专业技能关键词与薪资的关系
从职位描述的文本中提取专业技能关键词，选出需求频率最高的前10个技能，挑选出包含这些技能的招聘信息，根据包含某一个技能的所有招聘信息的工资计算出该技能对应的平均工资，如图3所示，横轴为技能关键词，纵轴为平均工资，点的大小代表该技能需求量的多少。



在前10项技能中，excel需求最大，但平均薪资最低，仅为144元，因为excel是数据分析工作最应该掌握的工具；Hadoop， Spark这两者需求少，但平均薪酬水平最高，超过200元，并且相对其他技能来说有比较大的差异，因为Hadoop， Spark都是应用于分布式数据处理；其他软件对应得平均薪资在160-200之间。因此专业技能对薪资有明显影响。

<img src="{{ 'assets/images/post_images/skill-salary.png'| relative_url }}" /> 

#### 2. LDA量化技能要求
通过LDA的提取10个主题的图中可以看出，专业技能描述也有高低之分，从前面的分析也可看出，要求应聘者掌握hadoop、spark等大数据分析相关技能的实习工资更高些，但仅通过从文本中提取技能关键词来衡量技能与薪资的关系，一来需要预先知道有哪些重要技能，二来提取的技能太多会使得技能因素分散在每个技能变量上，每个技能变量包含的信息较少，使得这种方法更为繁琐，缺乏普适性，且不利于分析技能与薪资的关系。

因此可以将每条样本的职位描述中专业技能描述的句子挑出来再提取细分的技能主题，根据句子所倾向技能主题的高低为句子所述的技能要求进行评分，这样无需一个个提取技能关键词，且把句子中的关键词综合考量。因此将提取了6个主题的LDA模型的第4个主题（技能主题）取值大于0.5的句子取出来，构成小型专业技能描述句子语料库，一共268句，587个词，对该语料库再次建立LDA主题模型，效果如图4所示：

```python
#把文本中的技能要求提取出来，然后再对技能类文本聚类，看看能不能聚出高端技能、低端技能

lda_score = np.zeros((len(dataset),num_topics))
for i in range(len(dataset)):
    line = dataset[i]
    bow_vector = dictionary.doc2bow(line.split())
    for index, score in sorted(model[bow_vector], key=lambda tup: -1*tup[1]):
        lda_score[i,index] = score
lda_score[:5,:]
classify = pd.DataFrame(lda_score,columns = ["lda"+str(i) for i in range(num_topics)])
classify.head()




classify = pd.read_csv("clean_text_with_index.csv",encoding = "utf-8-sig")
classify.head()
classify["skill"][classify["lda8"]>0.5]=1


#classify.to_csv("text_labels.csv",index = False,encoding = "gbk")
# 手动把有关大数据、spark、hadoop、hive的样本选上


def skill_text_combine(df):
    """
    相同index的文本合并
    """
   
    index_list = df["index"].unique()
    skill_text = pd.DataFrame({"index":index_list,"skill_text":[""]*len(index_list)})
    for i in index_list:
        skill_text["skill_text"][skill_text["index"]==i] = " ".join(list(df["text"][df["index"]==i]))

    return skill_text

classify = pd.read_csv("text_labels.csv",encoding = "gbk")
skill_text = skill_text_combine(classify[classify["skill"]>0])
skill_text.to_csv("skill_text.csv",index = False,encoding = "gbk")

#一类只出现了msoffice
#一类还出现了sas spss python
#一类是大数据的软件相关


#重新索引，与去重且重新索引后的data合并，看看能不能对应得上，如果有的职位描述没有专业技能句子，赋值0，有的话，根据三种聚类方式进行打分123取平均
data.tail()
#重新索引
data["index"]=list(range(len(data)))

data_with_skill = pd.merge(data,skill_text,how = "outer",on="index")
data_with_skill.head(10)
data_with_skill["score"] = data_with_skill["score"].fillna(0)
data_with_skill.to_csv("data_with_skill.csv",index = False,encoding = "gbk")

```


<img src="{{ 'assets/images/post_images/LDA-3-topics.png'| relative_url }}" /> 

当LDA提取3个主题时，每个主题的关键词正好能表达技能主题的高低。第2个主题的句子仅要求应聘者掌握msoffice软件和SQL查询语言，第1个主题除了要求掌握msoffice和SQL查询语言以外，还要求掌握其他统计分析软件，如sas、spss、python等，而第3个主题则还要求应聘者会应用与大数据、分布式有关的软件，如hive、hadoop、spark、Java等。

按技能高低给技能主题打分1分、2分、3分，该职位的技能要求分数为其在三个主题上的概率乘以三个主题的分值再求和，从而量化职位的技能要求。

图5散点图显示了技能分数与平均工资的关系，可以看出大部分实习工资集中在100-200之间，而当技能分数超过2.5时，有一些实习的工资能超过300，从loess拟合的回归曲线可以看出轻微渐升的趋势，说明技能要求越高，公司愿意支付的工资越多。

<img src="{{ 'assets/images/post_images/scartter-plot.png'| relative_url }}" /> 



## 技能与薪资的回归分析
实习工资的高低还跟很多因素有关，如地域、行业等，因此接下来把这些因素考虑进去，以实习工资为因变量进行回归分析，重点观察技能分数对实习工资的影响。

```python
import numpy as np
import pandas as pd
import os
import re

#os.chdir("E:/graduate/class/EDA/final")
os.chdir("/Users/situ/Documents/EDA/final")
data = pd.read_csv("data_with_skill.csv",encoding = "gbk")
data.head()
data.info()

data.drop(["jobname","jobgood","url","city"],axis = 1,inplace = True)
#数值型数据处理----------------------
#每周工作天数
data.jobway.unique()
mapping = {}
for i in range(2,7):
    mapping[str(i) + '天／周'] = i
print(mapping)
data['day_per_week'] = data['jobway'].map(mapping)
data['day_per_week'].head()


#公司规模
data["size"].unique()
data["comp_size"] = ""
data["comp_size"][data['size'] == '少于15人'] = '小型企业'
data["comp_size"][data['size'] == '15-50人'] = '小型企业'
data["comp_size"][data['size'] == '50-150人'] = '中型企业'
data["comp_size"][data['size'] == '150-500人'] = '中型企业'
data["comp_size"][data['size'] == '500-2000人'] = '大型企业'
data["comp_size"][data['size'] == '2000人以上'] = '大型企业'

#实习月数
data.month.unique()
mapping = {}
for i in range(1,22):
    mapping["实习"+str(i) + '个月'] = i
print(mapping)
data['time_span'] = data['month'].map(mapping)
data['time_span'].apply(lambda f:int(f))

#每天工资
def get_mean_salary(s):
    return np.mean([int(i) for i in s[:(len(s)-2)].split("-")])
data['average_wage'] = data['salary'].apply(lambda s:get_mean_salary(s))
data['average_wage'].head()

data.drop(['jobway','size','month','salary'], axis = 1,inplace=True)

#字符型数据处理--------------------------------
#（城市）处理
#北京、上海、杭州、深圳、广州

def get_less_dummies(data,feature,useful_classes,prefix):
    useful_classes_prefix = [prefix+"_"+token for token in useful_classes]
    dum = pd.get_dummies(data[feature],prefix=prefix).ix[:,useful_classes_prefix]
    if sum(np.sum(dum.isnull()))>0:
        dum = dum.fillna(0)
    search_index = np.where(np.sum(dum,axis=1)==0)[0]
    for j in range(len(useful_classes)):
        token = useful_classes[j]
        for i in search_index:
            if len(re.findall(token,data.ix[i,feature]))>0:
                dum.ix[i,useful_classes_prefix[j]] = 1
#    print(dum.head())
    
    data = pd.concat([data,dum],axis = 1)
    return data

feature = "address"
useful_classes = ["北京","上海","杭州","深圳","广州","成都","武汉"]
data = get_less_dummies(data,feature,useful_classes,prefix="city")

#行业
#互联网，计算机，金融，电子商务和企业服务
 


feature = "industry"
useful_classes = ["互联网","计算机","金融","电子商务","企业服务","广告","文化传媒","电子","通信"]
data = get_less_dummies(data,feature,useful_classes,"industry")

data.head()


data.drop(['address','industry'], axis = 1,inplace=True)


#专业要求
def get_imp_info(data,feature,useful_classes,prefix):
    """直接从文本中提取"""
    useful_classes_prefix = [prefix+"_"+token for token in useful_classes]
    dum = pd.DataFrame(np.zeros((len(data),len(useful_classes))),columns = useful_classes_prefix)
    dum = dum.fillna(0)
    for j in range(len(useful_classes)):
        token = useful_classes[j]
#        print(token)
        for i in range(len(data)):
#            print(i)
            if len(re.findall(token,data.ix[i,feature].lower()))>0:
                dum.ix[i,useful_classes_prefix[j]] = 1
    print(dum.head())
    
#    data = pd.concat([data,dum],axis = 1)
    return dum


feature = "contents"
useful_classes = ["统计","计算机","数学"]
dum = get_imp_info(data,feature,useful_classes,"subject")
data = pd.concat([data,dum],axis = 1)
data.head()

#技能要求
def get_imp_info2(data,feature,useful_classes,prefix):
    """从分词中提取"""
    useful_classes_prefix = [prefix+"_"+token for token in useful_classes]
    dum = pd.DataFrame(np.zeros((len(data),len(useful_classes))),columns = useful_classes_prefix)
    dum = dum.fillna(0)
    for j in range(len(useful_classes)):
        token = useful_classes[j]
#        print(token)
        for i in range(len(data)):
            word_list = data.ix[i,feature].split()
            if token in word_list:
                print(data.ix[i,feature])
                dum.ix[i,useful_classes_prefix[j]] = 1
    print(dum.head())
    
#    data = pd.concat([data,dum],axis = 1)
    return dum


feature = "contents"
#useful_classes = ["python","r语言","spss","excel","ppt","word","sql","sas","vba","office","msoffice",
#                  "hadoop","spark","hive","scala","hbase","java","matlab","linux","shell","c#"]
#                  "机器学习","数据挖掘","数学建模","自然语言处理","自然语言","文本挖掘",
useful_classes = ['excel', 'sql', 'python', 'sas', 'spss','hadoop', 'spark', 'hive', 'shell', 'java']                  
dum = get_imp_info(data,feature,useful_classes,"skill")
np.sum(dum)
# 技能要求前10：excel sql python sas spss | hadoop spark hive shell java 
data = pd.concat([data,dum],axis = 1)
data.head()

#技能与平均薪资
def mean_salary(useful_classes,data,salary,prefix):
    feature_list = [prefix+"_"+skill for skill in useful_classes]
    p = len(feature_list)
    df = pd.DataFrame(np.zeros((p,3)),columns = ["skill","mean_salary","count"])
    df["skill"] = useful_classes
    for i in range(p):
        df["mean_salary"][df["skill"]==useful_classes[i]] = np.mean(data[salary][data[feature_list[i]]==1])
        df["count"][df["skill"]==useful_classes[i]] = len(data[salary][data[feature_list[i]]==1])
    return df

useful_classes = ['excel', 'sql', 'python', 'sas', 'spss','hadoop', 'spark', 'hive', 'shell', 'java']                  
salary = "average_wage"
prefix = "skill"
df = mean_salary(useful_classes,data,salary,prefix)

import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
plt.figure(figsize=(8,5)) 
sns.stripplot(x = "skill",y="mean_salary",data=df,size = 10)
plt.xlabel("skill_software")
plt.ylabel("mean_salary")
plt.savefig("skill_salary.jpg")

# 公司
data["compname"].value_counts()


data.drop(['compname'], axis = 1,inplace=True)
#data = pd.get_dummies(data)

#data.to_csv("data_analysis.csv",index = False,encoding = "gbk")


from sklearn.linear_model import LinearRegression
X = data.drop(["average_wage",'contents','kmeans','gmm','nmf',"skill_text","index","compname"],axis = 1);Y = data["average_wage"]
X = pd.get_dummies(X)
regr = LinearRegression().fit(X,Y)
#输出R的平方
print(regr.score(X,Y))
regr.coef_
```

<img src="{{ 'assets/images/post_images/regression.png'| relative_url }}" /> 

从表3回归系数表可以看出，技能分数对实习工资有显著影响，实习分数每多一分，即多掌握一门常用统计软件甚至多掌握一门大数据分析相关软件，则平均实习工资涨约7元。因为仅仅是实习而不是正式员工，不同的实习，日实习工资几乎只在100-200内浮动，因此技能对工资上涨影响不太大。

其他方面，从实习时间上看，要求一周实习天数越多，说明公司越需人数，愿意开出的实习工资越高；从学历要求上看，要求学历是本科生的实习工资比不限专业的工资低12.59元；从专业要求上看，要求专业是计算机的实习工资比专业要求为其他的实习工资高16.44，计算机专业出身的学生仍是就业市场中的热点需求；从实习地点上看，北上广深杭的实习工资比其他城市多20元以上，其中杭州的实习比其他城市的实习高37元，而成都、武汉则比其他城市少5元以上；从公司行业上看，互联网、计算机行业的公司更为大方些，开出的实习工资更高；从公司规模上看，中型企业比小型企业开出的实习工资少10元，而大型企业则比小型企业多3元，工资条件仍是大公司吸引就业者的优势。

在该回归方程中，F检验显著，R方仅为0.6，说明自变量对实习工资的波动仅解释了60%，另外实习工资还跟具体公司规定，市场行情有关。

## 结论
本文通过爬取实习僧网站“数据分析”一职的实习信息，对“职位描述”的文本进行预处理、分句，使用LDA提取其中包含技能主题的句子，并对这些句子再次提取技能主题，区分不同层次的技能要求，并对职位的技能要求进行打分，从而实现岗位信息中技能要求的量化，使得技能与薪酬的关系能更深入地分析。通过以上分析，可以得出以下三个结论：

第一，	数据分析师需求频率排在前列的技能有：SQL，Excel, SAS，SPSS, Python, Hadoop和MySQL等，其中SQL和Excel简直可以说是必备技能；

第二，	海量数据、分布式处理框架是走向高薪的正确方向；

第三，	SQL语言和传统的SAS，SPSS两大数据分析软件，能够让你在保证中等收入的条件下，能够适应更多企业的要求，也就意味着更多的工作机会。

本文仅以实习僧网站的数据分析实习岗为例，阐述如何通过LDA提取并量化职位描述中的专业技能要求，因此数据量比较小，代表性不够好，另外结果适合于实习方面的数据分析岗而不是正式工作。另外本次分析主要针对工具型的技能进行了分析。但实际上数据分析师所需要具备的素质远不止这些，还需要有扎实的数学、统计学基础，良好的数据敏感度，开拓但严谨的思维等。

## 参考
散点图参考：
- [数据分析师挣多少钱？“黑”了招聘网站告诉你！](https://zhuanlan.zhihu.com/p/25704059)

