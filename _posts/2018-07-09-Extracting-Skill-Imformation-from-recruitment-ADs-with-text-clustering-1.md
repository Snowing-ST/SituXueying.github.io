---
title: 实习僧爬虫与招聘信息文本预处理
author: Situ
layout: post
categories: [big data]
tags: [爬虫,文本预处理,文本聚类]
---

本文通过爬取实习僧网站“数据分析”一职的实习信息，对“职位描述”的文本进行预处理、分句，使用文本聚类的方式提取每条实习信息中其中的描述专业技能的句子，并对其描述的专业技能进行量化，从而探究专业技能对薪资的影响。

# 《基于文本聚类的招聘信息中技能要求提取与量化》


- <font color="pink" >上篇 实习僧爬虫与招聘信息文本预处理</font>
- 中篇 [招聘信息文本聚类](Extracting-Skill-Imformation-from-recruitment-ADs-with-text-clustering-2.html)
- 下篇 [量化薪资与技能的关系](Extracting-Skill-Imformation-from-recruitment-ADs-with-text-clustering-3.html)

## 实习僧爬虫
在本次抓取中，一共抓取了实习僧上所有职位名称包含“数据分析”的实习信息351条，数据的主体为文本形式的数据。数据抓取的方式为使用python的request库获取具体实习信息的网页源代码，通过re模块使用正则表达式匹配出需要的信息。

#### 1. 薪资数字设了掩码，而且每隔一段时间换一次，不能直接爬取，要自己查看网页源代码找到0-9的对应掩码

```python
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'
}

#掩码和数字一一对应的字典，应该已过期
replace_dict={
    "&#xf09f":"0",
    "&#xeff8":"1",
    "&#xecfa":"2",
    "&#xf748":"3",
    "&#xf298":"4",
    "&#xed58":"5",
    "&#xee56":"6",
    "&#xe253":"7",
    "&#xe504":"8",
    "&#xecfd":"9"}
```
#### 2.  爬取搜索页面每条招聘广告的url，放入```get_infos``` 函数爬取具体信息得到一条样本one_pd，再合并表格得到all_pd

```python
import requests,re
import pandas as pd
import numpy as np

def get_links(start_url,n,replace_dict):
    all_pd = pd.DataFrame()
    for i in list(range(1,n+1)):
        print("————————————正在爬取第%d页招聘信息———————————————"%i)
        url = start_url+"&p=%s"%str(i)
        try:
            wb_data = requests.get(url,headers=headers)
            wb_data.encoding=wb_data.apparent_encoding
            links = re.findall('class="name-box clearfix".*?href="(.*?)"',wb_data.text,re.S)
            for link in links:
                print(link)
                try:
                    one_pd = get_infos('https://www.shixiseng.com'+link,replace_dict)
                except:
                    one_pd = pd.DataFrame({
                        "url":link,
                        "jobname":"",
                        "salary":"",
                        "address":"",
                        "education":"",
                        "jobway":"",
                        "month":"",
                        "jobgood":"",
                        "contents":"",
                        "compname":"",
                        "city":"",
                        "size":"",
                        "industry":""})
                    print("can't crawl"+link)
                all_pd = all_pd.append(one_pd)
        except:
            print("can't reach page %d"%i)
            pass
                
    return all_pd
```
#### 3. 对每个招聘广告页面用正则表达式找到所需的数据

```python
def get_infos(url,replace_dict):
    one_dict = {}
    wb_data = requests.get(url,headers=headers)
    print(wb_data.status_code)
    wb_data.encoding=wb_data.apparent_encoding
    jobname = re.findall('<div class="new_job_name" title="(.*?)">',wb_data.text,re.S)
    salarys = re.findall('class="job_money cutom_font">(.*?)</span>',wb_data.text,re.S)
    addresses = re.findall('class="job_position">(.*?)</span>',wb_data.text,re.S)
    educations = re.findall('class="job_academic">(.*?)</span>',wb_data.text,re.S)
    jobways = re.findall('class="job_week cutom_font">(.*?)</span>',wb_data.text,re.S)
    months = re.findall('class="job_time cutom_font">(.*?)</span>',wb_data.text,re.S)
    jobgoods = re.findall('class="job_good".*?>(.*?)</div>',wb_data.text,re.S)
    contents = re.findall(r'div class="job_til">([\s\S]*?)<div class="job_til">', wb_data.text, re.S)[0].replace(' ','').replace('\n', '').replace('&nbsp;', '')
    contents = re.sub(r'<[\s\S]*?>', "", str(contents))
    compname = re.findall('class="job_com_name">(.*?)</div>',wb_data.text,re.S)
    compintro = re.findall('<div class="job_detail job_detail_msg"><span>([\s\S]*?)</span></div>',wb_data.text,re.S)
    city,size,industry = re.sub(r'<[\s\S]*?>', " ", str(compintro[0])).split()
    for salary,address,education,jobway,month,jobgood in zip(salarys,addresses,educations,jobways,months,jobgoods):
        for key, vaule in replace_dict.items():
            salary = salary.replace(key, vaule)
            jobway = jobway.replace(key,vaule)
            month = month.replace(key,vaule)
            one_dict = {
                "url":url,
                "jobname":jobname,
                "salary":salary,
                "address":address,
                "education":education,
                "jobway":jobway,
                "month":month,
                "jobgood":jobgood,
                "contents":contents,
                "compname":compname,
                "city":city,
                "size":size,
                "industry":industry}
#    list_i=[url,salary,address,education,jobway,month,jobgood,contents,compname,city,size,industry]
    print(jobname)
    one_pd = pd.DataFrame(one_dict)
    return one_pd
```

#### 4. 爬虫时先识别某搜索词下一共有多少页的搜索结果，再爬取全部页面

```python
import time
from urllib.parse import urlencode
from lxml import etree
import os

def main():
    os.chdir("")
    print('请输入您想爬取内容的关键字：')
    compRawStr = input('关键字： \n')     #键盘读入 多个关键字则用空格隔开
    print('正在爬取“' + compRawStr.capitalize()+ '”有关实习信息!')
    d = {'k': compRawStr.encode('utf-8')}
    word = urlencode(d)

    start_url = "https://www.shixiseng.com/interns/st-intern_c-None_?%s" %word
    result = requests.get(start_url,headers=headers)
#    result.status_code
    result.encoding = 'utf-8'
    selector = etree.HTML(result.text)  
    last_page_link = selector.xpath('//*[@id="pagebar"]/ul/li[10]/a/@href')
    n = int(last_page_link[0].split("p=")[1])
    print("将爬取%d页的招聘信息"%n)
    time_start=time.time()
    df = get_links(start_url,n,replace_dict)
    df.to_csv(compRawStr+"_共"+str(n)+"页.csv",index = False,encoding = "gb18030")
    time_end=time.time()
    print("成功爬取%d条关于【%s】的招聘信息"%(len(df),compRawStr))
    print('totally cost %f seconds'%(time_end-time_start))

if __name__ == '__main__':
    main()
```

<img src="{{ 'assets/images/post_images/recruitmentAD.png'| relative_url }}" /> 

## 文本预处理
对招聘信息中“职位描述”的文本进行预处理、分句

#### 1.分句
由于招聘信息中的“职位描述”是大多按序号列出对应聘者的多条要求，技能要求一般包含在其中的某一句或某几句，因此首先要对每条“职位描述”的文本进行分句，分割的符号为句号、分号、冒号、换行符等。

原文：
职位描述：数据分析工程师/实习生岗位职责：1、股票、期货程序化交易数据分析，包括各类高频交易数据的管理、维护、清洗等；2、构建、训练机器学习模型；3、研究、学习各类金融数据以及获取途径任职要求：1、物理、数学、电子、计算机等相关专业在读研究生；2、熟悉Python，了解Python基本语法、数据结构、性能特征，熟悉动态语言的基本性质；3、熟悉机器学习、深度学习，熟悉C#、SQL/MySQL数据库者优先；4、具有较强的沟通能力	

分句结果：
|职位描述|
|---|
|数据分析工程师/实习生岗位职责|
|1、股票、期货程序化交易数据分析，包括各类高频交易数据的管理、维护、清洗等|
|2、构建、训练机器学习模型| 
|3、研究、学习各类金融数据以及获取途径任职要求|
|1、物理、数学、电子、计算机等相关专业在读研究生|
|2、熟悉Python，了解Python基本语法、数据结构、性能特征，熟悉动态语言的基本性质|
|3、熟悉机器学习、深度学习，熟悉C#、SQL/MySQL数据库者优先 |
|4、具有较强的沟通能力|

#### 3.去除停用词和特殊字符
去除停用词指过滤文本中的特殊字符和对文本含义无意义的词语。例如“的”，“啊”一类的语气语助词，对文本情感倾向判定无意义，却在文本向量表示时由于占据较大比重而对后续分析造成干扰，降低情感分类的准确性。研究中用到的停词表在《哈工大停用词表》的基础上，根据帖子文本特点进行了修改。


#### 4.去除超高频词与低频词
去除停用词后先做词频统计，发现词频极高的词，如“数据分析”、“职位描述”、“工作职责”、“负责”“工作”等不能体现具体岗位要求的词，因此删除前10个超高频词。

由于存在大量无意义的低频词（本文定义出现的频率仅为1次的为低频词）可能会降低分类精度，因此对去除停用词后的文本再删除低频词。


```python
import pandas as pd
import numpy as np
import os
import re
import jieba
from collections import Counter,defaultdict
import operator
from nltk import ngrams
import csv
import matplotlib.pyplot as plt


os.chdir("")
jieba.load_userdict("dict.txt")


#手动删除英文的招聘信息
data = pd.read_csv("数据分析_共47页.csv",encoding = "gbk")
data.head()
text = data["contents"]

#删除重复内容
sum(data["contents"].duplicated()) 
data[data["contents"].duplicated()]

data = data.drop_duplicates(["contents"])
#检查是否有空值
sum(data["contents"].isnull())

#文本预处理函数————————————————————————————————————————————————————————
def get_text(data):
    text=data["contents"]
    text = text.dropna() 
    len(text)
    text=[t.encode('utf-8').decode("utf-8") for t in text] 
    return text

def get_stop_words(file='stopWord.txt'):
    file = open(file, 'rb').read().decode('utf8').split(',')
    file = [line.strip() for line in file]
    return set(file)   #查分停用词函数

def rm_tokens(words):                                        # 去掉一些停用词和完全包含数字的字符串
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:   # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list

def rm_char(text):
    text = re.sub('\x01', '', text)  #全角的空白符
    text = re.sub('\u3000', '', text) 
    text = re.sub(r"[\)(↓%·▲ \s+】&【]","", text) 
    text = re.sub(r"[\d（）《》><‘’“”"".,-]"," ",text,flags=re.I)
    text = re.sub('\n+', " ", text)
    text = re.sub('[，、：。！？?；——]', " ", text)
    text = re.sub(' +', " ", text)
    return text

def convert_doc_to_wordlist(paragraph, cut_all=False):
    sent_list = [sent for sent in re.split(r"[。！;:\n.；：?]",paragraph)]
    sent_list = map(rm_char, sent_list)                       # 去掉一些字符，例如\u3000
    word_2dlist = [rm_tokens(jieba.cut(part, cut_all=cut_all))
                   for part in sent_list]                     # 分词
#    word_list = sum(word_2dlist, [])
    def rm_space_null(alist):
        alist = [s for s in alist if s not in [""," "]]
        return alist
    rm_space = [rm_space_null(ws) for ws in word_2dlist if len(ws)>0]
    return rm_space

def rm_1ow_freq_word(texts,low_freq=1):
#去除低频词
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [[token for token in text if frequency[token] > low_freq] for text in texts]
    return texts

def rm_short_len_word(texts,short_len=0):
#去除字符太短的样本
    texts = [[token for token in text if len(token)>short_len and len(token)<15] for text in texts]
    return texts

def rm_high_freq_word(texts,num=10,other_dele_file="delete_words.txt"):
# 去除高频词
    whole_text = []
    for doc in texts:
        whole_text.extend(doc)
    word_count = np.array(Counter(whole_text).most_common())
    high_freq = []
    for i in range(num):
        high_freq.append(word_count[i][0])
    if other_dele_file!=None:
        other_dele_list = open(other_dele_file, 'rb').read().decode('gbk').split('\n')
        high_freq.extend(other_dele_list)
        dele_list = np.unique(high_freq)
    else:
        dele_list = high_freq
#    print(dele_list)
    texts = [[token.lower() for token in text if token not in dele_list] for text in texts]
    return texts

def main():
    clean_text=[convert_doc_to_wordlist(line) for line in get_text(data)]     
    clean_text = [sent for para in clean_text for sent in para] #拆成一个个句子
    
#    length = np.array([len(sent) for sent in clean_text]) #检查太长的文本
#    plt.hist(length)
#    np.array(clean_text)[length>50]

    clean_text2 = rm_1ow_freq_word(clean_text)
    clean_text3 = rm_short_len_word(clean_text2)
    clean_text4 = rm_high_freq_word(clean_text3)
    clean_text5 = [sent for sent in clean_text4 if len(sent)>2]
    
    with open("clean_text.txt","w") as f2:
        for sent in clean_text5:
            sent1 = " ".join(sent)+"\n"
            f2.write(sent1)

if __name__ == '__main__':
    main()  
```
#### 2. 如何设置jieba用户自定义字典
由于数据分析领域存在不少专有词汇，如果只用jieba包默认的词典进行分析，则会无法识别这些专有词汇，因此在jieba包添加了自定义词典。

除本人对数据分析的了解而添加的词汇外，大部分词汇是通过统计一元和二元词频从而发现被误分的词组而添加的。 

这就是```dict.txt```的由来


```python
from collections import Counter,defaultdict
import operator
from nltk import ngrams
import csv

#导入数据
def loadDataset():
    '''导入文本数据集'''
    f = open('clean_text.txt','r')
    dataset = []
    for line in f.readlines():
#        print(line)
        dataset.append(line.strip())

    f.close()
    return dataset

text = loadDataset()

def word_count(texts):
#词频统计,转化成矩阵
    texts_list = [w for text_i in texts for w in text_i.split() ]
    word_count = np.array(Counter(texts_list).most_common())
    print (word_count[:10])
    csvfile = open("wordcount.csv",'w',newline='',encoding='utf-8-sig') 
    writer = csv.writer(csvfile)
    for row in word_count[0:1000,]:
        writer.writerow([row[0], row[1]])
    csvfile.close()

word_count(text)


# 统计2-gram词频,写入csv
def CountNgram(text,n=2,print_n=20):
    ngram_list = []
    for text_i in text:
        analyzer2 = ngrams(text_i.split(),n)
        Ngram_dict_i = Counter(analyzer2)        
        for k in Ngram_dict_i.keys():
            ngram_list.append("/".join(k))
    Ngram_dict = Counter(ngram_list)
    sortedNGrams = sorted(Ngram_dict.items(), key = operator.itemgetter(1), reverse=True) #=True 降序排列
    print("the top %d wordcount of %d gram model of period_1 is:\n" %(print_n,n),sortedNGrams[:print_n],"/n")

    csvfile = open("2gram_wordcount.csv",'w',newline='',encoding='utf-8-sig') 
    writer = csv.writer(csvfile)
    for line in sortedNGrams:
        writer.writerow([line[0],line[1]])
    csvfile.close()

CountNgram(text)
```
添加的部分自定义词汇示例：

|技能型词汇|专业、年级词汇|	通用技能、品质词汇|
|---|---|---|
|机器学习、深度学习、数据运营、数据挖掘、统计分析、数学、自然语言处理、R语言、办公软件|本科以上学历、相关专业、在读研究生、暑期实习|注重细节、合作精神、逻辑清晰、团队协作|

#### 3. 分词分句结果示例

|预处理后的文本|	描述类别|
|---|---|
|产品库 日常 内容 维护 编辑 录入 整理 撰写 发布	|任务描述|
|参与 产品库 优化 问题 整理 反馈	|任务描述|
|协助 对接 部门 录入 需求	|任务描述|
|大三 大四 学生 理工科 含 专业 专业 专业 考虑	|专业、学历描述|
|熟练 使用 各类 办公 设计 软件	|专业技能描述|
|较强 逻辑思维 归纳 总结 较强	|通用技能、品质描述|
|具有 良好 职业道德 踏实 认真 注重细节	|通用技能、品质描述|
|协助 数据运营 中心 进行 资料 搜集 整理 资料 审核	|任务描述|
|协助 数据分析师 公司 数据库 内 完成 数据清洗 配置 规则 监控 辅助	|任务描述|
|保证 半年 以上 内 每周 至少 天到 岗 时间	|实习时间描述|
|诚实 成熟 稳重 善于 交流	|通用技能、品质描述|
|良好 沟通 协调 团队协作 精神	|通用技能、品质描述|
|相关专业 统计学 数学 信息工程 计算机 本科	|专业、学历描述|
|熟练 使用 msoffice 办公软件 excel powerpoint	|专业技能描述|
|基于 公司 大数据 平台 海量 用户 运用 数据挖掘 理论 方法 准确 快速 处理	|任务描述|


文本预处理后的文本如表4所示，可以看到，每一句职位描述都有大致能看出其明确的类别，日常工作任务描述通常包含“整理”“录入”“搜集”这些动词；用人单位对应聘者专业的要求通常会指定具体专业和年级，如“大三”、“大四”、“研一”、“研二”、“统计学”、“数学”等；专业技能的描述则会指定应聘者需要掌握什么软件，如“excel”、“sql”等；通用技能、品质描述一般是要求应聘者“具有良好职业道德”、“细心”、“认真”等；实习时间描述一般是要求应聘者能保证实习“三个月”、“六个月”等，每周到岗“三天”、“四天”等。
由此可以预见，之后的文本聚类将会取得良好效果。


## 参考博客：

爬虫参考：

- [实习僧网站爬取](https://www.cnblogs.com/mayunji/p/8779016.html)
- [爬虫实战 破解“实习僧”网站字体加密](http://www.yidianzixun.com/0Ie72atu)

数据预处理参考：

- [Alfred1984的github项目](https://github.com/Alfred1984/interesting-python/tree/master/shixiseng)
