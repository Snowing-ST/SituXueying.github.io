---
title: 如何爬取网络新闻
author: Situ
layout: post
categories: [big data]
tags: [爬虫,xpath,文本分类,NLP,deep-learning,Master Thesis]
---


<font face="仿宋" >基于深度学习文本分类的网络新闻情感指数编制（一）<br>如何爬取网络新闻</font>

## <center>Introduction</center>
The Consumer Confidence Index is an indicator of the strength of consumer confidence. Since the first consumer confidence index was introduced in 1946, consumer confidence index surveys have often been conducted by means of telephone surveys or questionnaires. In recent years, the survey process has gradually led to some problems, such as the increase in the rate of refusal, and the proportion of elder interviewees is too large, which has a certain impact on the validity of the index. In addition to strengthen the quality control in the design and implementation of the index survey program, we can make a new interpretation of the problem through the big data mining method.

With the rapid development of Internet technology, the Internet has replaced traditional paper media as the main channel for people to obtain and express opinions. The news reflects the public's emotional life status to varying degrees, and people's emotional state is also affected by network media to some extent. Following this intuitive logic, we attempts to construct a consumer confidence index based on online news texts by mining the emotional tendencies of consumers, thereby avoiding some problems in the traditional consumer confidence index survey, and it is timelier and thriftier. However, because there is no authoritative research to prove the direct connection between online news and consumer psychology and behavior, in order to avoid disputes, we refers to the consumer confidence index based on the online news as “Online News Sentiment Index”, which is not directly related to “consumers”, but can be used to measure the attitudes and opinions of consumers reflected in the news text.

The paper starts from the six dimensions (economic development, employment status, price level, living conditions, housing purchase and investment). From Baidu News, we crawled 68,139 news articles related to consumer confidence of 2009.01 to 2018.06, thus obtaining the original text data of this article. First, 5,000 random stories are randomly sampled for each dimension, artificially labeled with “positive”, “neutral” and “negative”, and words in the text are represented as vectors through the word2vec method, using deep learning algorithm such as such as Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN). The text classification algorithm classifies the remaining news, thereby obtaining news texts with emotional tags. Then take the ratio of the difference between the number of "positive" texts and the number of "negative" texts in a quarter as the quarterly index, and then combine the quarterly index into equal weights and add 100 points to get the quarterly Online News Sentiment Index. Then we compare the Online News Sentiment Index with the macroeconomic indicators and the traditional consumer confidence index to illustrate that the Online News Sentiment Index is highly correlated with traditional consumer confidence index, and is partial preemptive and complementary to some macroeconomic indicators. Finally, the Online News Sentiment Index and its sub-indexes are used as independent variables to predict traditional consumer confidence index by time series regression analysis, dynamic regression analysis, VAR and other multivariate time series analysis methods. The model is from simple to complex, which leads to prediction accuracy growing step by step.

## Main step 1:<center>crawling the news from <a href="news.baidu.com">Baidu News</a> </center>

### description:
- input: search key words and time range of news
- output: news information in csv type 
- some key words of six dimensions

 |就业	|投资	|物价	|
 | ------ | ------ | ------ |
 |就业率，失业率，就业形势，就业压力，就业前景，就业满意度，求职压力等 	|市场情绪，投资意愿，投资热情，投资情绪等 |通胀预期，通胀压力，物价涨幅，居民物价，物价走势，物价指数，物价满意度等|

 |生活状况	|经济	|购房|
 | ------ | ------ | ------ |
 |居民收入，居民幸福，消费意愿，居民消费支出，居民消费能力，生活满意度，居民生活质量等 |经济形势，宏观经济，目前经济，中国经济前景，宏观经济数据，中国的经济发展态势，宏观经济运行等|楼市成交量，购房压力，购房成本，楼市热度，楼市前景，购房意愿，居民楼市信心，购房支出，房价满意度，房价预期等|

### code explanation

#### 1. The entire crawler code was written as a class named <i>baidu_news</i>.

```python
# initial variables: 
# search keyword,begin time,end time, browser header,crawl title and abstract of news or only title
def __init__(self,word,bt_ymd,et_ymd,headers):
    self.word = word
    self.bt_ymd = bt_ymd
    self.et_ymd = et_ymd
    self.headers = headers
    self.mode = "title"
 ```


```python
# an example 
bt_ymd = "2018-07-01"
et_ymd = "2019-06-30"
headers = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36',
    'Host':'www.baidu.com',
    'Cookie':'BIDUPSID=7C2C739A7BA8C15B187303565C792CA0; PSTM=1509410172; BD_UPN=12314753; BAIDUID=70698648FD1C0D4909420893B868092B:FG=1; MCITY=-%3A; BDORZ=B490B5EBF6F3CD402E515D22BCDA1598; BDUSS=N5eGZLbWZ5eWNuSTc5TUpobUIxWXU3ZmpoQklSUGJNZ1R5cnIwLTd6LWdBRVJkRVFBQUFBJCQAAAAAAAAAAAEAAAA1izQO0sDIu9DS0MQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAKBzHF2gcxxdZ1; pgv_pvi=166330368; ___wk_scode_token=Ct4MH%2FuNEgumb9NGCk8o1Aj%2BjCUcLU2ClmExi0Qz51M%3D; BD_CK_SAM=1; PSINO=7; BDRCVFR[PaHiFN6tims]=9xWipS8B-FspA7EnHc1QhPEUf; BDRCVFR[C0p6oIjvx-c]=mk3SLVN4HKm; BD_HOME=1; BDRCVFR[feWj1Vr5u3D]=I67x6TjHwwYf0; delPer=0; H_PS_PSSID=; sug=3; sugstore=1; ORIGIN=2; bdime=0; H_PS_645EC=f263%2FGdJfRrManRLCydAHWcUoMS0z2QF37c4uymvBok2x75KBHmMBsxhzWSqrwKXegg9lBNs; BDSVRTM=104'}
     #初始化
```

#### 2. Get the url of the search result of the keyword

```python
import time
from urllib.parse import urlencode

def get_url(self,page):#第几页
    bt = self.bt_ymd+" 00:00:00"
    et = self.et_ymd+" 00:00:00"
    bts = int(time.mktime(time.strptime(bt, "%Y-%m-%d %H:%M:%S")))#时间戳
    ets = int(time.mktime(time.strptime(et, "%Y-%m-%d %H:%M:%S")))
    
    pn = 20*(page-1)# 页码对应：0 20 40 60
    if self.mode=="news":
        qword = urlencode({'word': self.word.encode('utf-8')})
        url = "http://news.baidu.com/ns?%s&pn=%d&cl=2&ct=1&tn=newsdy&rn=20&ie=utf-8&bt=%d&et=%d"%(qword,pn,bts,ets)
    if self.mode=="title": 
        qword = self.word
        #url patern may have to be changed as the web page renew
        url = "https://www.baidu.com/s?tn=news&rtt=1&bsst=1&cl=2&wd="+qword+"&medium=1&gpc=stf%3D"+str(bts)+"%2C"+str(ets)+"%7Cstftype%3D2&pn="+str(pn)
    return url
```


#### 3. Jump to the obtained urls and crawl the new information
```python
import requests
from lxml import etree

def crawl(self,word):
    self.word = word
    i = 1
    is_nextpage=True
    newsData = pd.DataFrame()
    while is_nextpage:
        print("--------------正在爬取【%s】第%d页新闻----------------"%(self.word,i))
        url = self.get_url(i)
        print(url)

        result = requests.get(url,timeout=60,headers=self.headers)
        if result.status_code==200:
            print("\n请求成功")
        result.encoding = 'utf-8'
        selector = etree.HTML(result.text)  
        if self.mode=="news":

            for item in selector.xpath('//*[@class="result"]'):
    #            item = selector.xpath('//*[@class="result"]')[0]

    # news information includes news title,publish date,publish time and original web page 

                newsdict = {"title":[0],"date":[0],"time":[0],"source":[0],
                            "abstract":[0],"href":[0]}
                onenews = pd.DataFrame(newsdict)
                
                onenews["title"] = item.xpath('h3/a')[0].xpath("string(.)").strip()
                print(onenews["title"])
                onenews["href"] = item.xpath('h3/a/@href')[0]
                info = item.xpath('div')[0].xpath("string(.)")
                onenews["source"] , onenews["date"] , onenews["time"]= info.split()[:3]
                onenews["abstract"] = " ".join(info.split()[3:len(info.split())-1])
                newsData = newsData.append(onenews)
        if self.mode=="title":
            for item in selector.xpath('//*[@class="result"]'):
#                item = selector.xpath('//*[@class="result"]')[0]
                newsdict = {"title":[0],"date":[0],"time":[0],"source":[0],"href":[0]}
                onenews = pd.DataFrame(newsdict)
                
                onenews["title"] = item.xpath('h3/a')[0].xpath("string(.)").strip()
                onenews["href"] = item.xpath('h3/a/@href')[0]
                info = item.xpath('div')[0].xpath("string(.)")
#                print(info)
                #如果新闻是今天发的，则会显示“X小时前”，则日期改成今天
                if len(re.findall(r"小时前",info.split()[1]))>0:
                    onenews["source"] = info.split()[0]
                    nowtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                    onenews["date"] = "%s年%s月%s日"%tuple((nowtime.split()[0].split("-")))
                    onenews["time"] = nowtime.split()[1][:5]#只取分秒            
                else:
                    onenews["source"] , onenews["date"] , onenews["time"]= info.split()[:3]
                newsData = newsData.append(onenews)
        page_info = selector.xpath('//*[@id="page"]/a[@class="n"]/text()')
        print(page_info)
        if len(page_info)>=1 and "下一页>" in page_info:
            is_nextpage=True
            i=i+1
        else:
            is_nextpage=False
    newsData["keyword"] = self.word
    newsData.to_csv(self.word+"_"+self.bt_ymd+"_"+self.et_ymd+"_"+self.mode+".csv",index = False,encoding = "gb18030")

```

#### 4. Loop through 6 categories of news, and parallelly crawl news of same category

```python
import pandas as pd
from multiprocessing import Pool 

keywords = pd.read_csv("E:/graduate/Paper/code/keywords.csv",encoding = "gbk") 

for j in range(6):
    cl = keywords["class"][j]
    para = keywords["keywords"][j].split(",")
    full_class_name = os.path.join("E:/graduate/Paper/renew_data",cl)
    if not os.path.exists(full_class_name):
        os.makedirs(full_class_name) 
    os.chdir(full_class_name)
    
# we can also type the search keyword directly in the code
#para = ["中国贫富差距"]

# parallel        
    p=Pool(8)
    p.map(baidu_news_crawl.crawl,para)      
    p.close()
    p.join()
    print(cl +"新闻爬取完成，请打开【"+os.getcwd()+"】查看详情")

```
For more information about this project, please visit my [github](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index).


