---
title: 利用xpath爬取网络新闻
author: Situ
layout: post
categories: [big data]
tags: [爬虫,xpath]
---

<font face="仿宋" >用xpath并行爬取在[新浪新闻](search.sina.com.cn)中用关键词搜索的新闻摘要及全文。</font>

1. ```crawl```函数爬取搜索页面的元素：标题、发布日期及时间、摘要、链接、原网站名称

2. 由于可能爬取内容为空，导致程序中断，因此遇到空内容则以“”代替

```python
from urllib.parse import urlencode
from multiprocessing import Pool
from lxml import etree
import pandas as pd
import os
import requests
import re
import numpy as np

def crawl(para):
    compRawStr = para[0]
    startpage = para[1]
    n = para[2]
    d = {'q': compRawStr.encode('gbk')}
    word = urlencode(d)
    newsData = pd.DataFrame()
    for i in list(range(startpage,startpage+n)):
        print("--------------正在爬取第"+str(i)+"页新闻----------------")
#        url = 'http://search.sina.com.cn/?%s&range=all&c=news&sort=rel&num=20&col=1_7&page=%s' % (word, str(i))
        #按相关度排序，指定了时间范围
        url = "http://search.sina.com.cn/?c=news&%s&range=all&time=custom&stime=2018-04-16&etime=2018-05-19&num=10&sort=rel&page=%s" % (word, str(i))
        result = requests.get(url)
        result.encoding = 'gbk'
        selector = etree.HTML(result.text)  
        for item in selector.xpath('//*[@id="result"]/div/div'):
            newsdict = {"title":[0],"date":[0],"time":[0],"source":[0],
                        "abstract":[0],"detail":[0],"href":[0],"origin_url":[0]}
            onenews = pd.DataFrame(newsdict)
#            onenews = pd.DataFrame(np.zeros(8).tolist(),
#                                   columns=["title","date","time","source","abstract","detail","href","orgin_url"])
            try:
                onenews["title"] = item.xpath('h2/a')[0].xpath("string(.)")
            except:
                onenews["title"] = ""
            print(onenews["title"][0])
            try:
                onenews["abstract"]  = item.xpath('p')[0].xpath("string(.)")
            except:
                onenews["abstract"] = ""
            try:
                otherinfo = item.xpath('h2/span/text()')[0]
            except:
                otherinfo = "NA NA NA"
            onenews["source"] , onenews["date"] , onenews["time"]  = otherinfo.split()
            try:
                onenews["href"]  = item.xpath('h2/a/@href')[0]
            except:
                onenews["href"]  = ""
            onenews["origin_url"] ,onenews["detail"] = crawl_con(onenews["href"][0])
            newsData = newsData.append(onenews)
    newsData.to_csv(compRawStr+"_"+str(startpage)+"_"+str(n)+"相关新闻.csv",index = False,encoding = "gb18030")

```

3. ```crawl_con```函数在新闻详情页，爬取全文、原文链接，并嵌套进```crawl```函数中
```python
def crawl_con(href):
    if href!="":
        site = requests.get(href)
        site=site.content
        response = etree.HTML(site)
        try:
            origin_url = response.xpath('//*[@id="top_bar"]/div/div[2]/a/@href')[0] 
        except:
            origin_url =""
        detail = "\n".join(response.xpath('//div[@class="article"]/div/p/text()'))+\
                "\n".join(response.xpath('//div[@class="article"]/p/text()'))+\
                "\n".join(response.xpath('//div[@class="article"]/div/div/text()'))+\
                "\n".join(response.xpath('//*[@id="artibody"]/p/text()'))
        detail = re.sub('\u3000', '', detail)    #全角的空白符
        return origin_url,detail
    else:
        return "",""

```

4. 多个关键词并行爬取 
```python
def main():
    os.chdir("")
    print('请输入您想爬取内容的关键字：')
    compRawStr = input('关键字1 关键字2： \n')     #键盘读入 多个关键字则用空格隔开
    PageN = input('起始页,页数： \n')     #如键盘读入 1,5 1,5
    print('正在爬取“' + compRawStr.capitalize()+ '”有关新闻!')
    comp = compRawStr.split()
    pn = PageN.split()
    para = []
    if (len(comp)==len(pn)) & (len(comp)>1):#多个关键词
        l = len(comp) #多进程爬取不同搜索词新闻
        for i in range(l):
            para.append([comp[i]]+[int(c) for c in pn[i].split(",")])   
            #[['中美贸易战', 1, 5], ['外交部', 1, 5]]
#        crawl(comp[0]) #爬取成功会输出title
        #crawl(['中美贸易战', 1, 5])
    if (len(comp)==len(pn)) & (len(comp)==1):#一个关键词，页面并行爬取
        l=3
        sp,n = [int(i) for i in pn[0].split(",")]
        sep = int(round(n/l,0))
        para = [[comp[0],sp,sep],[comp[0],sp+sep,sep],[comp[0],sp+2*sep,n-sp-2*sep]]
    p=Pool(l)
    p.map(crawl,para)       #爬取4页内容
    p.close()
    p.join()
    print("爬取成功，请打开"+os.getcwd()+"查看详情")

if __name__=='__main__':
    main()
```

5. 可改写成类的形式，方便多次调用，参见[TradeWarCrawl - class.py](https://github.com/Snowing-ST/Statistical-Case-Studies/blob/master/Lab6%20Scraping%20with%20xpath/TradeWarCrawl%20-%20class.py)



以上完整代码欢迎访问我的[github](https://github.com/Snowing-ST/Statistical-Case-Studies/tree/master/Lab6%20Scraping%20with%20xpath)！


