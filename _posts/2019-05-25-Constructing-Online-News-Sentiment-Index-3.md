---
title: 对爬取的新闻做文本预处理
author: Situ
layout: post
categories: [big data]
tags: [文本预处理,文本分类,NLP,deep-learning,Master Thesis]
---

<font face="仿宋" >基于深度学习文本分类的网络新闻情感指数编制（三）<br>对爬取的新闻做文本预处理</font>
<style>
    body {font-family: "华文中宋"}
</style>

## Main step 3:<center>text prepossessing</center>
### description:
- split ```all_df.csv``` into training set and testing set by using stratified sampling according to year and keywords
- <b><font color="blue">artificially label the news in training set with “positive”, “neutral” and "negative"</font></b>
- text prepossessing including word segment and removing stop words

### code explanation:
#### 1. split ```all_df.csv``` into training set and testing set 
- input: crawl data:```all_df.csv```
- output:```train.csv```,```test.csv```
- artificially label the news in training set with “positive”, “neutral” and "negative"

```python
from sklearn.model_selection import train_test_split
from multiprocessing import Pool
import time
import os
import re
import pandas as pd
    
def sampling(subpath):
    file_names = os.listdir(subpath)
    file_name = [f for f in file_names if len(re.findall(r"all_df.csv",f))>0][0] #只读取后缀名为csv的文件
    all_df = pd.read_csv(os.path.join(subpath,file_name),encoding="gb18030",engine="python")
    all_df["sample"] = all_df.apply(lambda line: line["keyword"]+"-"+str(line["year"]),axis=1)   
    x = all_df.drop("sample",1)
    label = all_df["sample"]
    x_train,x_test,_,_ = train_test_split(x,label,test_size=0.6,random_state=1994)
    x_train.to_csv(os.path.join(subpath,"train.csv"),encoding="gb18030",index=False) #需要人工打标签的
    x_test.to_csv(os.path.join(subpath,"test.csv"),encoding="gb18030",index=False)  

def main():

    os.chdir("E:/graduate/Paper/")
    path = "E:/graduate/Paper/raw_data"

    #只做一个类别
    #sp = "物价"
    #sampling(os.path.join(path,sp))

    #并行方法，同时生成所有类别的训练集测试集
    subpaths = [os.path.join(path,sp) for sp in os.listdir(path)]
    p=Pool(len(subpaths))
    p.map(sampling,subpaths)      
    p.close()
    p.join()      
```

#### 2. delete stop words according to ```stopWord.txt```
```python
def get_text(text):
    text = text.dropna() 
    len(text)
    text=[t.encode('utf-8').decode("utf-8") for t in text] 
    return text


def get_stop_words(file='./code/stopWord.txt'):
    file = open(file, 'rb').read().decode('utf8').split(',')
    file = [line.strip() for line in file]
    return set(file)                                         #查分停用词函数


def rm_tokens(words):                                        # 去掉一些停用词和完全包含数字的字符串
    words_list = list(words)
    stop_words = get_stop_words()
    for i in range(words_list.__len__())[::-1]:
        if words_list[i] in stop_words:                      # 去除停用词
            words_list.pop(i)
        elif words_list[i].isdigit():
            words_list.pop(i)
    return words_list
```
#### 2. delete punctuation and special characters
```python
def rm_char(text):

    text = re.sub('\x01', '', text)                        #全角的空白符  感觉问好 感叹号不应该删除
    text = re.sub('\u3000', '', text) 
    text = re.sub(']'," ", text) 
    text = re.sub('\['," ", text) 
    text = re.sub('"'," ", text) 
    text = re.sub(r"[\)(↓%·▲】&【]","", text) 
    text = re.sub(r"[\d（）《》〖〗><‘’“”""''.,_:|-…]"," ",text,flags=re.I)
    text = re.sub('\n+', " ", text)
    text = re.sub('[，、：。；——]', " ", text)
    text = re.sub(' +', " ", text)
    text = re.sub(';', " ", text)
    return text
```

#### 3. word segment
- three approaches of word segment:
    - jieba——the most universal one
    - [thulac](http://thulac.thunlp.org/), a efficient Chinese text segmentation tool developed by Tsinghua University
    - [pkuseg](https://github.com/lancopku/pkuseg-python), a small package for Chinese word segmentation developed by Peking University
- input: ```train.csv```,```test.csv```
- output: a column names ```word_seg``` of ```data_train``` and ```data_test```

```python
import jieba
import thulac 
import pkuseg

def convert_doc_to_wordlist(text, tool = "jieba",cut_all=False,mode = "accuracy"):
    text = get_text(text)
    sent_list = map(rm_char, text)                       # 去掉一些字符，例如\u3000
    if tool=="jieba":
        jieba.load_userdict("./code/dict.txt")
        if mode == "accuracy":
            word_2dlist = [rm_tokens(jieba.cut(part, cut_all=cut_all))
                       for part in sent_list]                     # 分词
        if mode == "search":
            word_2dlist = [rm_tokens(jieba.cut_for_search(part))
                       for part in sent_list]
    if tool=="thulac":
        thu1 = thulac.thulac(user_dict="./code/dict_thu1.txt",seg_only=True)  #只进行分词，不进行词性标注
        word_2dlist = [rm_tokens(thu1.cut(part, text=True).split()) for part in sent_list]
    if tool=="pku":
        seg = pkuseg.pkuseg(user_dict="./code/dict_thu1.txt")
        word_2dlist = [rm_tokens(seg.cut(part)) for part in sent_list]
    def rm_space_null(alist):
        alist = [s for s in alist if s not in [""," "]]
        return alist
    rm_space = [rm_space_null(ws) for ws in word_2dlist if len(ws)>0]
    return rm_space


aspect = "物价" #一次处理一个类别的新闻
path = "./raw_data/"+aspect  
file_name = "train.csv"
data_train = pd.read_csv(os.path.join(path,file_name),sep = ",",encoding="gb18030",engine="python")
data_test = pd.read_csv(os.path.join(path,"test.csv"),sep = ",",encoding="gb18030",engine="python")
data_train.dropna(inplace=True)
data_train.head()
data_train.shape
data_train["label"].value_counts()

full_data = [data_train,data_test]
for dataset in full_data:
    clean_text=convert_doc_to_wordlist(dataset["title"],tool="pku",cut_all=False,mode ="accuracy")
    dataset["word_seg"] = [" ".join(line) for line in clean_text]
```

#### 4. removing high or low frequency words (optional)
```python
from collections import Counter
import operator

def rm_low_high_freq(texts,low_freq=1,high_topK=10):#texts为包含多个句子的列表
    whole_text = []
    for doc in texts:
        whole_text.extend(doc.split())
    frequency_dict = Counter(whole_text)
    frequency_dict = sorted(Counter(whole_text).items(), key = operator.itemgetter(1), reverse=True) #=True 降序排列
#    print("the top %d wordcount is:\n" %(high_topK),frequency_dict[:high_topK],"/n")
    word_count = np.array(frequency_dict)
    print("原词典长度为%d"%len(word_count))
#    high_freq_w = [wc[0] for wc in word_count[:high_topK]]
    low_freq_w = word_count[word_count[:,1]==str(low_freq),0].tolist()
    dele_list = low_freq_w
    print("现词典长度为%d"%(len(word_count)-len(dele_list)))
#    dele_list = high_freq_w+low_freq_w
    rm_freq_texts = [[token for token in doc.split() if token not in dele_list] for doc in texts]
#    sum(np.array(list(map(len,rm_freq_texts)))==1)
    dele_num = np.where(np.array(list(map(len,rm_freq_texts)))<1)[0]
    #哪些新闻被删得只剩0 个或1个词
#    data.ix[dele_index,"title"]
#    data = data.drop(dele_num,inplace = False)
#    data = data.reset_index(drop=True)
    print("删除词数少于1的新闻%d条"%len(dele_num))
    new_texts = [" ".join(line) for line in rm_freq_texts if len(line)>0]

    return new_texts
```
#### 5. view top k words of each class(optional)
```python
from jieba.analyse import extract_tags

def view_keywords(data,word_name = "word_seg",tag_name="label",topK=20):
    "用jieba看每个标签的关键词"
    def get_kw(text):
        return extract_tags(text, topK=topK, withWeight=True, allowPOS=())
    text_groupbyLabel = [" ".join(data[word_name][data[tag_name]==i]) for i in  range(-1,2)]
    news_kw = list(map(get_kw,text_groupbyLabel))
    
    for j in range(len(news_kw)):
        print("\n第"+str(j+1)+"类新闻的关键词：\n")
        for i in range(len(news_kw[j])):
            print(news_kw[j][i])

view_keywords(data_train,word_name = "word_seg",tag_name="label")
```

For more information about this project,please visit my [github](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index).
