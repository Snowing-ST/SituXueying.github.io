---
title: 基于有标签的新闻计算网络新闻情感指数
author: Situ
layout: post
categories: [big data]
tags: [文本分类,NLP,deep-learning,Master Thesis]
---

<font face="仿宋" >基于深度学习文本分类的网络新闻情感指数编制（七）<br>基于有标签的新闻计算网络新闻情感指数</font>

## Main step 7:<center>computing Online News Sentiment Index </center>
### description:
- compute six sub-indexes according to all the tagged news
- synthesize Online News Sentiment Index

### code explanation:

#### 1. compute index of each quarter according to the tagged news
```python
def get_index(data,prefix):
    """从打好标签的文本中计算出每个季度的指数"""
    if "year" not in data.columns or "quarter" not in data.columns:
        time = pd.to_datetime(data["date"],format="%Y年%m月%d日")
        data["year"] = list(pd.DatetimeIndex(time).year)
        data["quarter"] = list(pd.DatetimeIndex(time).quarter)

    grouped = data.groupby(["year","quarter"])
    get_index = lambda x:(sum(x==1)-sum(x==-1))/(sum(x==1)+sum(x==-1))
#    grouped.agg({"label":["count"]}) #每个季度多少条
    consumer_index = grouped.agg({"label":[get_index]}) #季度指数
    consumer_index.columns = consumer_index.columns.droplevel(1) #去除多重索引
    
    consumer_index["index"+"_"+prefix] = 100+consumer_index["label"]
    
    return consumer_index
```

#### 2. compute six sub-indexes
- input:```test_CNN.csv```,```train.csv``` 
- output:```CNN指数.csv```,including columns named ```index_all```,```index_test```,```index_train```

```python
#计算index_train index_test index_all及其相关系数
consumer_index_file = os.path.join("./raw_data/"+FLAGS.aspect,FLAGS.aspect+"CNN指数.csv")
if os.path.isfile(consumer_index_file):
    print("已存在该指数，直接读取")
    consumer_index = pd.read_csv(consumer_index_file,encoding="gb18030",engine="python")
else:
    data_train = pd.read_csv(os.path.join("./raw_data/"+FLAGS.aspect,"train.csv"),sep = ",",encoding="gb18030",engine="python")
    data = pd.concat([data_train,data_test],axis=0)
    consumer_index_all = get_index(data,prefix="all")
    consumer_index_train = get_index(data_train,prefix="train")
    consumer_index_test = get_index(data_test,prefix="test")
    consumer_index = pd.concat([consumer_index_all["index_all"],consumer_index_train["index_train"],consumer_index_test["index_test"]],axis=1)
    consumer_index.reset_index() #索引变列
    consumer_index.dropna(inplace=True)
    consumer_index.to_csv(consumer_index_file,encoding="gb18030")
print("训练集指数与测试集指数的相关系数为：%0.4f"%(consumer_index[["index_train","index_test"]].corr().ix[0,1]))
print("训练集指数与全集指数的相关系数为：%0.4f"%(consumer_index[["index_train","index_all"]].corr().ix[0,1]))
consumer_index[["index_train","index_test"]].plot()
```
#### 3. synthesize Online News Sentiment Index
- input:```CNN指数.csv``` of each class
- output:  ```CNN总指数.csv```

```python
def batch_read_index(path,ALL_INDEX_NAME):
    classes = os.listdir(path)
    all_index = pd.DataFrame()
    for asp in classes:
        sp = os.path.join(path,asp,asp+ALL_INDEX_NAME)
        temp_df = pd.read_csv(sp,encoding="gb18030",engine =  "python")
        print(asp)
        temp_df.set_index(["year","quarter"], inplace=True) #列变索引
        temp_df.columns = temp_df.columns.map(lambda x:asp+x) #更改列名
        all_index = pd.concat([all_index,temp_df],axis=1)
        
    
    all_index.reset_index(inplace=True) #索引变列
    train_list = [tr for tr in all_index.columns if len(re.findall(r"train",tr))>0]
    all_index["index_train"] = np.sum(all_index.ix[:,train_list],axis=1)-500
    
    test_list = [tr for tr in all_index.columns if len(re.findall(r"test",tr))>0]
    all_index["index_test"] = np.sum(all_index.ix[:,test_list],axis=1)-500
    
    all_list = [tr for tr in all_index.columns if len(re.findall(r"all",tr))>0]
    all_index["index_all"] =  np.sum(all_index.ix[:,all_list],axis=1)-500
    all_index.to_csv(ALL_INDEX_NAME,encoding="gb18030",index=False)

path = "./raw_data"    
ALL_INDEX_NAME = "CNN总指数.csv"
batch_read_index(path,ALL_INDEX_NAME)
```

