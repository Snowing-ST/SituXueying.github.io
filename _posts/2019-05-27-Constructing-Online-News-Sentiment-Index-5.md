---
title: Constructing Online News Sentiment Index 5
author: Situ
layout: post
categories: [big data]
tags: [deep learning, text classification, NLP]
---

<font face="仿宋" >基于深度学习文本分类的网络新闻情感指数编制（五）<br>深度学习文本分类模型之用word2vec做文本表示</font>
<style>
    body {font-family: "华文中宋"}
</style>

## Main step 5:<center>text classification with deep learning:<br>text representation using word2vec</center>
### description:
-  using word2vec pretrained by [Chinese Word Vectors](https://github.com/Embedding/Chinese-Word-Vectors)

### code explanation:
#### 1. aggregate all the news
- input:```all_df.csv``` of six classes
- output: ```all_news_title.csv``` saved in ```TRAIN_DATA``` folder

```python
import text_preprocess as tp
from collections import Counter
import pickle
import re

def get_all_news(TRAIN_DATA):
    """将所有方面的all_df.csv合并"""
    path = "./raw_data/"
    aspect = os.listdir(path)
    all_news_title = pd.DataFrame()
    for asp in aspect:
        subpath = os.path.join(path,asp)
        temp_df_name = [f for f in os.listdir(subpath) if len(re.findall(r"all_df.csv",f))>0]
        
        temp_df = pd.read_csv(os.path.join(subpath,temp_df_name[0]),encoding="gb18030",engine =  "python")
        all_news_title = pd.concat([all_news_title,temp_df],axis=0,ignore_index=True)
    all_news_title.to_csv(TRAIN_DATA,encoding="gb18030",index=False)    
    return all_news_title
```
#### 2. obtain dictionary including all the words in news
- input:  ```all_df.csv```
- text prepossessing using function in ```text_preprocess.py```
- obtain word count frequency, remove low frequency words
- output:dictionary saved in ```DICTIONARY_DIC``` folder

```python
from collections import Counter

def get_news_dic(TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW):
    """
    从所有新闻训练样本得到字典
    """
    dic = [] #存放词
    print('正在加载字典……')
    # 统计数据包总条数
    if os.path.isfile(DICTIONARY_DIC):#如果已有所有新闻生成的词典，则直接读取
        with open(DICTIONARY_DIC, "rb") as f:
            print("已存在由所有新闻生成的字典，直接读取")
            dic = pickle.load(f) 
    else:
        if os.path.isfile(TRAIN_DATA):
            print("已存在所有新闻合集，直接读取")
            all_news_title = pd.read_csv(TRAIN_DATA,encoding="gb18030",engine="python")
        else:
            all_news_title = get_all_news(TRAIN_DATA)
        
        tolal_line_num = len(all_news_title)
        print("共有新闻有%d条"%(tolal_line_num))
        #调用文本预处理函数
        x_train = tp.convert_doc_to_wordlist(all_news_title["title"],tool = "jieba",cut_all=False,mode ="accuracy")
        whole_text = []
        for line in x_train:
            whole_text.extend(line)

        frequency_dict = Counter(whole_text)    
#        frequency_dict = sorted(Counter(whole_text).items(), key = operator.itemgetter(1), reverse=True) #=True 降序排列
            
        for word in frequency_dict:
            if WORD_FREQUENCY_LOW < frequency_dict[word]:#去掉低频词
                dic.append(word)
        
        with open(DICTIONARY_DIC, 'wb') as f:
            pickle.dump(dic, f) # 把所有新闻的字典保存入文件

    print('字典加载完成,去除低频词后字典长度为%d'%(len(dic)))
    return dic
```

#### 3. using pretrained word2vec to replace the dictionary
- input:```DICTIONARY```
- output: ```pha``` ,a pretained word2vec matrix of ```DICTIONARY``` saved in ```WORD2VEC_SUB```

```python
import fileinput
import sys

def view_bar(text, num, total):
    """优化进度条显示"""
    rate = num / total
    rate_num = int(rate * 100)
    r = '\r' + text + '[%s%s]%d%%' % ("=" * rate_num, " " * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()

def get_word2vec(WORD2VEC_DIC, # 已预训练好的词向量文件地址
                 WORD2VEC_SUB, # 替换好的字典
                 TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW):# get_news_dic的变量
    """
    采用预训练好的部分词向量。仅使用部分，是为了节省内存。
    
    1. 遍历已训练好的词向量文件
    2. 替换掉本例词典中存在词的词向量
    
    在神经网络训练时，再用这个WORD2VEC_SUB将文本转化为词向量
    """
    print("正在加载预训练词向量……")
    
    if os.path.isfile(WORD2VEC_SUB): #如果之前已经生成过替换后的词向量
        print("已存在替换好的词向量，直接读取")
        with open(WORD2VEC_SUB, "rb") as f:
            pha = pickle.load(f)  
    else:
        if os.path.isfile(DICTIONARY_DIC):#如果之前已经生成过所有新闻的字典
            print("已存在由所有新闻生成的字典，直接读取")
            with open(DICTIONARY_DIC, "rb") as f:
                DICTIONARY = pickle.load(f)   
        else:
             # 生成利用所有新闻归纳出的字典
            DICTIONARY = get_news_dic(TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW)
        
        # 1. 生成[词向量个数, 300维]的随机均匀分布
        pha = np.random.uniform(-1.0, 1.0, [len(DICTIONARY), 300]) 
        # 2. 使用预训练好的词向量替换掉随机生成的分布
        if os.path.isfile(WORD2VEC_DIC):
            with fileinput.input(files=(WORD2VEC_DIC), openhook=fileinput.hook_encoded('UTF-8')) as f:
                count = 0
                for line in f:
                    word_and_vec = line.split(' ')
                    word = word_and_vec[0]
                    vec = word_and_vec[1:301]
                    
                    if word in DICTIONARY:#替换
                        pha[DICTIONARY.index(word)] = vec
                        #进度条
                        count += 1
                        if count % 36000 == 0:
                            # print('处理进度：', count / total_line_num * 100, '%')
                            view_bar('处理进度：', count,364991) #别人训练好的词向量有36万词 
            with open(WORD2VEC_SUB, 'wb') as f:
                pickle.dump(pha, f) # 把所有新闻的词向量保存入文件   
    print("预训练词向量加载完毕。")
    return pha
```
#### 4. transform news of each class into 3D matrix
- a word is a vector; one sample of news is a matrix; all the samples of news is a 3D matrix whose height is equal to maximum number of words in one sample news
- input: ```pha```,```DICTIONARY``` 
- output:```x_train_3d_vecs``` saved in ```ASPECT_WORD2VEC``` folder

```python

def get_asp_w2v(ASPECT_WORD2VEC,data_train,WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW,set_sequence_length):
    """
    将每个类别的新闻文本，进行文本预处理后，用替换好的所有新闻词典，生成词向量3d矩阵
    """
    if os.path.isfile(ASPECT_WORD2VEC):
        print("已存在该类别词向量3d矩阵，直接读取")
        with open(ASPECT_WORD2VEC, "rb") as f:
            x_train_3d_vecs = pickle.load(f)
            max_len = x_train_3d_vecs.shape[2]
    else:
        #文本预处理
        x_train = tp.convert_doc_to_wordlist(data_train["title"],tool = "jieba",cut_all=False,mode ="accuracy")
        #计算最大句子长度
        if set_sequence_length!=None:#直接指定最大句子长度
            max_len = set_sequence_length
        else:
            max_len = max(list(map(len,x_train)))
        #补零
        def padding_sent(sent,max_len=max_len ,padding_token="空"):
            """给不满足句子长度的句子补零"""
            if len(sent) > max_len:
                sent = sent[:max_len]
            else:
                sent.extend([padding_token] * (max_len - len(sent)))
            return sent
        x_train_pad = list(map(padding_sent,x_train))
        #读取词向量
        if os.path.isfile(WORD2VEC_SUB): #如果之前已经生成过替换后的词向量
            print("已存在替换好的词向量，直接读取")
            with open(WORD2VEC_SUB, "rb") as f:
                pha = pickle.load(f)
        else:
            pha = get_word2vec(WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW)
        #读取词典
        if os.path.isfile(DICTIONARY_DIC):
            print("已存在由所有新闻生成的字典，直接读取")
            with open(DICTIONARY_DIC, "rb") as f:
                DICTIONARY = pickle.load(f)
        else:
            DICTIONARY = get_news_dic(TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW) 
        #生成该类别新闻词向量3d矩阵
        x_train_3d_vecs = np.array([convert_sent_to_mat(sent,pha,DICTIONARY) for sent in x_train_pad])
        x_train_3d_vecs.shape
        #保存该类别的词向量3d矩阵
        with open(ASPECT_WORD2VEC, 'wb') as f:
            pickle.dump(x_train_3d_vecs, f) # 保存入文件       
    return x_train_3d_vecs,max_len
```

#### 5. Complete process to obtian word2vec 3D matrix
- input: ```train.csv```or ```test.csv``` 
- output: 3D matrix ```x_text```

```python
def load_data_and_labels(path,aspect,file_name,set_sequence_length=None,istest=False):

#    aspect="经济"  
#    file_name = "train.csv"
#    可能是训练集，可能是测试集，只是导入数据而已
#    path = "./raw_data/"
    data_train = pd.read_csv(os.path.join(path+aspect,file_name),sep = ",",encoding="gb18030",engine="python")
#    x_text = [w.split() for w in data_train["word_seg"]]
#    x_text,sequence_length = get_word2vec(x_text,size=embedded_size,min_count =min_count ,window = 5,method="padding",seq_dim=2)
    if istest:#测试集无标签
        ASPECT_WORD2VEC = path+aspect+"/%s%s_word2vec.pickle"%(aspect,"test")
        y = np.zeros((len(data_train)),int)
    else:
        ASPECT_WORD2VEC = "./code/%sword2vec.pickle"%aspect
        y =  np.array(pd.get_dummies(data_train["label"]))

    WORD2VEC_DIC = './code/pretrained word2vec/sgns.sogou.word'      # Chinese Word Vectors提供的预训练词向量
    TRAIN_DATA = "./code/all_news_title.csv" #所有标题合集
    DICTIONARY_DIC ="./code/all_news_original_dic.pickle"      # 存放总结出的字典，以节省时间
    WORD2VEC_SUB = "./code/word2vec_sub.pickle" # 替换后的词向量地址
    WORD_FREQUENCY_LOW = 0
    

    x_text,sequence_length = get_asp_w2v(ASPECT_WORD2VEC,data_train,WORD2VEC_DIC,WORD2VEC_SUB, TRAIN_DATA,DICTIONARY_DIC,WORD_FREQUENCY_LOW,set_sequence_length)
    
    return x_text, y,sequence_length
```
For more information about this project, please visit my [github](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index).