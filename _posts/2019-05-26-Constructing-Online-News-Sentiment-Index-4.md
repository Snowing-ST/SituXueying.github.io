---
title: 用向量空间模型进行文本分类
author: Situ
layout: post
categories: [big data]
tags: [文本分类,NLP,deep-learning,Master Thesis]
---

<font face="仿宋" >基于深度学习文本分类的网络新闻情感指数编制（四）<br>文本分类基准模型：向量空间模型</font>
<style>
    body {font-family: "华文中宋"}
</style>

## Main step 4:<center>traditional text classification with VSM</center>
### description:
- text representation: TF-IDF
- classification model ：logistic regression、Naïve Bayes、SVM
- best model ：SVM, accuracy:77% (as basic line)

### code explanation:

#### 1. word representation
- two approaches to transform text into matrix:
    1. tf-idf 2. one-hot
- input:two columns name ```word_seg``` and ```label``` of  ```data_train``` and ```data_test```
- output: ```data_train_tfidf```, ```tags``` ,```data_test_tfidf```

```python
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

def vectorize(data_train,data_test,word_name = "word_seg",tag_name="label",vectype = "tf-idf",ngram_range=(0,1),max_features=None,min_df=2):
    """
    文本表示：tf-idf or one-hot
    """
#    data.isnull().any()#哪些列存在缺失值
    words = data_train[word_name].tolist()+data_test[word_name].tolist()
    if vectype == "tf-idf":
        transformer=TfidfVectorizer(ngram_range=ngram_range,max_features=max_features,min_df=min_df)
        data_tfidf=transformer.fit_transform(words)
        transformer2 = TfidfVectorizer(vocabulary = transformer.vocabulary_)
        data_train_tfidf=transformer2.fit_transform(data_train[word_name])
        data_test_tfidf=transformer2.fit_transform(data_test[word_name])          
        return data_train_tfidf,data_train[tag_name].tolist(),data_test_tfidf
    if vectype == "one-hot":
        transformer=CountVectorizer(ngram_range=ngram_range,max_features=max_features,min_df=min_df)
         
        data_onehot=transformer.fit_transform(words)
        transformer2 = CountVectorizer(vocabulary = transformer.vocabulary_)
        data_train_onehot=transformer2.fit_transform(data_train[word_name])
        data_test_onehot=transformer2.fit_transform(data_test[word_name])           
        return data_train_onehot,data_train[tag_name].tolist(),data_test_onehot

data_train_tfidf, tags ,data_test_tfidf = vectorize(data_train,data_test,word_name = "word_seg",tag_name="label",vectype = "tf-idf",ngram_range=(0,1),max_features=None,min_df=1)
```

#### 2. several machine learning approaches
1. naive bayes 2. logistic regression 3. SVM

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression  
from sklearn import svm
from sklearn.model_selection import GridSearchCV,cross_val_score, cross_val_predict,KFold

def train_NB(data_tfidf,tags,cv):
    grid_values = {'alpha':np.arange(0.1,1.1,0.1)} # Decide which settings you want for the grid search. 

    grid = GridSearchCV(MultinomialNB(), 
                        grid_values, scoring = "accuracy", cv = cv) 
    grid.fit(data_tfidf,tags) 
    grid.grid_scores_
    print("【NB】The best parameters are %s with a score of %0.4f"
          % (grid.best_params_, grid.best_score_))
    return grid.best_estimator_

def train_lg(data_tfidf,tags,cv):
    grid_values = {'tol':[0.001,0.1,1],'C':range(1,10,2)} # Decide which settings you want for the grid search. 

    grid = GridSearchCV(LogisticRegression(penalty="l2", dual=True), 
                        grid_values, scoring = "accuracy", cv = cv,n_jobs=7) 
    grid.fit(data_tfidf,tags) 
    grid.grid_scores_
    print("【lg】The best parameters are %s with a score of %0.4f"
          % (grid.best_params_, grid.best_score_))
    return grid.best_estimator_

def train_SVM(data_tfidf,tags,cv):#5,1,1 #4,0.9,1 0.8238
    "调参影响大。学习率越小，所需迭代次数越多"
    grid_values = {'C':[1,4,7],'gamma':[0.1,0.5,0.9]} # Decide which settings you want for the grid search. 

    grid = GridSearchCV(svm.SVC(kernel='rbf',tol=1, degree=3, coef0=0.0, shrinking=True, probability=False),
                        grid_values, scoring = "accuracy", cv = cv) 
    grid.fit(data_tfidf,tags) 
    grid.grid_scores_
    print("【SVM】The best parameters are %s with a score of %0.4f"
          % (grid.best_params_, grid.best_score_))
    return grid.best_estimator_

#模型比较
cv = KFold(n_splits=10, shuffle=True, random_state=1994)
NB = train_NB(data_train_tfidf, tags,cv)
lg = train_lg(data_train_tfidf, tags,cv)
SVM = train_SVM(data_train_tfidf, tags,cv)
```

For more information about this project, please visit my [github](https://github.com/Snowing-ST/Construction-and-Application-of-Online-News-Sentiment-Index).
