#!/usr/bin/env python
# coding: utf-8

# # 评价情感分析
# 
# 通过3W条评论信息以及评分作为训练数据，前面的分析我们得知*样本很不均衡*。整体思路就是：文本特征处理(分词、去停用词、TF-IDF)—机器学习建模—模型评价。
# 
# 
# ### 数据读入和探索

# In[3]:


import pandas as pd
from matplotlib import pyplot as plt
import jieba
data = pd.read_csv('data.csv')
data


# ### 构建标签值
# 
# 评分分为1-5分，1-2为差评，4-5为好评，3为中评，因此我们把1-2记为0,4-5记为1,3为中评，对我们的情感分析作用不大，丢弃掉这部分数据，但是可以作为训练语料模型的语料。我们的情感评分可以转化为标签值为1的概率值，这样我们就把情感分析问题转为文本分类问题了。

# In[4]:


#构建标签特征 label值
def zhuanhuan(score):
    if score > 3:
        return 1
    elif score < 3:
        return 0
    else:
        return None
    
#特征值转换
data['target'] = data['stars'].map(lambda x:zhuanhuan(x))
data_model = data.dropna()
data_model.isnull().sum()                      #统计缺失值数量


# ### 文本特征处理
# 
# 中文文本特征处理，需要进行中文分词，jieba分词库简单好用。接下来需要过滤停用词，网上能够搜到现成的。最后就要进行文本转向量，有词库表示法、TF-ID
# 
# 这里我们使用sklearn库的TF-IDF工具进行文本特征提取。

# In[5]:


#中文文本特征处理，jieba分词、停用词过滤粗合理，文本转向量、
#切分测试集、训练集
from sklearn.model_selection import train_test_split
#随机种子设置为3
x_train, x_test, y_train, y_test = train_test_split(data_model['cus_comment'], data_model['target'], test_size=0.25)

#引入停用词
infile = open("stopwords.txt",encoding='utf-8')
stopwords_lst = infile.readlines()
stopwords = [x.strip() for x in stopwords_lst]

#中文分词
def fenci(train_data):
    words_df = train_data.apply(lambda x:' '.join(jieba.cut(x)))
    return words_df


# In[6]:


x_train.head()


# In[7]:


#使用TF-IDF进行文本转向量处理
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(stop_words=stopwords, max_features=3000, ngram_range=(1,2))
tv.fit(x_train)


# ### 机器学习建模
# 
#使用文本分类的经典算法朴素贝叶斯算法，而且朴素贝叶斯算法的计算量较少。特征值是评论文本经过TF-IDF处理的向量，标签值评论的分类共两类，好评是1，差评是0。情感评分为分类器预测分类1的概率值。

# In[8]:


#计算分类效果的准确率
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score, f1_score
classifier = MultinomialNB()
classifier.fit(tv.transform(x_train), y_train)
classifier.score(tv.transform(x_test), y_test)


# In[9]:


#计算分类器的AUC值
y_pred = classifier.predict_proba(tv.transform(x_test))[:,1]
roc_auc_score(y_test,y_pred)


# In[10]:


#计算一条评论文本的情感评分
def ceshi(model,strings):
    strings_fenci = fenci(pd.Series([strings]))
    return float(model.predict_proba(tv.transform(strings_fenci))[:,1])


# In[30]:


#
test1 = '不愧为大师中的大师,非常好,好,好，非常好,很好,很好' #5星好评
test2 = '真的不喜欢，不好看，没感觉，' #1星差评
print('好评实例的模型预测情感得分为{}\n差评实例的模型预测情感得分为{}'.format(ceshi(classifier,test1),ceshi(classifier,test2)))


# 可以看出，准确率和AUC值都非常不错的样子，但点评网上的实际测试中，5星好评模型预测出来了，1星差评缺预测错误。为什么呢？我们查看一下**混淆矩阵**

# In[18]:


from sklearn.metrics import confusion_matrix
y_predict = classifier.predict(tv.transform(x_test))
cm = confusion_matrix(y_test, y_predict)
cm


# 

# ### 过采样（SMOTE算法）
# 
# SMOTE（Synthetic minoritye over-sampling technique,SMOTE），是在局部区域通过K-近邻生成了新的反例。相较于简单的过采样，SMOTE降低了过拟合风险
# 

# In[19]:


#使用SMOTE进行样本过采样处理
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0)
x_train_vec = tv.transform(x_train)
x_resampled, y_resampled = oversampler.fit_sample(x_train_vec, y_train)


# In[20]:


#原始的样本分布
y_train.value_counts()


# In[21]:


#经过SMOTE算法过采样后的样本分布情况,进行插值处理
pd.Series(y_resampled).value_counts()
#正负样本的比例为1:1


# In[22]:


#使用过采样样本(SMOTE)进行模型训练，并查看准确率
clf3 = MultinomialNB()
clf3.fit(x_resampled, y_resampled)
y_pred3 = clf3.predict_proba(tv.transform(x_test))[:,1]
roc_auc_score(y_test,y_pred3)


# In[25]:


#查看混淆矩阵
y_predict3 = clf3.predict(tv.transform(x_test))
cm = confusion_matrix(y_test, y_predict3)
cm


# In[31]:


test3 = '真的不喜欢，不好看，没感觉，有点好奇这部片为何被过度赞誉到这个地步'
ceshi(clf3,test3)


# 
# ### 模型评估测试
# 

# In[32]:


#词向量训练
tv2 = TfidfVectorizer(stop_words=stopwords, max_features=3000, ngram_range=(1,2))
tv2.fit(data_model['cus_comment'])

#SMOTE插值
X_tmp = tv2.transform(data_model['cus_comment'])
y_tmp = data_model['target']
sm = SMOTE(random_state=0)
X,y = sm.fit_sample(X_tmp, y_tmp)

clf = MultinomialNB()
clf.fit(X, y)

def fenxi(strings):
    strings_fenci = fenci(pd.Series([strings]))
    return float(clf.predict_proba(tv2.transform(strings_fenci))[:,1])


# In[34]:


#随机测试好评
fenxi(
'年的奥斯卡颁奖礼上，被如日中天的《阿甘正传》掩盖了它的光彩，而随着时间的推移，这部电影在越来越多的人们心中的地位已超越了《阿甘》。每当现实令我疲惫得产生无力感，翻出这张碟，就重获力量。毫无疑问，本片位列男人必看的电影前三名！回顾那一段经典台词：“有的人的羽翼是如此光辉，即使世界上最黑暗的牢狱，也无法长久地将他围困！”')


# In[ ]:

if (fenxi>=50):
    print("好评")
else:
print("差评即包括中评，以及用户表达语义不明评语")


