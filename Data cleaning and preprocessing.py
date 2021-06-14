#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

data = pd.read_excel('review.xlsx',encoding = 'utf-8')


# In[3]:


data.duplicated()  #以所有列标签，查看是否存在的重复值
data_all = data.drop_duplicates( keep='last') # 以所有列标签，仅保存最后一个


# In[4]:


data_all.isnull().sum()                      #统计缺失值数量


# In[5]:


#删除缺失值低于99.9%的数据，删除行
data_all = data_all.dropna(subset=['user_name', 'user_review', 'user_likes_number', 'user_movice_watching'])
data_all.isnull().any()  #查看是否存在缺失值


# In[ ]:





# In[6]:


from  sklearn import ensemble
from sklearn.preprocessing import LabelEncoder

def set_missing(df,data_list,data_col):
    col_list=data_list
    col_list.append(data_col)   
    process_df = df.loc[:,col_list]
    class_le= LabelEncoder()
    for i in col_list[:-1]:
        process_df.loc[:,i]=class_le.fit_transform(process_df.loc[:,i].values)
    # 分成已知该特征和未知该特征两部分
    known=process_df[process_df[data_col].notnull()].values
    known[:, -1]=class_le.fit_transform(known[:, -1])
    unknown = process_df[process_df[data_col].isnull()].values
    # X为特征属性值
    X = known[:, :-1]
    # y为结果标签值
    y = known[:, -1]
    # fit到RandomForestRegressor之中
    rfr = ensemble.RandomForestRegressor(random_state=1, n_estimators=200,max_depth=4,n_jobs=-1)
    rfr.fit(X,y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, :-1]).round(0).astype(int)
    predicted=class_le.inverse_transform(predicted)
#     print(predicted)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df[data_col].isnull()),data_col] = predicted
    return df

#通过随机森林对数据进行拟合填充
#处理的训练集数据集所在估计缺失值的字段列表
data_list = ['movice','user_review_time','user_movice_watching']  
data_col = 'user_star'          #缺失列字段名称   评分
set_missing(data_all,data_list,data_col)


# In[7]:


data_list = ['movice','user_review_time','user_movice_watching','user_star']  
data_col =   'user_place'     #缺失列字段名称 用户常居地
set_missing(data_all,data_list,data_col)

data_list = ['movice','user_review_time','user_movice_watching','user_star','user_place' ]  
data_col =   'user_review_number'    #缺失列字段名称   用户评论数量
set_missing(data_all,data_list,data_col)

data_list = ['movice','user_review_time','user_movice_watching','user_star','user_place','user_review_number' ]  
data_col =  'user_movice_want'     #缺失列字段名称   用户想看电影数量
set_missing(data_all,data_list,data_col)

data_list = ['movice','user_review_time','user_movice_watching','user_star','user_place','user_review_number','user_movice_want' ]  
data_col =  'user_movice_watch'     #缺失列字段名称   用户正在看电影数量
set_missing(data_all,data_list,data_col)


# In[8]:


#查看缺失情况,得到缺失值其所在百分比
data_all.apply(lambda col:sum(col.isnull())/col.size)     


# In[25]:


import numpy as np
##筛选各评价中每个等级相应的行数据
# five_star = data_all.query("user_star == '力荐'") 
# four_star = data_all.query("user_star == '推荐'")
# three_star = data_all.query("user_star == '还行'")
# two_star = data_all.query("user_star == '较差'")
# one_star = data_all.query("user_star == '很差'")
star = pd.DataFrame(np.random.randn(5,5), columns=['力荐', '推荐','还行','较差','很差'])
#筛选各评价中每个等级相应的行数量
star['five_star'] = data_all.groupby(['user_star']).size()['力荐']
star['four_star'] = data_all.groupby(['user_star']).size()['推荐']
star['three_star'] = data_all.groupby(['user_star']).size()['还行']
star['two_star'] = data_all.groupby(['user_star']).size()['较差']
star['one_star'] = data_all.groupby(['user_star']).size()['很差']


# In[26]:


import re
#正则表达式去除列表中的多余中文字符
data_all['user_review_number'] = data_all['user_review_number'].str.replace('[\u4e00-\u9fa5]','')
data_all['user_registration_time'] = data_all['user_registration_time'].str.replace('[\u4e00-\u9fa5]','')
data_all['user_movice_watching'] = data_all['user_movice_watching'].str.replace('[\u4e00-\u9fa5]','')
data_all['user_movice_want'] = data_all['user_movice_want'].str.replace('[\u4e00-\u9fa5]','')
data_all['user_movice_watch'] = data_all['user_movice_watch'].str.replace('[\u4e00-\u9fa5]','')


# In[27]:



# #拆分用户评论时间： 年月份 与 二十四小时时间 正则表达式进行拆分
# def re_split_time(data_all):
#     for time in data_all['user_review_time'].astype(str):
#         #时间具体至日期
#         day = re.sub('\s\d{2}:\d{2}:\d{2}', '', time)
#         #时间具体至秒
#         second = re.sub('\d{4}-\d{2}-\d{2}\s','', time)
#         return day, second

# data_all['day'], data_all['second'] = re_split_time(data_all)
data_all['day'] = data_all['user_review_time'].astype(str).replace('\s\d{2}:\d{2}:\d{2}','',regex=True)
data_all['second'] = data_all['user_review_time'].astype(str).replace('\d{4}-\d{2}-\d{2}\s','',regex=True)

data_all.head()


# In[12]:


import jieba
#使用结巴分词，# 精确分词,变为列表
data_all['words_jieba'] =data_all['user_review'].astype(str).map(lambda word: ' '.join(jieba.cut(word)))


# In[13]:


# 启用停用词
def stop_words(word):
    with open(r"stop_words.txt",'r',encoding='utf-8') as words:
        stop_word = words.read()
    stopword = ''
    for i in word.split(" "):
        if i not in stop_word:
            stopword += i + " "
    return stopword

#调用停用词函数,并创建去除停用词后的字段
data_all['review'] = data_all['words_jieba'].map(lambda word : stop_words(word))
data_handle = data_all   #梳理清洗后的数据
data_handle.to_csv('data_handle.csv',index=False)  #将数据保存至本地


# In[14]:


# 进行词频统计
all_words = data_handle['review'].str.cat()
from collections import Counter 
cipin = Counter(all_words.split(" "))
cipin = pd.DataFrame(cipin.items(),columns=['word','counts'])
# cnts

# 排序
cipin.sort_values(by='counts' ,ascending=False)[:20]
cipin.to_csv('词频.csv')


# In[15]:


plot = ['情节', '故事', '编剧', '节奏', '高潮', '开篇', '铺垫', '转折', '叙事'
        ,'桥段', '主线','主题','题材','结局','剧本'] # 情节
frame = ['画面', '场景', '场面', '道具', '布景', '摄影', '实景'
        ,'视觉','镜头', '特效', '视效','全景','剪接','美景','景致','蓝光','高清'
        ,'特技','服装','构图','亮点','拍摄','色彩','造型','照明','奇幻','特写','景物','剪辑'] # 画面
sound = ['配音', '声音', '音效','配乐', '美妙', '配音','声音','音效','配乐','音乐','听觉','消音','音节','音频','音响','音奏','音色','音量']
Filmmaker = ['表演', '演技', '演的', '表演', '演员','明星','影人','导演','阵容','出场','戏份','主演','出演','扮演','饰演','角色','扮演者','演戏'] # 影人
# 关键词，词频表
def keywordCount(keyword,cipin):
    l = []
    for i in keyword:
        for j in cipin["word"]:
            if i == j:
                index = cipin[cipin['word'].isin([j])].index.values[0]
                l.append(index)
#                 count = np.array(cnts.iloc[[index],[1]]).tolist()[0][0]  
#                 sum += count
    return cipin.iloc[l,:]
plot_df = keywordCount(plot,cipin) # 剧情
frame_df = keywordCount(frame,cipin) # 画面
sound_df = keywordCount(sound,cipin) # 音效
Filmmaker_df = keywordCount(Filmmaker,cipin) # 演员
#在评论中谈及到的四大类型数量
print("剧情: ", plot_df['counts'].sum())
print("画面: " ,frame_df['counts'].sum())
print("音效: " ,sound_df['counts'].sum())
print("演员: " , Filmmaker_df['counts'].sum())


# In[16]:


data_handle


# In[17]:


import numpy as np
#通过以进行分词、删除停用词后的评论数据，匹配各用户与关键词表达的类型，进行匹配度评价因素
factor = []#与内容所匹配因素列表
for i in data_handle['review']:
    word_list = i.split(" ") #将字符串拆分，以列表循环匹配
    num = []    #所匹配到的与关键词类型相同的元素个数
    for j in [plot,frame,sound,Filmmaker]:
        #若有评论数据 与 关键词 有交集  
        #即匹配，合并为同一集合，可通过集合内个数，查看元素匹配个数
        num.append(len(set(word_list) & set(j)))
#     print(num)
    #匹配完成后，通过匹配个数，得出匹配度因素1, 1, 0, 0]  3.
    if np.max(num)== 0: #若最大值为零,则未匹配到关键词
        index = -1    #类型为：-1，即未匹配
    else:
        index = np.argmax(num) # 通过某列值最高，则影评归纳为与某关键词匹配度高的
        # 0为剧情, 1为画面, 2为音效, 3为演员
    factor.append(index) 
    
data_handle['factors'] = factor
#通过布尔值索引逆函数，isin()接受一个列表，判断该列中元素是否等于-1，实现除去。
data_handle =  data_handle[~data_handle['factors'].isin([-1])]
data_handle
data_handle.to_csv('data_factors.csv')


# In[18]:


factors = {
    0: '剧情',
    1: '画面',
    2: '音效',
    3: '演员'
}
data_handle['factors'] = data_handle['factors'].map(factors)


# In[19]:


# 查看影评四个方面star分布情况
result_df = pd.DataFrame(data_handle['user_star'].groupby(data_handle['factors']).value_counts())
result_df = result_df.unstack()


# In[20]:


result_df.columns = ['力荐', '推荐', '很差', '较差', '还行']
result_df


# In[21]:


#保存包含电影评价因素数据
data_factors = data_handle 
data_factors.to_csv('data_factors_no_num.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


from wordcloud import WordCloud 

# # 去除停用词的词云
# all_words = data_all['review'].str.cat()

# wc = WordCloud(width=450,height=300,max_font_size=150,font_path='SimHei.ttf')
# wc.generate_from_text()
# plt.imshow(wc)

# plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




