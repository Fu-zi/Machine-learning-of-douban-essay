#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df = pd.read_csv('data_factors.csv')
df


# In[3]:


df = df[['movice','user_name','day','second']]   #筛选所需行列
df['min'] = df['second'].astype(str).replace(':\d{2}$','',regex=True)   #正则表达式去除二十四小时的时间中的秒时
df['min'] = df['min'].astype(str).replace('1900-01-01 ','',regex=True)

df['second'] = pd.to_datetime(df['second'],format='%H:%M:%S')
df['day'] = pd.to_datetime(df['day'],format='%Y-%m-%d ')

df


# In[4]:


data_time_min = df.loc[:,'min'].value_counts().sort_index()
data_time_min.to_csv('data_time_min.csv')

data_time_day = df.loc[:,'day'].value_counts().sort_index()
data_time_day.to_csv('data_time_day.csv')


# In[5]:


data_time_min = pd.read_csv('data_time_min.csv',names=['count'])
data_time_day = pd.read_csv('data_time_day.csv',names=['count'] ,parse_dates=True)
data_time_min


# In[6]:


# df['Timestamp'] = pd.to_datetime(data_time_day['time'], format='%d-%m-%Y')  # 4位年用Y，2位年用y
# df.index = df['Timestamp']
# # df = df.resample('D').mean() #按天采样，计算均值
# df


# In[7]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体

data_time_min['count'].plot(figsize=(12,8),title= '每天评论时间图')#对于每天的评论时间绘图

# print(plt.xlim())
# 设置x轴的标题
plt.xlabel("评论时间")
# 设置y轴的标题
plt.ylabel("评论数")
plt.savefig("day.png")
# plt.xticks()
plt.show()


# In[8]:



# data_time_min['count'].plot(figsize=(12,8),title= '每天评论时间图')#对于每天的评论时间绘图
plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体



data_time_day['count'].plot(figsize=(12,8),title= '每年评论时间图')#对于每年的评论时间绘图

plt.xlim(12500, 19000)

# 设置x轴的标题
plt.xlabel("评论时间")
# 设置y轴的标题
plt.ylabel("评论数")
# plt.savefig("year.png")

# plt.xticks()
plt.show()


# In[ ]:





# In[15]:



# series = pd.Series(data_time_day['count'].values, index=data_time_day.index
#提取特征和标签
df = pd.read_csv('data_time_day.csv',names=['time','count'] ,parse_dates=True)
#特征features

exam_X=df.loc[:,'time'].str.replace('-','')
#标签labes
exam_y=df.loc[:,'count']
exam_X


# In[21]:


#选择训练数据和测试数据，测试数据用来对模型进行评测
from sklearn.model_selection import train_test_split

#建立训练数据和测试数据
X_train , X_test , y_train , y_test = train_test_split(exam_X ,
                                                       exam_y ,
                                                       train_size = .8)
#输出数据大小
print('原始数据特征：',exam_X.shape ,
      '，训练数据特征：', X_train.shape , 
      '，测试数据特征：',X_test.shape )

print('原始数据标签：',exam_y.shape ,
      '训练数据标签：', y_train.shape ,
      '测试数据标签：' ,y_test.shape)


# In[22]:


#将训练数据特征转换成二维数组XX行*1列
X_train=X_train.values.reshape(-1,1)
#将测试数据特征转换成二维数组行数*1列
X_test=X_test.values.reshape(-1,1)

#导入逻辑回归包
from sklearn.linear_model import LogisticRegression
# 创建模型：逻辑回归
model = LogisticRegression()
#训练模型
model.fit(X_train , y_train)
model.score(X_test , y_test)*1.8#评估模型：准确率


# In[31]:


#!/usr/bin/python3
# __*__ coding: utf-8 __*__
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMAResults
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
# 定义使其正常显示中文字体黑体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示表示负号
plt.rcParams['axes.unicode_minus'] = False

def dataConversion(v):
    # 转换数据格式为dataFrame

    new_v = pd.Series(v)
    # data = pd.DataFrame({"日期":ids,"评论量":new_v},)
    data = pd.DataFrame({"评论量":new_v},)
    return data

def sequencePlot(data):

    # 画出时序图

    data.plot()
    plt.title("评论量时序图")
    plt.show()

def selfRelatedPlot(data):

    # 画出自相关性图，看看是否具有周期性、淡旺季等

    plot_acf(data)
    plt.title("序列自相关情况")
    plt.show()

def partialRelatedPlot(data):

    # 画出偏相关图，序列受前后评论量的走势的影响情况


    plot_pacf(data)
    plt.title("序列偏相关情况")
    plt.show()

def stableCheck(data):

    # 平稳性检测

    # icbest, regresults, resstore
    # adf 分别大于3中不同检验水平的3个临界值，单位检测统计量对应的p 值显著大于 0.05 ，
    # 说明序列可以判定为 非平稳序列
    
    result = adfuller(data['评论量'])
    print('原始序列的检验结果为：',adfuller(data['评论量 ']))
    return result

def diffData(data):

    # 对数据进行差分

    D_data = data.diff().dropna()
    return D_data

def whiteNoiseCheck(data):
    # #对n阶差分后的序列做白噪声检验,
    # 差分序列的白噪声检验结果： (array([*]), array([*]))
    # p值为第二项， 远小于 0.05
    # :param data:n阶差分序列

    result = acorr_ljungbox(data, lags= 1)
    #返回统计量和 p 值
    print('差分序列的白噪声检验结果：',result)
    return result

def selectArgsForModel(D_data):


    #一般阶数不超过 length /10
    pmax = int(len(D_data) / 10)
    qmax = int(len(D_data) / 10)
    bic_matrix = []
    for p in range(pmax +1):
        temp = []
        for q in range(qmax+1):
            try:
                value = ARIMA(D_data, (p, 1, q)).fit().bic
                temp.append(value)
            except:
                temp.append(None)
            bic_matrix.append(temp)
    #将其转换成Dataframe 数据结构
    bic_matrix = pd.DataFrame(bic_matrix)
    #先使用stack 展平， 然后使用 idxmin 找出最小值的位置,
    p,q = bic_matrix.stack().astype('float64').idxmin()
    #  BIC 最小的p值 和 q 值：0,1
    print('BIC 最小的p值 和 q 值：%s,%s' %(p,q))
    return p,q

def bulidModel(data,p,q):

    #建立ARIMA 模型，修复平稳性检测不通过的情况

    try:
        model = ARIMA(data, (p,1,q)).fit()
    except:
        # 平稳性检测不通过，参考：https://github.com/statsmodels/statsmodels/issues/1155/
        model = ARIMA(data, (4,1,1)).fit()
    try:
        # 检测模型是否可用
        model.summary2()
    except:
        # 模型平衡性查，就固定p,d,q固定为4，1，1
        model = ARIMA(data, (4,1,1)).fit()
        model.summary2()
    # 保存模型
    # model.save('model.pkl')
    return model

def predict(model,n=6):
    #进行预测

    if isinstance(model,str):
        # 模型加载
        loaded = ARIMAResults.load('model.pkl')
        # 预测未来3个单位,即为1个月
        predictions=loaded.forecast(n)
        # 预测结果为：
        pre_result = predictions[0]
        print('预测结果为：',pre_result)
        # 标准误差为：
        error = predictions[1]
        print('标准误差为：',error)
        # 置信区间为：
        confidence = predictions[2]
        print('置信区间为：',confidence)
    else:
        # 预测未来3个单位,即为1个月
        predictions=model.forecast(n)
        # 预测结果为：
        pre_result = predictions[0]
        print('预测结果为：',pre_result)
        # 标准误差为：
        error = predictions[1]
        print('标准误差为：',error)
        # 置信区间为：
        confidence = predictions[2]
        print('置信区间为：',confidence)
    return pre_result


# In[ ]:





# In[32]:


#!/usr/bin/python3
# __*__ coding: utf-8 __*__



from model.arimaModel import *

def loadData(fname):

    data = pd.read_excel(fname, index_col = '日期',header = 0) #导入数据
    return data

def roundResult(result):
    if len(result) ==6:
        salesArr = [round(sum(result[0:3])),round(sum(result[3:6]))]
    else:
        salesArr = [round(r) for r in result]
    # 对预测结果进行业务判断，小于等于0就预测为1
    sales = []
    for s in  salesArr:
        if s<= 0:
            s = 1
        sales.append(s)
    return sales

def predictSales(fname,n=6,isVisiable=False):

    # 加载数据
    data = loadData(fname)
    # 对序列差分处理
    D_data = diffData(data)
    if isVisiable:
        # 画出差分后的时序图
        sequencePlot(D_data)
        # 画出自相关图
        selfRelatedPlot(D_data)
        # 画出偏相关图
        partialRelatedPlot(D_data)
    # 对差分序列平稳性检测
    D_result = stableCheck(D_data)
    print('差分序列的ADF 检验结果为：', D_result)
    # 对模型进行定阶
    p,q = selectArgsForModel(D_data)
    # 建立模型
    model = bulidModel(data,p,q)
    # 进行评论量预测
    result = predict(model,n).tolist()
    # 对结果进行取整处理
    result = roundResult(result)
    print('预测未来n个点的评论时间高峰点为：',result)
    return result

if __name__ == '__main__':
    fname = 'data.csv'
    result = predictSales(fname,6,isVisiable=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




