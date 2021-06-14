#!/usr/bin/env python
# coding: utf-8

# In[1]:


#导入模块和数据
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('data_factors.csv')
df.drop('Unnamed: 0',axis=1, inplace=True) #删除多余值
df


# In[2]:


df['user_likes_number'].astype("int")
#查看 变量异常值的核心指标
df.describe()


# In[3]:


#相关分析，
#查看各数据间的相关关系
# 计算相关性矩阵
data = df[['user_likes_number', 'user_review_number', 'user_movice_watching', 'user_movice_want', 'user_movice_watch', 'factors']]
data.corr()


# In[4]:


sns.heatmap(data.corr(), cmap = 'GnBu')
#用户的被点赞量与评价因素、用户在看电影数量的变量成正相关关系，

#用户的被点赞量与用户想看电影数量的变量成负相关关系，


# In[5]:


#模型建立 
#通过RFM模型， R用户流失，F用户评论频率，M用户被点赞数量
data_6 = df[[ 'user_movice_watch','user_review_number','user_likes_number']]
data_6.head() 


# In[6]:


#构建特征
features = pd.concat([data_6],axis=1)
features.columns=['R','F','M']
print('RFM属性行为：\n',features.head())


# In[7]:


fig=plt.figure(figsize=(15,5))
ax1=fig.add_subplot(1,3,1)
sns.distplot(features['R'])
ax2=fig.add_subplot(1,3,2)
sns.distplot(features['F'])
ax3=fig.add_subplot(1,3,3)
sns.distplot(features['M'])


# In[8]:


from sklearn.preprocessing import StandardScaler
import numpy as np
#标准化
data=StandardScaler().fit_transform(features)
np.savez('./scale.npz',data)
print('标准化后RFM三个属性为：\n',data[:5,:])
fig=plt.figure()
ax=plt.axes(projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2])
ax.set_xlabel('R')
ax.set_ylabel('F')
ax.set_zlabel('M')


# In[ ]:


from sklearn.preprocessing import StandardScaler
# 正规化数据集 X
def normalize(X, axis=-1, p=2):
    lp_norm = np.atleast_1d(np.linalg.norm(X, p, axis))
    lp_norm[lp_norm == 0] = 1
    return X / np.expand_dims(lp_norm, axis)

# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances


class KMedoids():
    """
    k-medoids聚类算法.


    Parameters:
    -----------
    k: int
        聚类簇的数目.
    max_iterations: int
        最大迭代次数. 
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon, 
        则说明算法已经收敛
    """
    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 随机初始化k个聚类中心
    def init_random_medoids(self, X):
        n_samples, n_features = np.shape(X)
        medoids = np.zeros((self.k, n_features))
        for i in range(self.k):
            medoid = X[np.random.choice(range(n_samples))]
            medoids[i] = medoid
        return medoids

    # 返回离该样本最近的中心的索引
    def closest_medoid(self, sample, medoids):
        distances = euclidean_distance(sample, medoids)
        closest_i = np.argmin(distances)
        return closest_i
    
    # 将每一个样本分配到与其最近的一个中心
    def create_clusters(self, X, medoids):
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            medoid_i = self.closest_medoid(sample, medoids)
            clusters[medoid_i].append(sample_i)
        return clusters

    # 计算cost (所有样本到其相应中心的距离之和)
    def calculate_cost(self, X, clusters, medoids):
        cost = 0
        # For each cluster
        for i, cluster in enumerate(clusters):
            medoid = medoids[i]
            cost += euclidean_distance(medoid, X[cluster]).sum()
        return cost

    # Returns a list of all samples that are not currently medoids
    def get_X_no_medoids(self, X, medoids):
        no_medoids = []
        for sample in X:
            if not sample in medoids:
                no_medoids.append(sample)
        return no_medoids

    # 获取每个样本的label, 方法是将每个簇的索引号记做该簇中样本的label
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for i, cluster in enumerate(clusters):
            y_pred[cluster] = i
        return y_pred
   
    
    # Do Partitioning Around Medoids and return the cluster labels
    def predict(self, X):
        # 随机初始化self.k个中心
        medoids = self.init_random_medoids(X)
        # 进行cluster，将整个数据集中样本分配到与其最近的中心
        clusters = self.create_clusters(X, medoids)

        # 计算初始损失 (所有样本到其相应中心的距离之和)
        cost = self.calculate_cost(X, clusters, medoids)

        # 迭代, 直到 cost 不再下降
        for i in range(self.max_iterations):
            best_medoids = medoids
            lowest_cost = cost
            # 遍历所有中心(或者簇(clusters))
            for medoid in medoids:
                # 获取所有非中心的样本
                X_no_medoids = self.get_X_no_medoids(X, medoids)
                # 遍历所有非中心的样本
                for sample in X_no_medoids:
                    # Swap sample with the medoid
                    new_medoids = medoids.copy()
                    new_medoids[medoids == medoid] = sample
                    # 按照新的中心划分簇(clusters)
                    new_clusters = self.create_clusters(X, new_medoids)
                    # 计算中心更新之后的 cost
                    new_cost = self.calculate_cost(X, new_clusters, new_medoids)
                    # 如果中心更新之后的cost < 更新之前的cost, 则将中心, cost进行更新
                    if new_cost < lowest_cost:
                        lowest_cost = new_cost
                        best_medoids = new_medoids
            # If there was a swap that resultet in a lower cost we save the
            # resulting medoids from the best swap and the new cost             
            if lowest_cost < cost:
                cost = lowest_cost
                medoids = best_medoids
            # Else finished
            else:
                break
                
        # 按照最终(最优)的中心再划分簇(clusters)
        final_clusters = self.create_clusters(X, medoids)
        final=final_clusters
        # 按照最终(最优)的簇(clusters)获取所有样本的label
        return self.get_cluster_labels(final_clusters, X),medoids

def main():

    # 用Kmeans算法进行聚类
    X=data
    clf = KMedoids(k=3)
    y_pred,features= clf.predict(X)
    global cluster_center
    cluster_center=pd.DataFrame(features,columns=['R','F','M'])
    print(cluster_center)
    print('各样本类别标签为\n',y_pred)
    r1 = pd.DataFrame(y_pred,columns=['总数'])['总数'].value_counts()  # 统计不同类别样本的数目
    result=pd.concat([cluster_center,r1],axis=1).sort_index()
    print('最终每个类别的数目为：\n',result)
    # 可视化聚类效果
    print('聚类之后的3d图\n')
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.scatter(X[:,0],X[:,1],X[:,2],c=y_pred)
    ax.set_xlabel('R')
    ax.set_ylabel('F')
    ax.set_zlabel('M')
    
if __name__ == "__main__":
    main()
    


# In[ ]:



labels=cluster_center.columns.values
kinds=list(cluster_center.index)
result=pd.concat([cluster_center,cluster_center[[labels[0]]]],axis=1)

centers=np.array(result.iloc[:,:])

n=len(labels)
angle=np.linspace(0,2*np.pi,n,endpoint=False)
angle=np.concatenate((angle,[angle[0]]))

fig=plt.figure()
ax=fig.add_subplot(111,polar=True)

for i in range(len(kinds)):
    ax.plot(angle, centers[i], linewidth=1, label=kinds[i])

ax.set_thetagrids(angle * 180/np.pi,labels)
plt.title('用户价值雷达图')
plt.legend(loc='lower right')
plt.show()


# In[ ]:




