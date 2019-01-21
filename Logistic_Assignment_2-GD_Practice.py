
# coding: utf-8

# # Dataload and import library

# In[7]:


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


# In[8]:


## 본인의 디렉토리에 맞게 설정하세요 
data = pd.read_csv('C:/Users/j3eun/OneDrive/Desktop/tobigs/assignment_2.csv')
data.head()


# In[9]:


y = data.Label
#data['salary']=data['salary'].apply(lambda x : x/10000)


# In[10]:


y


# # Logistic regression 해야하는 data 의 scatter plot

# In[11]:


# filter out the applicants that got admitted
MALE = data[data['Label']==1]
FEMALE = data[data['Label'] == 0]


# In[12]:


# plots
fig = plt.figure()
plt.scatter(MALE.iloc[:, 2], MALE.iloc[:, 3], s=10, label='Male')
plt.scatter(FEMALE.iloc[:, 2], FEMALE.iloc[:, 3], s=10, label='Female')
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Experience')
ax.set_ylabel('Salary')
plt.legend()
plt.show()


# In[13]:


## 독립변수들
X = data.iloc[:, 1:]

# rescaled_X 
normalized_X = (X[['experience','salary']]-X[['experience','salary']].mean())/X[['experience','salary']].std()
normalized_X['bias'] = X['bias']
cols = normalized_X.columns.tolist()
cols = cols[-1:] + cols[:-1]
normalized_X = normalized_X[cols]
normalized_X = normalized_X.values.tolist()
X= np.array(normalized_X)


#Basic_X
# X = X.values.tolist()
# X = np.array(X)
#X

# 종속변수 Target = T 
T = data['Label']
T = T.values.tolist()
T = np.array(T)
#T

# 회귀계수
beta = np.random.randn(3)
#beta= np.array([1,1,1])
#beta
print(X.shape)


# ## 회귀식
# ## $$ a_n=B^TX = \hat{B}_0 +\hat{B}_1x_1 +\hat{B}_2x_2 ,    \quad\mbox{n=데이터 수}$$
# 
# 
# 

# In[14]:


# 내적 -> 회귀계수와 X의 선형결합 
a = X.dot(beta)
print(a.shape)


# ## $$ \mbox 로그오즈 = 회귀식을\  p에\  대해서\  정리 $$
# ## $$ log(Odds) = \hat{B}_0 +\hat{B}_1x_1 + \hat{B}_2x_2 $$
# ## $$ \mbox P는 \ 로지스틱 \ 함수 $$
# ## $$ P = \frac{\mathrm{1} }{\mathrm{1} + e^{-a_n}} = \sigma(x,B) $$ 

# In[15]:





# In[16]:


'''
함수를 구현하세요 
INPUT: 회귀식(a)
Output: P
'''
def log_odds(a):
        return 1 / (1+ np.exp(a))
p=log_odds(a)
print(p)


# ## $$ \mbox 각\  데이터가\  따르는\  확률분포를\   label\  y와\  엮어서\  표현  $$
# 
# ## $$ P(x_i,y_i | B) = \begin{cases} \sigma(x,B)^{y}, & \mbox{if }y\mbox{ = 1} \\ (1-\sigma(x,B))^{1-y}, & \mbox{if }y\mbox{ = 0} \end{cases} $$

# ## $$ \mbox 위의 \ 확률분포를 \ 따르는 \   데이터에서 \ 얻은 \ Likelihood  $$
# 
# ## $$ L = \sigma(x,B)^{y}(1-\sigma(x,B))^{1-y} $$ 

# ## $$ \mbox log 변환  $$
# 
# ## $$ L^*=log(L) = y\sigma(x,B) + (1-y)(1-\sigma(x,B)) $$ 

# ## $$ \mbox Convex\  function으로\  만들기\  위한 \ (-) 곱   $$
# 
# ## $$ \mbox J는\ Loss\ Function  $$
# 
# ## $$ J= - L^* $$
# 
# ## $$ \mbox Loss\ function \ = \ Negative\ log \ likelihood\  $$

# ## $$ \mbox N개의 \ 데이터에\ 대한\ Negative\ Likelihood  $$
# ## $$ J(x_i,y_i|B) = -\sum_{i=1}^N y_i\sigma(x_i,B)- \sum_{i=1}^N(1-y_i)(1-\sigma(x_i,B)) $$ 
# 

# In[18]:


'''
Negative_Likelihood를 구현하세요
INPUT: P, Y(LABEL)
OUTPUT : 각 데이터들의 Negative_Likelihood 값의 합

'''
def negative_likelihood(p,y):
    sum1=np.sum(np.log(p)*y)
    sum2=np.sum((1-y)*np.log(1-p))
    sumt= -sum1-sum2
    return sumt
negative_likelihood= negative_likelihood(p,y)
print(negative_likelihood)
#def negative_likelihood1(p,y):
#    return np.sum(np.log(p)*y)
#sum1=negative_likelihood1(p,y)
#def negative_likelihood2(p,y):
#    return np.sum((1-y)*np.log(1-p))
#sum2=negative_likelihood2(p,y)
#negative_likelihood = -(sum1+sum2)
#negative_likelihood


# ## Loss function 에 대한 Gradient를 구하기 위한 작업 

# ## $$ \mbox J를 \ B에 \ 대해서\ 편미분  $$
# 
# ## $$ \mbox J는 \ P에 \ 대한\ 함수이고, \ P는 \ a에 \ 대한 \ 함수이고,\ a는\ B에 \ 대한\ 함수  $$
# 
# ## $${\partial J\over\partial B} =  -\sum_{i=1}^N {\partial J\over\partial P_i} {\partial P_i\over\partial a_i} {\partial a_i\over\partial B} $$ 

# ## $${\partial J\over\partial P_i} = \frac{y_i}{P_i}- \frac{1-y_i}{1-P_i} ,\quad {\partial P_i\over\partial a_i} = P_i(1-P_i), \quad {\partial a_i\over\partial B}=X^T $$ 

# ## $$ \mbox 또한 \ B는 \ B_0,\ B_1,\ B_2이\  있어서\ 각각\ 편미분    $$
# ## $$ \mbox 간단하게 \  표현하면   $$

# ## $${\partial J\over\partial B} =  -\sum_{i=1}^N {\partial J\over\partial P_i} {\partial P_i\over\partial a_i} {\partial a_i\over\partial B} = - X^T(y-P) =  X^T(P-y)$$ 

# In[17]:





# In[28]:


'''
위에서 구한 Negative Likelihood 함수를 각각 베타에 편미분하여 Gradient를 계산하고
Learning_rate와 곱하여 회귀계수를 업데이트 해주세요.
회귀계수가 업데이트 될 때 마다, P도 다시 업데이트 되야 합니다. 
'''
learning_rate=0.01
iteration = 50
def gradient(X,p,y, learning_rate, iteration):
    for i in range(iteration):
        jb=np.dot(np.transpose(X),np.array(p-y))
        beta_learn=beta-learning_rate*jb
        a=X.dot(beta_learn)
        p=log_odds(a)
    return beta_learn
beta_learn=gradient(X,p,y,learning_rate, iteration)
#print(beta_learn)
a = X.dot(np.array(beta_learn))
#print(a)
p=log_odds(a)
print(p)
#nl= negative_likelihood(p,y)
#nl


# In[51]:


X.T.shape


# In[52]:





# In[54]:


'''추정된 회귀계수가 beta=[B0,B2,B3]이라면, 다음 코드를 통해 시각화가 가능합니다'''


fig = plt.figure()
plt.scatter(MALE.iloc[:, 2], MALE.iloc[:, 3], s=10, label='Male',alpha=0.5)
plt.scatter(FEMALE.iloc[:, 2], FEMALE.iloc[:, 3], s=10, label='Female',alpha=0.5)
#plt.scatter(X[:,1], X[:,2],c=T , s=10, alpha=0.8)
x_axis = np.linspace(0, 10,50)
y_axis = -(beta[0] + x_axis*beta[1]) / beta[2]

ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Experience')
ax.set_ylabel('Salary')
plt.plot(x_axis, y_axis)
plt.legend()
plt.show()




# ### 직접 구현을 한 rough한 GD이기 때문에 성능이 안좋을 수 있습니다
# ### 초기값에 매우 민감하며, 기타 하이퍼 파라미터에 따라 성능이 달라집니다
