
# coding: utf-8

# In[1]:


from sklearn.linear_model import LinearRegression
import pandas as pd
import datetime
from datetime import date
import numpy as np
from sklearn.model_selection import cross_val_score as CV
import math
import numpy as np
import matplotlib.pyplot as plt
import random


# In[23]:


master_train = pd.read_csv("Auction_master_train.csv")
pd.set_option('display.max_columns', 500)
master_train.head()


# In[24]:


#데이터 전처리 시작!
master_train.info()
master_train.isnull().sum()


# In[30]:



master_td=master_train.drop(['addr_li','addr_san','addr_etc','addr_bunji1','addr_bunji2','Specific','road_bunji1','road_bunji2','Close_date','point.x','point.y'], axis=1)


# In[31]:


def change_datetime(x):
    today = datetime.date.today()
    someday = datetime.date(int(x[:4]), int(x[5:7]), int(x[8:10]))
    diff = today - someday
    return diff.days
master_tds=master_td[master_td.Preserve_regist_date != "1111-11-11 00:00:00"]
master_tds["Appraisal_date"] = master_tds["Appraisal_date"].apply(lambda x : x[:10])
master_tds["First_auction_date"] = master_tds["First_auction_date"].apply(lambda x : x[:10])
master_tds["Final_auction_date"] = master_tds["Final_auction_date"].apply(lambda x : x[:10])
master_tds["Preserve_regist_date"] = master_tds["Preserve_regist_date"].apply(lambda x : x[:10])
master_tds["Appraisal_date"] = master_tds["Appraisal_date"].apply(change_datetime)
master_tds["First_auction_date"] = master_tds["First_auction_date"].apply(change_datetime)
master_tds["Final_auction_date"] = master_tds["Final_auction_date"].apply(change_datetime)
master_tds["Preserve_regist_date"] = master_tds["Preserve_regist_date"].apply(change_datetime)


# In[32]:


#등기등록날짜부터 감정일까지의 노후정도를나타내는 변수 = age_day
master_tds["age_day"]=master_tds["Preserve_regist_date"]-master_tds["Appraisal_date"]
#총감정가의 몇 %를 최저매각가격으로 놓았는지 나타내는 변수 = minprice_per
master_tds["minprice_per"]=master_tds["Minimum_sales_price"]/master_tds["Total_appraisal_price"]
#건물별 맨위층 맨아래층 중간층은 가격차이가 있는점을 보고 층의 정도를 나타내는 변수 = floor_per, 아래층0 ~ 1위층
master_tds["floor_per"]=master_tds["Current_floor"]/master_tds["Total_floor"]
#경매 기간 days
master_tds["duration"]=master_tds["First_auction_date"]-master_tds["Final_auction_date"]
master_tds.info()


# In[33]:


#master_tds["Auction_class"].value_counts()
master_tds.Auction_class.replace({'임의':0, '강제':1},inplace=True)
master_tds.Apartment_usage.replace({'아파트' :0, '주상복합': 1}, inplace=True)
#auction_class_dum= pd.get_dummies(master_tds["Auction_class"])


# In[29]:


#print(master_tds.Apartment_usage.unique())
master_tds["Apartment_usage"].value_counts()


# In[67]:


master_tda=master_tds.drop(['Auction_key','addr_do','addr_si','addr_dong','Preserve_regist_date','Creditor','Appraisal_company','Appraisal_date','Minimum_sales_price','Total_appraisal_price','Current_floor','Total_floor','First_auction_date','road_name','Final_auction_date'],axis=1)
len(master_tda.columns.tolist())


# In[70]:


corr_matrix = master_tda.corr()
abs(corr_matrix["Hammer_price"]).sort_values(ascending=False)
low_index = cor_del.tail(5).index


# In[87]:



new_master= master_tda.drop(list(low_index),axis=1)
new_master_1=pd.get_dummies(new_master)
new_master_1.head()
model = LinearRegression().fit(pd.DataFrame(new_master_1.Hammer_price), new_master_1.drop('Hammer_price',axis=1))


#더미화 한 후 모델 만들고 crossvalidation으로 검증
x = master_tda.drop('Hammer_price',axis=1)
x = pd.get_dummies(x)
y = master_tds.Hammer_price
model = LinearRegression()
#cv는 10-fold라는 뜻, scoring은 어떤 지표를 쓸거냐
scores = CV(model,x, y, cv=10, scoring='neg_mean_squared_error')


# In[88]:


scores


# In[89]:


scores.mean()

