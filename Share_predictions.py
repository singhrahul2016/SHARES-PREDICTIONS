#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[50]:


s_media=pd.read_csv("OnlineNewsPopularity.csv")


# In[51]:


s_media.head()


# In[52]:


s_media.info()


# In[53]:


s_media.drop('url',axis=1,inplace=True)


# In[54]:


s_media.info()


# In[55]:


corr=s_media.corr()


# In[56]:


corr


# In[57]:


sns.heatmap(corr)


# In[58]:


sns.heatmap(corr,cmap='coolwarm')


# In[59]:


sns.countplot(s_media['shares'])


# In[60]:


from sklearn import metrics
train,test=train_test_split(s_media,test_size=0.3)


# In[61]:


train.shape


# In[62]:


test.shape


# In[63]:


train.columns


# In[64]:


train_x=train.drop('shares',axis=1)


# In[65]:


train_x.columns


# In[66]:


train_y=train['shares']


# In[67]:


test_x=test.drop('shares',axis=1)


# In[73]:


test_y=test['shares']


# In[74]:


test_y.head()


# In[69]:


linear_reg=LinearRegression()


# In[70]:


linear_reg.fit(train_x,train_y)


# In[77]:


pred=linear_reg.predict(test_x)


# In[79]:


plt.scatter(test_y,pred)


# In[83]:


linear_reg.score(pred,test_y)


# In[87]:


import pickle
pickle.dump(linear_reg, open('model.pkl','wb'))


# In[89]:


model = pickle.load(open('model.pkl','rb'))


# In[92]:


list1=[731,12,219,0.663594467,0.999999992,0.815384609,4,2,1,0,4.680365297,5,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,496,496,496,1,0,0,0,0,0,0,0,0.500331204,0.37827893,0.040004675,0.041262648,0.040122544,0.521617145,0.092561983,0.0456621,0.01369863,0.769230769,0.230769231,0.378636364,0.1,0.7,-0.35,-0.6,-0.2,0.5,-0.1875,0,0.1875]
print(len(list1))


# In[93]:


print(model.predict([list1]))


# In[ ]:




