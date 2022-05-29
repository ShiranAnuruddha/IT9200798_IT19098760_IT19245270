#!/usr/bin/env python
# coding: utf-8

# In[38]:


import pandas as pd
import numpy as np


# In[39]:


cancer = pd.read_csv('cancer1.csv')


# In[40]:


cancer.isnull()


# In[41]:


cancer.isnull().sum()


# In[42]:


cancer['Level']


# In[43]:


cancer.head()


# In[99]:


cancer.dtypes


# In[44]:


cancer['Level'].replace('High','2',inplace=True)


# In[45]:


cancer.head()


# In[46]:


cancer['Level'].replace('Low','0',inplace=True)
cancer['Level'].replace('Medium','1',inplace=True)
cancer['Level'].replace('High','2',inplace=True)


# In[107]:


cancer.head(10)


# In[48]:


cancer.shape


# In[51]:


x = cancer.drop(['Level'],axis=1).values
y = cancer['Level'].values


# In[52]:


print(x)


# In[53]:


print(y)


# In[54]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[63]:


from sklearn.linear_model import LinearRegression
model_nl = LinearRegression()
model_nl.fit(x_train,y_train)


# In[56]:


y_pred = model_nl.predict(x_test)
print(y_pred)


# In[57]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[58]:


x_pred = model_nl.predict(x_train)
r2_score(y_train, x_pred)


# In[61]:


model_nl.predict[['33','1','2','4','5','4','3','2','2','4','3','2','2','4','3','4','2','2','3','1','2','3','4']]


# In[85]:


import matplotlib.pyplot as plt
plt.figure(figsize = (20,10))
plt.scatter(y_test,y_pred)
plt.xlabel('Actual Level')
plt.ylabel('Predicted Level')
plt.title('Actual Level vs Predicted Level')


# In[112]:


pred_y_cancer = pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred})
pred_y_cancer[:10]


# In[ ]:




