#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import seaborn as sns


# In[2]:


df=pd.read_excel('medical.xlsx')


# In[3]:


df.shape


# In[4]:


df.head


# In[ ]:





# In[ ]:





# In[ ]:





# In[5]:


df.drop('Patient Id',axis=1,inplace=True)


# In[6]:


df.head()


# In[18]:


df.plot()


# In[ ]:


sns.set_style('darkgrid')


# In[16]:


sns.pairplot(df);


# In[23]:


sns.countplot(df['Level'])


# In[9]:


Level_mapping={'HIGH':2,'MEDIUM':1,'LOW':0}


# In[10]:


def preprocess_inputs(df):
    df=df.copy()
    data['label']=data['Level'].replace(Level_mapping)
    
    y=data['Level'].copy()
    X=data.drop('Level',axis=1).copy()
    
    X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=123)
    
    return X_train,X_test,y_train,y_test


# In[11]:


X=df.drop('Level',axis=1)
y=df['Level']


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)


# In[13]:


svclassifier=SVC(kernel='linear')
svclassifier.fit(X_train,y_train)


# In[14]:


y_pred=svclassifier.predict(X_test)
print(y_pred)


# In[15]:


print(classification_report(y_test,y_pred))


# In[ ]:




