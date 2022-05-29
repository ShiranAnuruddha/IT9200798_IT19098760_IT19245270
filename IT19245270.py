#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


dataset = pd.read_csv('cancer1.csv')
dataset


# In[ ]:





# In[3]:


dataset['Level'].replace('High','2',inplace=True)
dataset['Level'].replace('Medium','1',inplace=True)
dataset['Level'].replace('Low','0',inplace=True)


# In[4]:


dataset.head()


# In[5]:


x = dataset.drop(['Level'],axis=1).values
y = dataset['Level'].values


# In[6]:


print(x)


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 101, stratify=y, test_size = 0.25)


# In[8]:


from sklearn.tree import DecisionTreeClassifier


# In[9]:


clf = DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)


# In[10]:


clf.get_params()


# In[11]:


x_test


# In[12]:


predictions = clf.predict(x_test)
predictions


# In[13]:


clf.predict_proba(x_test)


# In[14]:


clf.score(x_train, y_train)


# In[15]:


clf.score(x_test, y_test)


# In[16]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[17]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)


# In[18]:


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, target_names=['Low','Medium','High']))


# In[19]:


train_accuracy = []
validation_accuracy = []
for depth in range(1,10):
    clf = DecisionTreeClassifier(max_depth=depth, random_state=10)
    clf.fit(x_train, y_train)
    train_accuracy.append(clf.score(x_train, y_train))
    validation_accuracy.append(clf.score(x_test, y_test))


# In[20]:


frame = pd.DataFrame({'max_depth':range(1,10), 'train_acc':train_accuracy, 'valid_acc':validation_accuracy})
frame.head()


# In[ ]:





# In[ ]:




