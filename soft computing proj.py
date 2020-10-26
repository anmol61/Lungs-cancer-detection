#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[14]:


df=pd.read_csv('lungs.csv')
df.drop(["Name","Surname"],inplace=True,axis=1)
df


# In[15]:


df.shape


# In[16]:


df.info()


# In[17]:


df.head()


# In[19]:


df['Result'].value_counts()


# In[20]:


X=df.iloc[:,:-1].values
X.shape


# In[21]:


Y=df['Result'].values
Y.shape


# In[22]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[23]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.20)


# In[24]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[25]:


from sklearn import svm
model = svm.SVC()


# In[26]:


model.fit(X_train, y_train)


# In[27]:


y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))


# In[ ]:




