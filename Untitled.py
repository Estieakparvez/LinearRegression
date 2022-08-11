#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


df=pd.read_csv('Height-Weight Data.csv')
df


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.Height


# In[6]:


df.Weight


# In[7]:


df.Height.mean()


# In[8]:


df.Weight.mean()


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[11]:


plt.figure(figsize=(12,9))
plt.scatter(df.Height,df.Weight)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Height-Weight Data',color='red')


# In[12]:


x=df[['Height']]
y=df[['Weight']]


# In[13]:


from sklearn.model_selection import train_test_split as tts


# In[14]:


xtrain,xtest,ytrain,ytest=tts(x,y,test_size=.30)


# In[15]:


from sklearn.linear_model import LinearRegression


# In[16]:


reg = LinearRegression()


# In[17]:


reg.fit(xtrain,ytrain)


# In[18]:


m = reg.coef_
m


# In[19]:


c = reg.intercept_
c


# In[20]:


pred = reg.predict(xtest)
pred


# In[21]:


df['predicted result']=reg.predict(x)


# In[22]:


df.head()


# In[24]:


x1=250
y=m*x1+c
y


# In[ ]:




