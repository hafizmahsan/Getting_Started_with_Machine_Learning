#!/usr/bin/env python
# coding: utf-8

# In[67]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[68]:


df = pd.read_csv('USA_Housing_Data.csv')


# In[69]:


df.head()


# In[70]:


df.info()


# In[71]:


df.describe()


# In[72]:


df.columns


# In[73]:


sns.pairplot(df)


# In[74]:


sns.heatmap(df.corr(), annot=True)


# In[75]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]

y = df[['Price']]


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 20)


# In[77]:


model = LinearRegression()


# In[78]:


model.fit(X_train, y_train)


# In[79]:


model.coef_


# In[80]:


model.intercept_


# In[81]:


predictions = model.predict(X_test)

predictions


# In[82]:


plt.scatter(y_test, predictions)


# In[83]:


sns.distplot((y_test-predictions), bins=50)


# In[89]:


#Printing Accuracy of Data

accuracy = model.score(X_test, y_test)

print('Accuracy : ', round(accuracy*100, 3), '%')


# In[ ]:




