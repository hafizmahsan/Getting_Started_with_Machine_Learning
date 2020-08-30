#!/usr/bin/env python
# coding: utf-8

# In[108]:


#Importing all the Packages and Libraries

import sklearn
from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn import metrics
from sklearn import preprocessing


# In[109]:


#Reading the Data

df = pd.read_csv('houses_to_rent.csv')


# In[110]:


df.head()


# In[111]:


#Loading the Data

df = df[['city', 'area', 'rooms', 'bathroom', 'parking spaces', 'animal', 'furniture', 'rent amount', 'fire insurance']]


# In[112]:


df.head()


# In[113]:


#Processing of the Data

df['rent amount'] = df['rent amount'].map(lambda i: int(i[2:].replace(',', '')))
df['fire insurance'] = df['fire insurance'].map(lambda i: int(i[2:].replace(',', '')))


# In[114]:


df.head()


# In[115]:


pre = preprocessing.LabelEncoder()

df['furniture'] = pre.fit_transform((df['furniture']))
df['animal'] = pre.fit_transform((df['animal']))


# In[116]:


df.head()


# In[117]:


#Checking Null Data

df.isnull().sum()


# In[118]:


#Splitting the Data

x = np.array(df.drop(['rent amount'], 1))
y = np.array(df['rent amount'])


# In[119]:


#Splitting the Data into Training Data and Testing Data

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=10)


# In[120]:


print('shape of x train: ',  xTrain.shape)
print('shape of x test: ',  xTest.shape)


# In[121]:


print('shape of y train: ',  yTrain.shape)
print('shape of y test: ',  yTest.shape)


# In[122]:


#Training the Data

model = linear_model.LinearRegression()


# In[123]:


model.fit(xTrain, yTrain)


# In[124]:


#Printing Accuracy of Data

accuracy = model.score(xTest, yTest)

print('Accuracy : ', round(accuracy*100, 3), '%')


# In[125]:


#Printing Shape of Predicted Values of Data

testVals = model.predict(xTest)
print(testVals.shape)


# In[126]:


#Comparison between Actual Values and Predicted Values and Also Printing the Error between Actual Values and Predicted Values

error = []
for i, testVal in enumerate(testVals):
    error.append(yTest[i]-testVal)
    print(f'Actual Value : {yTest[i]} Prediction : {int(testVal)} Error : {int(error[i])}')


# In[127]:


#Plotting the Data

plt.plot(yTest, testVals)


# In[128]:


#Plotting the Data

plt.scatter(yTest, testVals)


# In[ ]:




