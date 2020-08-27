#!/usr/bin/env python
# coding: utf-8

# In[220]:


#Importing all the Packages and Libraries

import sklearn
from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn import metrics


# In[221]:


#Formula of Linear Regression and Formula of Conversion of Celsius to Farenheit

# y = mx + c (whatever you named the constant)

# F = 1.8C + 32


# In[222]:


#Making a list of Temperatures in Celsius

x = list(range(0, 120)) #Temp in Celsius

# y = [1.8*F + 32 for F in x] #This is called List Comprehension of For Loop

#Using List Comprehension of For Loop for Converting Celsius to Farenheit and using random Function to add some noise in the Data

y = [1.8*F + 32 + random.randint(-3,3) for F in x]


# In[223]:


#Printing x (Celsius) and y (Farenheit)

print(f'X: {x}')
print(f'Y: {y}')


# In[224]:


#Plotting the Graph of Celsius and Farenheit Data

plt.plot(x, y, '-*r')


# In[225]:


#Reshaping the Data for using through SkLearn

x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)


# In[226]:


#Splitting the Data into Training Data and Testing Data

xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(x, y, test_size=0.2)


# In[227]:


#Printing the Shape of xTrain

print(xTrain.shape)


# In[228]:


#Using Linear Regression Model

model = linear_model.LinearRegression()


# In[229]:


#Using the fit function to Train the Model

model.fit(xTrain, yTrain)


# In[230]:


#Printing the Coefficient/Slope of Data

print(f'Coefficient (m) : {model.coef_}') #m


# In[231]:


#Printing the Intercept of Data

print(f'Intercept (b) : {model.intercept_}') #b or c whatever you named it


# In[232]:


#Checking the Accuracy of Model

accuracy = model.score(xTest, yTest)


# In[233]:


print(f'Accuracy : {round(accuracy*100,  2)}')


# In[234]:


#Again reshaping the Data to Plot the Data and Information

#Plotting the Graph between original values and predicted values

x1 = x.reshape(1, -1)[0]
m = model.coef_[0][0]
c = model.intercept_[0]
y1 = [m*F + c for F in x]
plt.plot(x, y, '-*r')
plt.plot(x1, y1, '-+b')
plt.show()


# In[235]:


#Predicting the Data using the values of xTest

pred = model.predict(xTest)

#Printing the values of our predictions and the values of yTest

print(f'Predictions : {pred}')

print(f'yTest : {yTest}')


# In[236]:


#Plotting the graph between yTest values and predicted values

plt.scatter(yTest, pred)


# In[237]:


#Finding the Errors in Our Model

print('Mean Absolute Error , MAE:', metrics.mean_absolute_error(yTest, pred))
print('Mean Squared Error , MAE:', metrics.mean_squared_error(yTest, pred))
print('Root Mean Squared Error , RMSE:', np.sqrt(metrics.mean_squared_error(yTest, pred)))


# In[ ]:




