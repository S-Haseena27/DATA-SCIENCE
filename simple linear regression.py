#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv("Salary_Data.csv")


# In[3]:


data.info()


# In[4]:


data.describe()


# In[5]:


data.head()


# In[6]:


data.isnull().sum()


# # data visualization

# In[7]:


plt.figure(figsize=(4,5))


plt.title("simple linear regression")

plt.xlabel("Years of experience")

plt.ylabel("salary")


plt.scatter(x=data['YearsExperience'],y=data['Salary'],color='g')


plt.show()


# # splitting the data

# In[8]:


X=data['YearsExperience']      ### independent 


# In[9]:


y=data['Salary']               ### dependent


# In[ ]:





# In[10]:


X=X.values.reshape(-1,1)


# In[ ]:





# In[11]:


from sklearn.model_selection import train_test_split


# In[ ]:





# In[12]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[ ]:





# # plotting training data

# In[13]:


plt.figure(figsize=(4,5))

plt.title("simple linear regression")

plt.xlabel("X_train")

plt.ylabel("y_train")


plt.scatter(X_train,y_train)

plt.show()


# # building the model

# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


reg=LinearRegression()


# In[16]:


X_train.shape


# In[ ]:





# In[17]:


reg.fit(X_train,y_train)


# In[18]:


reg.coef_    # m(slope value)


# In[20]:


reg.intercept_   # c(intercept or constant value)


# In[ ]:





#  - our model is y = mx + c
#             - that is y = 9426.03876907 * x + 25324.33537924433 

# In[22]:


y_train_pred = reg.predict(X_train)


# In[23]:


y_train_pred


# In[24]:


X_train_1 = X_train.flatten()


# In[27]:


data_comparison = pd.DataFrame({'x_train':X_train_1,'y_train_actual':y_train,'y_train_pred':y_train_pred})


# In[28]:


data_comparison 


# - visualizing difference between actual points and prediction points

# In[30]:


plt.figure(figsize=(4,5))

plt.title("simple linear regression")

plt.xlabel("X_train")

plt.ylabel("y_train")


plt.scatter(X_train,y_train,color='r')
plt.plot(X_train,y_train_pred,color='g')


plt.show()


#  # finding the accuracy

# In[31]:


from sklearn.metrics import r2_score


# In[32]:


r2_score(y_train,y_train_pred)


# In[ ]:





# # for test data

# In[34]:


X_test


# In[35]:


y_test


# In[37]:


y_test_pred=reg.predict(X_test)


# In[41]:


X_test=X_test.flatten()


# In[ ]:





# In[42]:


test_data_compar=pd.DataFrame({'X_test': X_test,'y_test_actual': y_test,'y_test_pred': y_test_pred})


# In[43]:


test_data_compar


# # checking accuracy for test data

# In[45]:


r2_score(y_test,y_test_pred)


# - checking with real data

# In[50]:


reg.predict([[2.1]])


#          - our model is performing well because accuracy of test data is 95%

# In[ ]:




