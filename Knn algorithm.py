#!/usr/bin/env python
# coding: utf-8

# # Knearest neighbour

# - Importing packages

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Loading the data


# In[3]:


data= pd.read_csv('healthcare-dataset-stroke-data.csv')


# In[4]:


data


# # Checking null values

# In[5]:


data.isnull().sum()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[ ]:





# In[8]:


data= data.drop(['id'],axis=1)


# In[9]:


data


# In[10]:


data.isnull().sum()


# In[11]:


mean= data['bmi'].mean()


# In[12]:


data['bmi']=data['bmi'].fillna(mean)


# In[13]:


data['bmi'].isnull().sum()


# In[14]:


data['stroke'].unique()


# In[15]:


data.info()


# # Converting categorical to Numerical

# In[16]:


data['gender'].unique()


# In[17]:


data['gender']=data['gender'].map({'Male':0,'Female':1,'Other':2}).astype(int)


# In[18]:


data['ever_married'].unique()


# In[19]:


data['ever_married']=data['ever_married'].map({'Yes':0,'No':1}).astype(int)


# In[20]:


data['work_type'].unique()


# In[21]:


data['work_type']=data['work_type'].map({'Private':0,'Self-employed':1,'Govt_job':2,'children':3,'Never_worked':4})


# In[22]:


data['Residence_type']=data['Residence_type'].map({'Urban':0,'Rural':1})


# In[23]:


data['smoking_status'].unique()


# In[24]:


data['smoking_status']=data['smoking_status'].map({'formerly smoked':0,'never smoked':1,'smokes':2,'Unknown':3}).astype(int)


# In[25]:


data.info()


# In[26]:


data['smoking_status']


# In[27]:


data['stroke'].value_counts()


# In[28]:


x=data.iloc[:,:-1]


# In[29]:


y=data.iloc[:,-1]


# In[30]:


# upsampling(balancing the data)


# In[31]:


from imblearn.over_sampling import SMOTE


# In[32]:


reg=SMOTE(random_state=42)


# In[33]:


x,y=reg.fit_resample(x,y)


# # Building the model

# In[34]:


from sklearn.model_selection import train_test_split as tts


# In[35]:


x_train, x_test, y_train, y_test = tts(x, y, test_size=0.33, random_state=42)


# In[36]:


from sklearn.neighbors import KNeighborsClassifier


# In[37]:


reg=KNeighborsClassifier()


# In[38]:


reg.fit(x_train,y_train)


# # Checking training accuracy

# In[39]:


y_train_pred=reg.predict(x_train)


# In[40]:


train_comparision=pd.DataFrame({'actual':y_train,'predicted':y_train_pred})


# In[41]:


train_comparision


# In[42]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[43]:


print(classification_report(y_train,y_train_pred))


# In[44]:


accuracy_score(y_train,y_train_pred)


# In[45]:


confusion_matrix(y_train,y_train_pred)


# # for test data

# In[46]:


y_test_pred=reg.predict(x_test)


# In[47]:


confusion_matrix(y_test,y_test_pred)


# In[48]:


accuracy_score(y_test,y_test_pred)


# In[49]:


print(classification_report(y_test,y_test_pred))

