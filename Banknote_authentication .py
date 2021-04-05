#!/usr/bin/env python
# coding: utf-8

# In[84]:


import pandas as pd #data processing
import numpy as np #linear algebra
import matplotlib.pyplot as plt # data visualization
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# In[3]:


df= pd.read_csv('D:\Datascience\Bank note authentication\BankNote_Authentication.csv') # reading the Data


# ###### Data Inspection

# In[22]:


df.head()


# In[21]:


df=df.rename(columns={'class': 'Auth_status'}) #change the target variable name for convience


# In[23]:


print(df.shape)


# In[44]:


df.isnull().any() # checking if the is any null values present in the dataset


# The above data we can see we have independent variable and one dependent variable lets, Go for for details information regarding this dataset 

# In[24]:


df.info()


# the above information implies that all variables are continious. But as per the business problem we understand that dependent variable should be categorical in nature lets do some more analysis

# In[26]:


plt.figure(figsize=(10,8))
sns.countplot(x='Auth_status', data=df)
plt.xlabel('Auth_status', fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.title('Graphically presenting the dependent variable', fontsize=15)
plt.xticks(fontsize=14)
plt.show()


# from the above description we can clearly see that the dependent variable is categorical in nature, because values of the "class" variable is either 1 or 0

# So now is very much evident that the dependent variable is a categorical variable with two levels 1 and 0 in it.
# there are around 

# In[27]:


df['Auth_status']=df['Auth_status'].astype('category') #changed the datatype of the class variable into categorical


# In[46]:


df.describe(include='all').T


# ##### Target Variable: inspection 

# In[28]:


df['Auth_status'].value_counts()


# In[30]:


size=[df.Auth_status[df['Auth_status']==1].count(), df.Auth_status[df['Auth_status']==0].count() ]


# In[32]:


label=['Authenticated', 'Not_authenticated']


# In[42]:


plt.figure(figsize=(10,10))
plt.pie(size, labels=label, explode=(0,0.05), shadow=True, autopct='%1.1f%%', startangle=45)
plt.show()


# Almost 44.5% of the data is authenticated

# #### Out-Lier analysis

# In[54]:


sns.set(font_scale=1.5)
plt.figure(figsize=(15,10))
n=1
for col in df.select_dtypes('float64'):
    plt.subplot(2,2,n)
    sns.boxplot(df[col])
    plt.tight_layout()
    n=n+1
        


# variace and skewness are not having any outliers. where as curtosis and entropy show some outliers. which are are not necessarily outliers. these are call as business outliers which means they add value to the data set

# #### Univariate Analysis

# In[58]:


n=1
plt.figure(figsize=(10,10))
for col in df.select_dtypes('float64'):
    plt.subplot(2,2,n)
    sns.distplot(df[col], kde=True, bins=20)
    plt.tight_layout()
    n=n+1
    


# Curtosis is positively skewed, 
# Entropy and skewnessis negatively skewed,

# ##### Bi-Variate Analysis

# In[59]:


plt.figure(figsize=(15,10))
n=1
for col in df.select_dtypes('float64'):
    plt.subplot(2,2,n)
    sns.boxplot(x=df['Auth_status'], y=df[col])
    plt.tight_layout()
    n=n+1


# using the above graph we can see the relationship between the target variable and independent variable.
# 
# Variace : if we look at the  behaviour of target variable with respect to variance the inter quartile range between 0 to -3 for authenticated bank note where as  the inter quartile range between 1 to 4 for non authenticated bank note. We ca say that the this variable will be significant variable for this business problem
# 
# skewness: similarly when we look at the this variable the IQR for non authenticated bank note lies between 1 to 9 where as the IQR for authenticated bank note lies between -7  to 4. we can say this variable could also be a significant variable

# In[83]:


plt.figure(figsize=(8,8))
corr=np.round(df.corr(),2)
ax=sns.heatmap(corr, annot=True)
bottom, top=ax.get_ylim()
ax.set_ylim(top-0.5, bottom+0.5)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()


# We see a lot a strong negative correlation between skewness and curtosis. earlier we also saw that skewess could be a significant variable for the target variable. This negative correlation of curtosis with skewness might be an important point to be considered

# ##### Data Preprocessing 

# In[ ]:





# In[ ]:





# In[ ]:




