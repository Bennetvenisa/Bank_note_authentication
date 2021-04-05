#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd #data processing
import numpy as np #linear algebra
import matplotlib.pyplot as plt # data visualization
import seaborn as sns


# In[26]:


df= pd.read_csv('D:\Datascience\Bank note authentication\BankNote_Authentication.csv') # reading the Data


# ###### Data Inspection

# In[27]:


df.head()


# The above data we can see we have independent variable and one dependent variable lets, Go for for details information regarding this dataset 

# In[28]:


print(df.shape)


# In[31]:


df.info()


# the above information implies that all variables are continious. But as per the business problem we understand that dependent variable should be categorical in nature lets do some more analysis

# In[32]:


df.describe()


# from the above description we can clearly see that the dependent variable is categorical in nature, because values of the "class" variable is either 1 or 0

# In[33]:


plt.figure(figsize=(10,8))
sns.countplot(x='class', data=df)
plt.xlabel('Class', fontsize=15)
plt.ylabel('Counts', fontsize=15)
plt.title('Graphically presenting the dependent variable', fontsize=15)
plt.xticks(fontsize=14)
plt.show()


# So now is very much evident that the dependent variable is a categorical variable with two levels 1 and 0 in it.
# there are around 

# In[30]:


df['class']=df['class'].astype('category') #changed the datatype of the class variable into categorical


# In[ ]:





# In[ ]:




