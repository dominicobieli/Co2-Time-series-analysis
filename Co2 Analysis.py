#!/usr/bin/env python
# coding: utf-8

# # Importing Libaries to help the exploration

# In[1]:


#Co2 TimeSeries Analysis in London 
import numpy as np
import pandas as pd
import seaborn as sns
import datetime

import matplotlib.pylab
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 16
get_ipython().run_line_magic('matplotlib', 'inline')
#ignore warning messages 
import warnings
import itertools
warnings.filterwarnings('ignore') 


# In[2]:


#Loading the Pre-Interview Dataset
data=pd.read_csv("Tdata.csv")
#number of rows and columns
data.shape


# In[3]:


# First 5 rows snapshot
data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# # Dataset interrogation and cleansing

# In[6]:


print(data.isnull().sum())


# In[7]:


data['N1Class'] = data['N1Class'].fillna('NotVan')
data


# In[8]:


data['Euro'].fillna(data['Euro'].mode()[0], inplace=True)
data['VehicleType'].fillna(data['VehicleType'].mode()[0], inplace=True)
data['FuelType'].fillna(data['FuelType'].mode()[0], inplace=True)


# In[9]:


print(data.isnull().sum())


# # Dataset has been cleared of Missing values now. time to start visualising and storytelling.

# In[10]:


plt.figure(figsize=(12,6))
sns.countplot(x='VehicleType',data=data)
plt.title('Count of Vehicle Type')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(12,6))
sns.countplot(x='FuelType',data=data)
plt.title('Count of Fuel Type')
plt.ylabel('Count')
plt.show()


plt.figure(figsize=(12,6))
sns.countplot(x='Euro',data=data)
plt.title('Count of Euro')
plt.ylabel('Count')
plt.show()


# In[11]:


import matplotlib.pyplot as plt
plt.figure(figsize=(8,6))
data['Euro'].hist()
plt.title('Euro standards')
plt.ylabel('Count')
plt.xlabel('Euro')
plt.show()


# In[12]:


plt.figure(figsize=(8,6))
g = sns.countplot(data["zone"])


# In[13]:


date = data.Year,data.Month


# In[14]:


date


# In[15]:


data.dtypes


# In[16]:


D = (data.loc[data["FuelType"]== 'DIESEL'])
P = (data.loc[data["FuelType"]== 'PETROL'])
Z = (data.loc[data["FuelType"]== 'ZEV'])
H = (data.loc[data["FuelType"]== 'HYBRID'])
U = (data.loc[data["FuelType"]== 'Unknown/Unread'])


# In[17]:


data['Day'] = 1


# In[18]:


data['date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])


# In[19]:


data.head()


# In[20]:


dfvh = data.groupby(['date','VehicleType'],as_index=False)['Distinct_DayAve'].sum()


# In[21]:


dfvh


# In[22]:


data[data['zone'] == 'All Cameras'].groupby(['date','VehicleType'],as_index=False)['Distinct_DayAve'].sum()


# In[23]:


dfvh_baseline= dfvh[dfvh['date'] == '2019-01-01']


# In[24]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x='date', 
           y='Distinct_DayAve',
          hue = 'VehicleType',
          data = dfvh[dfvh.VehicleType.isin(['PHV','L','M1','M2','M3'])])
plt.title('Vehicle Types in All Cameras')


# In[25]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x='date', 
           y='Distinct_DayAve',
          hue = 'VehicleType',
          data = dfvh[dfvh.VehicleType.isin(['N1','N2','N3','T','MHC','BUS'])])
plt.title('Vehicle Types in All Cameras')


# In[26]:


dfEU = data.groupby(['date','Euro',],as_index=False)['Distinct_DayAve'].sum()


# In[27]:


dfEU


# In[28]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x='date', 
           y='Distinct_DayAve',
          hue = 'Euro',
          data = dfEU)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Euro standards in All Cameras')


# In[29]:


data[data['zone'] == 'All Cameras'].groupby(['date','Euro'],as_index=False)['Distinct_DayAve'].sum()


# In[30]:


dfEU2 = data[data['zone'] == 'Central + Inner'].groupby(['date','Euro'],as_index=False)['Distinct_DayAve'].sum()


# In[31]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x='date', 
           y='Distinct_DayAve',
          hue = 'Euro',
          data = dfEU2)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Euro standards in Central+Inner')


# In[32]:


dfvh2= data[data['zone'] == 'Central + Inner'].groupby(['date','VehicleType'],as_index=False)['Distinct_DayAve'].sum()


# In[33]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x='date', 
           y='Distinct_DayAve',
          hue = 'VehicleType',
          data = dfvh2[dfvh2.VehicleType.isin(['PHV','L','M1','M2','M3'])])
plt.title('Vehicle Types in Central+Inner')


# In[34]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.lineplot(x='date', 
           y='Distinct_DayAve',
          hue = 'VehicleType',
          data = dfvh2[dfvh2.VehicleType.isin(['N1','N2','N3','T','MHC','BUS'])])
plt.title('Vehicle Types in Central+Inner')


# In[ ]:





# In[ ]:





# In[ ]:




