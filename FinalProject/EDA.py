#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis on Adult Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('adult.data', header=None, sep=', ', na_values='?')


# In[3]:


# Total records of original samples
print('The number of original samples is ', len(data))  


# In[4]:


# Total records after removing NA
data_dropna = data.dropna(axis=0, how='any')
print('The number of samples after removing NA is ',len(data_dropna)) 


# In[5]:


# Drop final weights column
data_dropna = data_dropna.drop(columns=[2], axis=1)


# In[6]:


# Assign column names
data_dropna.columns = ['age', 'workclass', 'education', 'education-num',
                     'marital-status', 'occupation', 'relationship',
                     'race', 'sex', 'capital-gain', 'capital-loss',
                     'hours-per-week', 'native-country', '50K?']


# In[17]:


# Education Year and Education Level
df_edu_stats = pd.crosstab(data_dropna['education'], data_dropna['education-num'], normalize=True)
df_edu_stats


# In[13]:


fig, ax = plt.subplots(dpi=200)
for item in list(data_dropna['education'].unique()):
    ax.hist(data_dropna[data_dropna['education'] == item]['education-num'], color=tuple(np.random.rand(3)), label=item)
ax.set_xlabel('Education Years')
ax.set_title('Education Years and Education Level')
ax.legend(loc=[1, 0])
plt.show()


# In[38]:


fig, ax2 = plt.subplots(dpi=200)
sns.heatmap(df_edu_stats, cmap="YlGnBu", annot=True, annot_kws={"size": 8,"va": 'top'}, 
            linewidths=.5)
ax2.set_title('Correlation between Education Year and Education Level')
plt.show()


# In[7]:


# Marital status and Relationship
df_marital_stats = pd.crosstab(data_dropna['education'], data_dropna['relationship'], normalize=True)
df_marital_stats


# In[37]:


fig, ax1 = plt.subplots(dpi=200)
sns.heatmap(df_marital_stats, cmap="YlGnBu", annot=True, annot_kws={"size": 8,"va": 'top'}, 
            linewidths=.5)
ax1.set_title('Correlation between Marital Status and Relationship')
plt.show()


# In[41]:


# Country and Race
data_dropna['country_num'] = 0
data_dropna.loc[data_dropna['native-country'] == 'United-States', 'country_num'] = 1
df_county_stats = pd.crosstab(data_dropna['country_num'], data_dropna['race'], normalize=True, margins=True, margins_name='Total')
df_county_stats


# In[44]:


data_dropna = data_dropna.drop(['country_num'], axis=1)


# In[45]:


# capital gain and loss
cg_na = sum(data_dropna['capital-gain']==0)/len(data_dropna)
cl_na = sum(data_dropna['capital-loss']==0)/len(data_dropna)
print('The proportion of 0 in capital-gain is', cg_na, '\n')
print('The proportion of 0 in capital-loss is', cl_na, '\n')


# In[56]:


f, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=200)
sns.distplot(data_dropna['capital-gain'], ax=axes[0]).set_title('Distribution of Capital Gain')
sns.distplot(data_dropna['capital-loss'], ax=axes[1]).set_title('Distribution of Capital Loss')
plt.tight_layout()
plt.show()


# In[58]:


# Workclass and Salary
df_job_stats = pd.crosstab(data_dropna['workclass'], data_dropna['50K?'], normalize=True, margins=True, margins_name='Total')
df_job_stats


# In[69]:


fig, ax1 = plt.subplots(dpi=200)
sns.heatmap(df_job_stats, cmap="YlGnBu", annot=True, annot_kws={"size": 8,"va": 'top'}, 
            linewidths=.5)
ax1.set_title('Correlation between Workclass and Salary')
plt.show()


# In[ ]:




