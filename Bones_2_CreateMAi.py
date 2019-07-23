#!/usr/bin/env python
# coding: utf-8

# # Create MAs using individual ECs
# 
# This script is a modified version of Bones_1_CreateMA.ipynb in order to create MAs for each type of EC.

# In[ ]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')

# remove warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# In[ ]:


import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.externals import joblib


# ## Create MAs by each EC

# In[ ]:


# read datasets
df_source = pd.read_csv('./datasets/ds.bones_MA_details.csv')


# In[ ]:


df_source.head()


# In[ ]:


df_source.columns


# In[ ]:


exper_conds  = ['Animal', 'Treat', 'Period']
features     = ['PTH', 'BALP', 'BGLAP', 'INTP', 'TRAP', 'CTX-I']


# In[ ]:


print('-> Creating MAs for each EC ...')
df_raw = df_source.copy()

for experim in exper_conds:    # for each experimental condition colunm
    for descr in features:  # for each input feature of metabolism
        # calculate the Avg for a pair of experimental conditions and a descriptor
        avgs = df_raw.groupby(experim, as_index = False).agg({descr:"mean"})

        # rename the avg column name
        avgs = avgs.rename(columns={descr: 'avg-'+ descr + '-' + experim})

        # merge an Avg to dataset
        df_raw = pd.merge(df_raw, avgs, on=experim)

        # add MA to the dataset for pair Exp cond - descr
        df_raw['MA-'+descr+'-'+experim] = df_raw[descr] - df_raw['avg-'+descr+'-'+experim]
print("Done!")


# In[ ]:


df_raw.columns


# In[ ]:


print('--> Saving the dataset with all MA details ...')
df_raw.to_csv('./datasets/ds.bones_MA2_details.csv', index=False)
print('Done!')


# In[ ]:




