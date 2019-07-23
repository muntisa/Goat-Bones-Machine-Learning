#!/usr/bin/env python
# coding: utf-8

# # Create Datasets
# 
# We will use previous files with MA (all EC) and MA2 (individual EC), and class transformed outputs. After mixing all data we have, we will create different datasets for each output using different combinations of input features. The datasets with MAs will be saved in *datasets2* and the datasets with classical one-hot encoding of ECs (no MAs!) will be saved in *datasets3*.

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


# In[ ]:


# read 3 previous generated datasets
df_oneHot = pd.read_csv('./datasets/ds.bones_raw_onehot.csv')  # outputs as numbers + one-hot encoding
df_MAmix  = pd.read_csv('./datasets/ds.bones_MA_details.csv')  # outputs as classes + MAmix (MAs for a set of ECs)
df_MAi    = pd.read_csv('./datasets/ds.bones_MA2_details.csv') # outputs as classes + MAi   (MAs for individual EC)


# In[ ]:


print(list(df_oneHot.columns))


# In[ ]:


print(list(df_MAmix.columns))


# In[ ]:


print(list(df_MAi.columns))


# Concatenate datasets into one single file with all data. Use of SerNum ID to mix data.

# In[ ]:


# define sets of column names for future merging

cols_EC = ['Animal', 'Treat', 'Period']
cols_EC_onehot = ['Animal_Fetus', 'Animal_Mon', 'Treat_Con', 'Treat_Res', 'Period_Late', 'Period_Mid']

cols_outputNum = ['Fw', 'Fl', 'Fd', 'Hw', 'Hl', 'Hd']
cols_outputClass = ['Fw_Class', 'Fl_Class', 'Fd_Class', 'Hw_Class', 'Hl_Class', 'Hd_Class']

cols_metabolism = ['PTH', 'BALP', 'BGLAP', 'INTP', 'TRAP', 'CTX-I']
cols_MAmix = ['MA-PTH-experim' , 'MA-BALP-experim', 'MA-BGLAP-experim',
              'MA-INTP-experim', 'MA-TRAP-experim', 'MA-CTX-I-experim']
cols_MAi= ['MA-PTH-Animal', 'MA-BALP-Animal', 'MA-BGLAP-Animal', 'MA-INTP-Animal',
           'MA-TRAP-Animal','MA-CTX-I-Animal','MA-PTH-Treat', 'MA-BALP-Treat', 
           'MA-BGLAP-Treat', 'MA-INTP-Treat', 'MA-TRAP-Treat', 'MA-CTX-I-Treat', 
           'MA-PTH-Period', 'MA-BALP-Period', 'MA-BGLAP-Period', 'MA-INTP-Period', 
           'MA-TRAP-Period', 'MA-CTX-I-Period']

cols_aux = ['no1', 'no2']
cols_key = ['SerNum']

cols_pi =   ['Prob_Animal','Prob_Treat','Prob_Period']
cols_pmix = ['Prob_Mix']


# In[ ]:


df = df_oneHot.copy() # starting with oneHot dataset


# In[ ]:


df_aux = df_MAmix[cols_key + cols_outputNum] # get the outputs as classes
df_aux.columns = cols_key + cols_outputClass # rename the names including Class
df_aux.columns


# In[ ]:


df = pd.merge(df, df_aux, on=['SerNum']) # add classes
df.columns


# In[ ]:


df_aux = df_MAmix[cols_key + cols_MAmix] # get the MA mix
df_aux.columns


# In[ ]:


df = pd.merge(df, df_aux, on=['SerNum']) # add MA mix
df.columns


# In[ ]:


df_aux = df_MAi[cols_key + cols_MAi] # get the MA mix
df_aux.columns


# In[ ]:


df = pd.merge(df, df_aux, on=['SerNum']) # add MAi
df.columns


# In[ ]:


# save dataset
print('-> Saving new dataset ...')
df.to_csv('./datasets/ds.bones_complete.csv', index=False)
print('Done!')


# ## Generation of datasets
# 
# Combine list of columns to generate different dataset:
# - cols_EC
# - cols_outputNum
# - cols_outputClass
# - cols_metabolism
# - cols_MAmix
# - cols_MAi
# - cols_aux
# - cols_key
# - cols_pi
# - cols_pmix

# In[ ]:


# Generate ds for classification: MAmix (6)
suf = 'MAmix'
for col in cols_outputClass:
    cols = [col] + cols_MAmix
    newdf = df[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAmix
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAi (18)
suf = 'MAi'
for col in cols_outputClass:
    cols = [col] + cols_MAi
    newdf = df[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAi
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAi + MAmix (24)
suf = 'MAi_MAmix'
for col in cols_outputClass:
    cols = [col] + cols_MAi + cols_MAmix
    newdf = df[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAi + cols_MAmix
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAi + MAmix + Metab (30!)
suf = 'MAi_MAmix_Metab'
for col in cols_outputClass:
    cols = [col] + cols_MAi + cols_MAmix + cols_metabolism
    newdf = df[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAi + cols_MAmix + cols_metabolism
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAi + Metab (24)
suf = 'MAi_Metab'
for col in cols_outputClass:
    cols = [col] + cols_MAi + cols_metabolism
    newdf = df[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAi + cols_metabolism
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAmix + Metab (12)
suf = 'MAmix_Metab'
for col in cols_outputClass:
    cols = [col] + cols_MAmix + cols_metabolism
    newdf = df[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAmix + cols_metabolism
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# ## Add probabilities for ECs

# In[ ]:


# Add both individual probability of each EC in dataset and mixed probability of a set of EC (product of individual probability)

dff = df.copy()
dff['Prob_Mix'] = 1 # create mixed probability = product of EC probabilities

for experim in cols_EC:
    dfx = df[cols_EC+cols_key].copy()

    # counts of ECs
    countEC = dfx.groupby(experim, as_index = False).agg({cols_key[0]:"count"})

    # calculate probabilities
    countEC[cols_key[0]] = countEC[cols_key[0]]/countEC[cols_key[0]].sum()
    
    # rename column
    countEC = countEC.rename(columns={cols_key[0]: 'Prob_'+ experim})
    
    dff = pd.merge(dff, countEC, on=experim)
    
    dff['Prob_Mix'] = dff['Prob_Mix'] * dff['Prob_'+ experim]


# In[ ]:


# save the new dataset
dff.to_csv('./datasets/ds.bones_complete.csv', index=False)


# In[ ]:


# Generate ds for classification: MAmix + pmix (7)
suf = 'MAmix_Pmix'
for col in cols_outputClass:
    cols = [col] + cols_MAmix + cols_pmix
    newdf = dff[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAmix + cols_pmix
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAi + pi (21)
suf = 'MAi_Pi'
for col in cols_outputClass:
    cols = [col] + cols_MAi + cols_pi
    newdf = dff[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAi + cols_pi
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAi + MAmix + Pi + Pmix (28)
suf = 'MAi_MAmix_Pi_Pmix'
for col in cols_outputClass:
    cols = [col] + cols_MAi + cols_MAmix + cols_pi + cols_pmix
    newdf = dff[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAi + cols_MAmix + cols_pi + cols_pmix
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAi + MAmix + Metab + Pi + Pmix (34!)
suf = 'MAi_MAmix_Metab_Pi_Pmix'
for col in cols_outputClass:
    cols = [col] + cols_MAi + cols_MAmix + cols_metabolism + cols_pi + cols_pmix
    newdf = dff[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAi + cols_MAmix + cols_metabolism + cols_pi + cols_pmix
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAi + Metab - Pi (27)
suf = 'MAi_Metab_Pi'
for col in cols_outputClass:
    cols = [col] + cols_MAi + cols_metabolism + cols_pi
    newdf = dff[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAi + cols_metabolism + cols_pi
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:


# Generate ds for classification: MAmix + Metab + Pmix (13)
suf = 'MAmix_Metab_Pmix'
for col in cols_outputClass:
    cols = [col] + cols_MAmix + cols_metabolism + cols_pmix
    newdf = dff[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_MAmix + cols_metabolism + cols_pmix
    newFile = './datasets2/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# ## Datasets without MAs
# 
# Let's try some datasets without MAs but with one hot representation of the ECs. This is the classical ML alternative to the MA method.

# In[ ]:


# Generate ds for classification: MAmix (6)
suf = 'OneHot'
for col in cols_outputClass:
    cols = [col] + cols_EC_onehot + cols_metabolism
    newdf = df[cols].copy()
    newdf.columns = [col+'_'+suf] + cols_EC_onehot + cols_metabolism
    newFile = './datasets3/ds.'+str(col)+'_'+suf+'.csv'
    print('-> Saving', newFile, '...')
    newdf.to_csv(newFile, index=False)
print('Done!')


# In[ ]:




