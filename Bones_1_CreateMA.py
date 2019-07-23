#!/usr/bin/env python
# coding: utf-8

# # Create MAs for a set of experimental conditions - Bones project
# 
# The model will try to classify bone parameters 'Fw', 'Fl', 'Fd', 'Hw', 'Hl', 'Hd' using 6 metabolism results 'PTH', 'BALP', 'BGLAP', 'INTP', 'TRAP', 'CTX-I' for 3 types of experimental conditions 'Animal', 'Treat', 'Period':
# - PTH	Parathyroid hormone	ng/ml
# - BALP	Bone Alkaline Phosphatase	mU/ml
# - BGLAP	Bone Gla protein	ng/ml
# - INTP	Cross-linked N-terminal Telopeptides of type I collagen	ng/ml
# - TRAP	Tartrate Resistant Acid Phosphatase	U/L
# - CTX-I	Cross-linked C-terminal Telopeptides of type I collagen	ng/ml
# - Fw	Femur_weight	g
# - Fl	Femur_length	mm
# - Fd	Femur_diamater	mm
# - Hw	humerus_weight	g
# - Hl	humerus_length	mm
# - Hd	humerus_diamater	mm
# 
# Gestation periods:
# - middle	middle-gestation
# - late	late-gestation
# 
# Animal:
# - Fetus	Fetus
# - Maternal	Mon
# 
# Experimental Conditions (EC):
# - two animal resources (fetus, maternal) for femur and humerus
# - two different treatments (control group and nutrition restriction group)
# - two different gestation periods (middle- and late- gestations)
# 
# **The script will do:**
# - Data integration - Integrate metabolism and bone data using 'SerNum','Animal','Treat','Period'; this way we are sure that both data are from the same experiment;
# - Output variables transformation - Z-score transform of the output variables depending on Animal + class transform using a cutoff (the bone parameters depends on the animal type)
# - MAmix and MAi - feature engineering for inputs using experimental conditions and metabolism; MAmix = MA for a set of EC (EC1,EC2,EC3); MAi = MA using each type of EC.
# - all these transformation will be saved as intermediate datasets in *datasets*.
# 
# **Note** - We will try two methodologies:
# 1. Transformed MAs for the metabolism data using ECs (differences of original features with the average value in specific ECs); we will use MA for mixed ECs and MA for each type of EC. In addition, we add probabilities of mixed or individual EC in the dataset. In *dataset2*, we will combine all these input features to create **72 datasets** for each output variables.
# 2. Classical ML with one hot representation of the ECs and original metabolism data (no MAs!); this method uses EC information directly as input binary features for each values of an EC (1/0): 'Animal_Fetus', 'Animal_Mon', 'Treat_Con', 'Treat_Res', 'Period_Late', 'Period_Mid'. We will create **6 datasets** in *datasets3*.

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


# ## Dataset integration = bones + metabolism

# In[ ]:


# read bone and metabolism datasets as dataframes
df_params = pd.read_csv('./datasets/bone_params.csv')
df_metabolites = pd.read_csv('./datasets/bone_metabolites.csv')


# In[ ]:


# check bones data
df_params.head()


# In[ ]:


# check metabolism data
df_metabolites.head()


# Check *null* values in both datasets:

# In[ ]:


print('No of Null values in df_params = ', df_params.isnull().values.sum())
print('No of Null values in df_metabolites = ', df_metabolites.isnull().values.sum())


# **Concatenate datasets** with bone dimensions and metabolism concentrations using serum Id and experimental conditions to be sure that the SerNum corresponds to the same experiment.

# In[ ]:


df = pd.merge(df_params, df_metabolites, on=['SerNum','Animal','Treat','Period'])
df


# In[ ]:


# save integrated dataset
print('-> Saving new dataset ...')
df.to_csv('./datasets/ds.bones_raw.csv', index=False)
print('Done!')


# In[ ]:


# new dataset dimension
print(df.shape)


# In[ ]:


# new datasset columns
print(df.columns)


# ## Process categorical nominal experimental conditions
# 
# This part is dedicated to the classical ML methodolody where the ECs are directly included as one-hot representations in the input features.

# In[ ]:


# make a copy of categorical experimental conditions
df_cat = df[['Animal', 'Treat', 'Period']].copy()
df_cat.head()


# In[ ]:


# check the number of values for each categorical feature
df_cat.describe()


# In[ ]:


# count the number of each EC
print(df_cat['Animal'].value_counts())


# In[ ]:


# count the number of each EC
print(df_cat['Treat'].value_counts())


# In[ ]:


# count the number of each EC
print(df_cat['Period'].value_counts())


# In[ ]:


# create dataframe for one-hot transform
df_cat_onehot = df_cat.copy()
df_cat_onehot.head()


# In[ ]:


# transform original ECs in one-hot features
df_cat_onehot = pd.get_dummies(df_cat_onehot, 
                               columns=['Animal', 'Treat', 'Period'], 
                               prefix = ['Animal', 'Treat', 'Period'])
df_cat_onehot.head()


# In[ ]:


# add these transformed features to the initial datasframe
df = pd.concat([df, df_cat_onehot], axis=1)
df.head()


# In[ ]:


# check all the column names
df.columns


# In[ ]:


# save new dataset with one-hot ECs as inputs
print('-> Saving new dataset ...')
df.to_csv('./datasets/ds.bones_raw_onehot.csv', index=False)
print('Done!')


# In[ ]:


df.columns


# ## Scaling of output variable
# 
# http://benalexkeen.com/feature-scaling-with-scikit-learn/
# 
# Because we shall search for classification models, we need to transform the 6 output variables into classes (Fw, Fl, Fd, Hw, Hl, Hd). First, let's check for outliers in order to decide what type of nornalization to use:

# In[ ]:


df_outs = df[['SerNum','Animal','Fw', 'Fl', 'Fd', 'Hw','Hl', 'Hd']].copy()
df_outs.head()


# Using direct observation on output variables and metabolism data, we observe that there is a clear separation between the ranges for Fetus and Mon. This means that if we choose to transform in classes using the average values of the outputs, the resulted classes will be 1 for Mom and 0 for Fetus because always the smaller values are from Fetus and the largest values are from Mon. We will see the differences in the following boxplots. Therefore, this transformation has no sense because we need to classify if an output has low or high value for both types of animals. In conclusion, we need to transform in classes using separated groups of animals.
# 
# Separate data by Aminal type:

# In[ ]:


df_outs_Fetus = df_outs[df_outs['Animal']=='Fetus'].copy()
df_outs_Fetus.shape


# In[ ]:


df_outs_Fetus.columns


# In[ ]:


# plot output variables for Fetus to see the range value and the possible outliers
df_outs_Fetus.boxplot()


# Let's do the same for Mon:

# In[ ]:


df_outs_Mon = df_outs[df_outs['Animal']=='Mon'].copy()
df_outs_Mon.shape


# In[ ]:


df_outs_Mon.columns


# In[ ]:


# plot output variables for Mon to see the range value and the possible outliers
df_outs_Mon.boxplot()


# **Conclusion about outliers**: we observ outliers in all the output variables for both type of animals. Thus, the safest method is to use **Robust scaller**.

# In[ ]:


# trasnform the output variables for Fetus

scaler = RobustScaler()
robust_scaled_df = scaler.fit_transform(df_outs_Fetus[['Fw', 'Fl', 'Fd', 'Hw','Hl', 'Hd']].copy())

# save the scaler 
joblib.dump(scaler, "scaler_Fetus_outs.save")

# create a dataframe with the scaled data
robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['Fw', 'Fl', 'Fd', 'Hw','Hl', 'Hd'])
robust_scaled_df


# In[ ]:


# concatenate scaled data to df
df_outs_Fetus_scaled = robust_scaled_df.copy()
df_outs_Fetus_scaled['SerNum']= list(df_outs_Fetus['SerNum'])
df_outs_Fetus_scaled['Animal']= list(df_outs_Fetus['Animal'])
df_outs_Fetus_scaled


# In[ ]:


# check data
df_outs_Fetus_scaled.describe()


# In[ ]:


# trasnform the output variables for Mon

scaler2 = RobustScaler()
robust_scaled_df2 = scaler2.fit_transform(df_outs_Mon[['Fw', 'Fl', 'Fd', 'Hw','Hl', 'Hd']].copy())

# save the scaler
joblib.dump(scaler2, "scaler_Mon_outs.save")

# create a dataframe with the scaled data
robust_scaled_df2 = pd.DataFrame(robust_scaled_df2, columns=['Fw', 'Fl', 'Fd', 'Hw','Hl', 'Hd'])
robust_scaled_df2


# In[ ]:


# concatenate scaled data to df
df_outs_Mon_scaled = robust_scaled_df2.copy()
df_outs_Mon_scaled['SerNum']= list(df_outs_Mon['SerNum'])
df_outs_Mon_scaled['Animal']= list(df_outs_Mon['Animal'])
df_outs_Mon_scaled


# In[ ]:


df_outs_Mon_scaled.describe()


# ## Scaled to Class
# 
# Because we used Robust scaler, the cutoff will be 0. Thus, the class transformed outputs will have value of 1 if the output is greater than 0.0 (median of output values) and 0 if the output value will be less than 0.0 (median of output values).

# In[ ]:


cutoff = 0.0 # due to Robust scaler


# In[ ]:


# transform of output scaled values to classes for Fetus

df_outs_Fetus_class = df_outs_Fetus_scaled.copy()
for output in ['Fw', 'Fl', 'Fd', 'Hw','Hl', 'Hd']:
    df_outs_Fetus_class[output].values[df_outs_Fetus_class[output] >= 0] = 1
    df_outs_Fetus_class[output].values[df_outs_Fetus_class[output] <  0] = 0
df_outs_Fetus_class


# In[ ]:


# transform of output scaled values to classes for Mon

df_outs_Mon_class = df_outs_Mon_scaled.copy()
for output in ['Fw', 'Fl', 'Fd', 'Hw','Hl', 'Hd']:
    df_outs_Mon_class[output].values[df_outs_Mon_class[output] >= 0] = 1
    df_outs_Mon_class[output].values[df_outs_Mon_class[output] <  0] = 0
df_outs_Mon_class


# In[ ]:


# append both animal type output variables transformed in classes
df_outs_class = df_outs_Fetus_class.append(df_outs_Mon_class)
print(df_outs_class.shape)
print(df_outs_class.columns)


# In[ ]:


# merge df with df_outs_class

print(df.columns)
print(df_outs_class.columns)


# In[ ]:


df_class = pd.merge(df.drop(['Fw', 'Fl', 'Fd', 'Hw', 'Hl', 'Hd'],axis = 1), 
                    df_outs_class, on=['SerNum','Animal'])
df_class.columns


# In[ ]:


# save new dataset
print('-> Saving new dataset ...')
df_class.to_csv('./datasets/ds.bones_class_allOuts.csv', index=False)
print('Done!')


# ## Experiment-centered scaling (Moving Averages) 
# 
# On other methodology is to calculate the MAs of the original metabolism input features using ECs. This way, the input features will be differences with respect with the average values in specific ECs. Thus, we will not use one-hot encoding of ECs but we will encode the EC information into the generated MAs.

# In[ ]:


# define sets of columns
experim  = ['Animal', 'Treat', 'Period']
features = ['PTH', 'BALP', 'BGLAP', 'INTP', 'TRAP', 'CTX-I']


# In[ ]:


# Transform original features into MAs using ECs

print('-> Creating MAs ...')
df_raw = df_class.copy()
for descr in features:  # for each input feature of metabolism
    # calculate the Avg for a pair of experimental conditions and a descriptor
    avgs = df_class.groupby(experim, as_index = False).agg({descr:"mean"})
    
    # rename the avg column name
    avgs = avgs.rename(columns={descr: 'avg-'+ descr + '-' + 'experim'})
    
    # merge an Avg to dataset
    df_raw = pd.merge(df_raw, avgs, on=experim)
    
    # add MA to the dataset for pair Exp cond - descr
    df_raw['MA-'+descr+'-'+'experim'] = df_raw[descr] - df_raw['avg-'+descr+'-'+'experim']
print("Done!")


# In[ ]:


# check new columns
df_raw.columns


# In[ ]:


print('--> Saving the dataset with all MA details ...')
df_raw.to_csv('./datasets/ds.bones_MA_details.csv', index=False)
print('Done!')


# In[ ]:




