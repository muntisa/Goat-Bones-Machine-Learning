#!/usr/bin/env python
# coding: utf-8

# # Pipelines for classifiers using LOOCV
# 
# For each dataset, classifier and folds:
# - Robust scaling
# - LOOCV
# - balanced accurary as score
# 
# We will use folders *datasets2* and *results2_LOOCV*.

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

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.metrics import confusion_matrix,accuracy_score, roc_auc_score,f1_score, recall_score, precision_score
from sklearn.utils import class_weight

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import RBF
from sklearn.svm import LinearSVC

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import RFECV, VarianceThreshold, SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel, SelectPercentile, f_classif
import os


# In[ ]:


get_ipython().system('ls ./datasets2/*')


# In[ ]:


get_ipython().system('ls ./results2_LOOCV/*')


# In[ ]:


# get list of files in datasets2 = all datasets
dsList = os.listdir('./datasets2')
print('--> Found', len(dsList), 'dataset files')


# In[ ]:


# create a list with all output variable names 
outVars = []
for eachdsFile in dsList:
    outVars.append( (eachdsFile[:-4])[3:] )


# ### Define script parameters

# In[ ]:


# define list of folds
foldTypes = [56] # LOOCV (56 = number of instances, it is not used in any calculation) 

# define a label for output files
targetName = '_Outer_LOOCV'

seed = 42


# ### Function definitions

# In[ ]:


def  set_weights(y_data, option='balanced'):
    """Estimate class weights for umbalanced dataset
       If ‘balanced’, class weights will be given by n_samples / (n_classes * np.bincount(y)). 
       If a dictionary is given, keys are classes and values are corresponding class weights. 
       If None is given, the class weights will be uniform """
    cw = class_weight.compute_class_weight(option, np.unique(y_data), y_data)
    w = {i:j for i,j in zip(np.unique(y_data), cw)}
    return w 


# In[ ]:


def getDataFromDataset(sFile, OutVar):
    # read details file
    print('\n-> Read dataset', sFile)
    df = pd.read_csv(sFile)
    #df = feather.read_dataframe(sFile)
    print('Shape', df.shape)
    # print(list(df.columns))
    
    # select X and Y
    ds_y = df[OutVar]
    ds_X = df.drop(OutVar,axis = 1)
    Xdata = ds_X.values # get values of features
    Ydata = ds_y.values # get output values

    print('Shape X data:', Xdata.shape)
    print('Shape Y data:',Ydata.shape)
    
    # return data for X and Y, feature names as list
    return (Xdata, Ydata, list(ds_X.columns))


# In[ ]:


# define a function to calculate ACC with LOOCV pipelines

def getACC(y, yhat):
    good = 0 # number of cases where y = yhat = good prediction
    
    for i in range(len(yhat)):
        if yhat[i] == y[i]: good +=1
            
    return float(good/len(yhat))


# In[ ]:


def Pipeline_OuterCV(Xdata, Ydata, label = 'my', class_weights = {0: 1, 1: 1}, folds = 3, seed = 42):
    # inputs:
    # data for X, Y; a label about data, number of folds, seeed
    # default: 3-fold CV, 1:1 class weights (ballanced dataset)
    
    # define classifiers
    names = ['KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB']
    classifiers = [KNeighborsClassifier(3),
                   SVC(kernel="linear",random_state=seed,gamma='scale'),
                   SVC(kernel = 'rbf', random_state=seed,gamma='auto'),
                   LogisticRegression(solver='lbfgs',random_state=seed),
                   DecisionTreeClassifier(random_state = seed),
                   RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=seed),
                   XGBClassifier(n_jobs=-1,seed=seed)
                  ]
    # results dataframe: each column for a classifier
    df_res = pd.DataFrame(columns=names)

    # build each classifier
    print('* Building scaling+feature selection+outer '+str(folds)+'-fold CV for '+str(len(names))+' classifiers:', str(names))
    total = time.time()
    
    # define a fold-CV for all the classifier
    outer_cv = LeaveOneOut()
    
    resML = [] # list with ML results
    
    # for each classifier
    for name, clf in zip(names, classifiers):
        start = time.time()
        
        # create pipeline: scaler + classifier
        estimators = []
        
        # SCALER
        estimators.append(('Scaler', RobustScaler() ))
        
        # add Classifier
        estimators.append(('Classifier', clf)) 
        
        # create pipeline
        model = Pipeline(estimators)
        
        # evaluate pipeline
        scores = cross_val_score(model, Xdata, Ydata, cv=outer_cv, scoring='balanced_accuracy', n_jobs=-1)
        ACC = getACC(list(Ydata), list(scores))
        resML.append(ACC)
        print('%s, ACC=%0.2f, Time:%0.1f mins' % (name, ACC, (time.time() - start)/60))
        
    print('Total time:', (time.time() - total)/60, ' mins')             
    
    # return a list of lists for each classifier
    return resML


# ### Calculations

# In[ ]:


# for each subset file
all_results =[ ] # all results 

for OutVar in outVars:
    sFile = './datasets2/ds.'+str(OutVar)+'.csv'

    # get data from file
    Xdata, Ydata, Features = getDataFromDataset(sFile,OutVar)

    # Calculate class weights
    class_weights = set_weights(Ydata)
    print("Class weights = ", class_weights)
        
    # try different folds for each subset -> box plots
    for folds in foldTypes:
        
        # calculate outer CV for different binary classifiers
        resList = Pipeline_OuterCV(Xdata, Ydata, label = OutVar, class_weights = class_weights, folds = folds, seed = seed)
        resList.append(OutVar)
        
    # add each result to a summary dataframe
    all_results.append(resList)


# In[ ]:


df_results = pd.DataFrame(all_results,columns=['KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB','Dataset'])
df_results['folds'] = 'LOOCV'
df_results


# In[ ]:


resFile = './results2_LOOCV/'+'ML_Outer-LOOCV.csv'
df_results.to_csv(resFile, index=False)


# ### Mean scores

# In[ ]:


df_means =df_results.groupby(['Dataset','folds'], as_index = False).mean()[['Dataset', 'folds','KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB']]


# In[ ]:


resFile_means = './results2_LOOCV/'+'ML_Outer-LOOCV_means.csv'
df_means.to_csv(resFile_means, index=False)


# ### Best ML results

# In[ ]:


# find the maximum value rows for all MLs
bestMLs = df_means[['KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB']].idxmax()
print(bestMLs)


# In[ ]:


# get the best score by ML method
for ML in ['KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB']:
    print(ML, '\t', list(df_means.iloc[df_means[ML].idxmax()][['Dataset', 'folds', ML]]))


# In[ ]:


# Add a new column with the original output name (get first 2 characters from Dataset column)
getOutOrig = []
for each in df_means['Dataset']:
    getOutOrig.append(each[:2])
df_means['Output'] = getOutOrig
df_means


# In[ ]:


resFile_means2 = './results2_LOOCV/'+'ML_Outer-LOOCV_means2.csv'
df_means.to_csv(resFile_means2, index=False)


# ### Get the best ML for each type of output

# In[ ]:


for outName in list(set(df_means['Output'])):
    print('*********************')
    print('OUTPUT =', outName)
    df_sel = df_means[df_means['Output'] == outName].copy()
    for ML in ['KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB']:
        print(ML, '\t', list(df_sel.loc[df_sel[ML].idxmax(),:][['Dataset', 'folds', ML]]))


# In[ ]:


df_sel.loc[df_sel[ML].idxmax(),:]


# ### Get the best ML for each type of output for LOOCV

# In[ ]:


df_LOOCV = df_means[df_means['folds']=='LOOCV'].copy()
df_LOOCV.head()


# In[ ]:


for outName in list(set(df_LOOCV['Output'])):
    print('*********************')
    print('OUTPUT =', outName)
    
    df_sel = df_LOOCV[df_LOOCV['Output'] == outName].copy()
    print('MAX =',df_sel[['KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB']].max().max())
    
    for ML in ['KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB']:
        print(ML, '\t', list(df_sel.loc[df_sel[ML].idxmax(),:][['Dataset', 'folds', ML]]))


# In[ ]:


print('* Summary LOOCV')
for outName in list(set(df_LOOCV['Output'])):
    df_sel = df_LOOCV[df_LOOCV['Output'] == outName].copy()
    print(outName,df_sel[['KNN', 'SVM linear', 'SVM', 'LR', 'DT', 'RF', 'XGB']].max().max())


# **Conclusions LOOCV**: using LOOCV, we are able to obtain classifiers with ACC > 0.70 for almost all outputs. In addition, the best ACC > 0.91 for Fw!

# In[ ]:




