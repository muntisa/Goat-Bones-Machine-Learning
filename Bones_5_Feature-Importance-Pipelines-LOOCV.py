#!/usr/bin/env python
# coding: utf-8

# # Feature selection for Goat Bones project
# 
# We used pipelines for classifiers with LOOCV.
# 
# For each dataset, classifier and folds:
# - Robust scaling
# - LOOCV
# - balanced accurary as score
# 
# We will use folders *datasets2* for the best classifiers:
# 
# | Output | Classifier | Features          | Best AUC |
# |--------|------------|-------------------|----------|
# | Fw     | SVM        | MAi, MAmix, Metab | 0.911    |
# | Fl     | SVM linear | MAmix             | 0.875    |
# | Fd     | SVM        | MAmix, Metab      | 0.893    |
# | Hw     | SVM linear | MAmix             | 0.750    |
# | Hl     | SVM        | MAmix             | 0.696    |
# | Hd     | SVM        | MAmix, Metab      | 0.857    |

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


# ### Define script parameters

# In[ ]:


# define a label for output files
targetName = '_FeatImp_'
foldTypes = [56]
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
    names = ['SVM']
    classifiers = [SVC(kernel = 'rbf', random_state=seed,gamma='auto')]
    
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


# ## Feature importance by removoal for SVM RBF

# In[ ]:


# for each subset file
all_results =[ ] # all results 
outVars = ['Fw_Class_MAi_MAmix_Metab',
           'Fd_Class_MAmix_Metab',
           'Hl_Class_MAmix',
           'Hd_Class_MAmix_Metab']

OrigAUC = [0.911, 0.893, 0.696, 0.857]
k=0
for OutVar in outVars:
    sFile = './datasets2/ds.'+str(OutVar)+'.csv'

    # get data from file
    Xdata, Ydata, Features = getDataFromDataset(sFile,OutVar)
    
    # Feature importance by elimination (eliminate each feature and see score diff)
    for i in range(len(Features)):
        Xdata_new = np.delete(Xdata, i,1)

        # Calculate class weights
        class_weights = set_weights(Ydata)
        print("Class weights = ", class_weights)

        # try different folds for each subset -> box plots
        for folds in foldTypes:

            # calculate outer CV for different binary classifiers
            resList=[]
            myACC = Pipeline_OuterCV(Xdata_new, Ydata, label = OutVar, class_weights = class_weights, folds = folds, seed = seed)
            print(myACC)
            resList.append(float(myACC[0]))
            resList.append(float(myACC[0])-OrigAUC[k]) # add feature name to ACC
            resList.append(Features[i]) # add feature name to ACC
            resList.append(OutVar) # add file tag
        
        
        # add each result to a summary dataframe
        all_results.append(resList)
    k=k+1


# Save the results ordered by Dataset and Difference between the new AUC without a feature and the ACC for all features:

# In[ ]:


df_results = pd.DataFrame(all_results,columns=['New ACC', 'Diff with Pool ACC', 'Removed Feature', 'Dataset'])
df_results.sort_values(by=['Dataset','Diff with Pool ACC'], inplace=True)
df_results


# In[ ]:


resFile = './results2_LOOCV/'+'FeatImpbyRemoval-LOOCV_SVM_ACC.csv'
df_results.to_csv(resFile, index=False)


# ## Feature importance by removal for SVM linear

# In[ ]:


def Pipeline_OuterCV2(Xdata, Ydata, label = 'my', class_weights = {0: 1, 1: 1}, folds = 3, seed = 42):
    # inputs:
    # data for X, Y; a label about data, number of folds, seeed
    # default: 3-fold CV, 1:1 class weights (ballanced dataset)
    
    # define classifiers
    names = ['SVM linear']
    classifiers = [SVC(kernel="linear",random_state=seed,gamma='scale')]
    
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


# In[ ]:


# for each subset file
all_results =[ ] # all results 
outVars = ['Fl_Class_MAmix','Hw_Class_MAmix']
OrigAUC = [0.875, 0.750]
k=0

for OutVar in outVars:
    sFile = './datasets2/ds.'+str(OutVar)+'.csv'

    # get data from file
    Xdata, Ydata, Features = getDataFromDataset(sFile,OutVar)
    
    # Feature importance by elimination (eliminate each feature and see score diff)
    for i in range(len(Features)):
        Xdata_new = np.delete(Xdata, i,1)

        # Calculate class weights
        class_weights = set_weights(Ydata)
        print("Class weights = ", class_weights)

        # try different folds for each subset -> box plots
        for folds in foldTypes:

            # calculate outer CV for different binary classifiers
            resList=[]
            myACC = Pipeline_OuterCV2(Xdata_new, Ydata, label = OutVar, class_weights = class_weights, folds = folds, seed = seed)
            print(myACC)
            resList.append(float(myACC[0]))
            resList.append(float(myACC[0])-OrigAUC[k]) # add feature name to ACC
            resList.append(Features[i]) # add feature name to ACC
            resList.append(OutVar) # add file tag
        
        
        # add each result to a summary dataframe
        all_results.append(resList)
    k=k+1


# In[ ]:


df_results = pd.DataFrame(all_results,columns=['New ACC', 'Diff with Pool ACC', 'Removed Feature', 'Dataset'])
df_results.sort_values(by=['Dataset','Diff with Pool ACC'], inplace=True)
df_results


# In[ ]:


resFile = './results2_LOOCV/'+'FeatImpbyRemoval-LOOCV_SVMlinear_ACC.csv'
df_results.to_csv(resFile, index=False)


# Hf with ML!
# 
# @muntisa
