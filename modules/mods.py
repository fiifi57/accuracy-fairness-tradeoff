#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur June 08 10:10am 2023

@authors: arhink@icloud.com
"""
import sys 

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.datasets import make_blobs
from  sklearn.datasets import make_classification
from  sklearn.datasets import make_regression
from matplotlib import pyplot
import pandas as pd    
import numpy as np

def generate_data(sample_size, n_features, informative_features, redundant_features, n_target, n_target_flip, random_state, pred_ease, col_prefix, target_weight=[.60,.40]):
    # function based on 
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html

    """
    sample_size = sample size
    n_features = nuber of features 
    n_target = number of cadres
    informative = random state
    seed = random state
    """

    X,y = make_classification(n_samples = sample_size, 
                              n_features = n_features, 
                              n_informative = informative_features,  
                              n_classes = n_target,  
                              random_state = random_state, 
                              n_redundant = redundant_features, 
                              class_sep = pred_ease, 
                              flip_y = n_target_flip,
                              weights = target_weight)

    c_columns = [col_prefix+str(x) for x in range (n_features)]
    df = pd.DataFrame(X, columns = c_columns)
    df['target'] = y
    df['id'] = df.index
    return df

def create_clusters(df, cluster_features, n_clusters, cluster_col_name, tool):
    from sklearn.cluster import KMeans
    from sklearn.cluster import DBSCAN
    
    z = df.copy()
    
    if tool == 'kmeans':
        print('clustering based on KMEANS')
        print()
        kmeans = KMeans(n_clusters = n_clusters)
        kmeans.fit(z[cluster_features])
        z[cluster_col_name] = kmeans.labels_
        return z
    elif tool == 'dbscan':
        print('clustering based on DBSCAN')
        print()
        dbs = DBSCAN(eps = n_clusters)
        dbs.fit(z[cluster_features])
        z[cluster_col_name] = dbs.labels_
        return z
    else:
        print('sorry, the only "tool" options available right now are "kmeans" or "dbscan"')
        print()

def append_preds(main, x_data, y_data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    logr = LogisticRegression()
    rforest = RandomForestClassifier()
    
    scimodel = [logr, rforest]
    
    df = main.copy()
    
    if (len(x_data) == 3) and (len(y_data) == 3):
        df1 = x_data[0].copy()
        df2 = x_data[1].copy()
        df3 = x_data[2].copy()
        
        dfs_x = [x_data[0], x_data[1]]
        X_train = pd.concat(dfs_x)
        y_train =  y_data[0].append(y_data[1])
        X_test =  x_data[2]
        y_test = y_data[2]
        
        for s in scimodel:
            print('now predicting with ', s)
            s.fit(X_train, y_train)
            preds = s.predict(X_test)
       
            df[type(s).__name__+'_preds'] = preds
        
        return df
    
    else:
        print('!DataframeError: expected 2 frames each, got ', len(x_data), ' for X and ', len(y_data), ' for y')

def get_accuracy(y_true, y_predict):
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_true, y_predict)

def get_balanced_accuracy(y_true, y_predict):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y_true, y_predict)

def assess_preds(df, target_col, group_col):
    z = df.copy()
    # select out t5 cols
    pred_cols=[]
    for col in z.keys():
        if 'pred' in col:
            pred_cols.append(col)
    print('found: ', pred_cols)
    print()
    
    unique_groups = z[group_col].unique()
    new_df = pd.DataFrame()
    new_df['groups'] = unique_groups
    
    for pr in pred_cols:
        print('accuracy for',pr, ':', round(get_accuracy(z[target_col], z[pr]), 2))
        maj = z[z[group_col]=='majority']
        maj_ratio = len(maj[maj[pr]==1])/len(maj)
        
        group_ratios=[]
        for g in z[group_col].unique():
            n = z[z[group_col]==g]
            total_g = len(n)
            total_sel = len(n[n[pr]==1])
            sr = total_sel/total_g
            air = sr/maj_ratio
            group_ratios.append(air)
        new_df[pr] = group_ratios
    print()
    return new_df
        
def get_cleanlab_labels(main, x_data, y_data, target_col):
    import cleanlab
    from cleanlab.filter import find_label_issues
    from xgboost import XGBClassifier
    from sklearn.svm import NuSVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from skclean.models import RobustLR
    #import lightgbm
    
    logr = LogisticRegression()
    rforest = RandomForestClassifier()
    xgb = XGBClassifier()
    nsvc = NuSVC(probability=True)
    dtree = DecisionTreeClassifier(max_depth=8)
    knn = KNeighborsClassifier() #(n_neighbors=50, weights='distance', leaf_size=5)
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()
    mnb = MultinomialNB()
    gnb = GaussianNB()
    adaboost = AdaBoostClassifier()
    gradboost = GradientBoostingClassifier()
    enet = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001)
    rblr = RobustLR()
    #lgb = LGBMClassifier()
    
    scimodels = [rblr]

    # select metrics you want to see
    
    """
    x_data: should be arranged strictly in this order - train, val, test
    y_data: should be arranged strictly in this order to match x_data - train, val, test
    pred_path: location to save csv file with predictions
    
    """
    placeholder = main.copy()
    
    if (len(x_data) == 3) and (len(y_data) == 3):
        df1 = x_data[0].copy()
        df2 = x_data[1].copy()
        df3 = x_data[2].copy()
        
        dfs_x = [x_data[0], x_data[1]]
        X_train = pd.concat(dfs_x)
        y_train =  y_data[0].append(y_data[1])
        X_test =  x_data[2]
        y_test = y_data[2]
        
        for s in scimodels:

            # Feed the training data through the pipeline
            fp = s.fit(X_train, y_train)
            preds = fp.predict(X_test)
            #probs = fp.predict_proba(X_test)[:, 1]
            probs_full = fp.predict_proba(X_test)
            
        issues = find_label_issues(y_test, probs_full)
        placeholder['cleanlab_issue'] = issues
            
        new_labels = []
        for index,row in placeholder.iterrows():
            if (row['cleanlab_issue']==True) & (row[target_col]==0):
                new_labels.append(1)
            elif (row['cleanlab_issue']==True) & (row[target_col]==1):
                new_labels.append(0)
            elif (row['cleanlab_issue']==False) & (row[target_col]==1):
                new_labels.append(1)
            elif (row['cleanlab_issue']==False) & (row[target_col]==0):
                new_labels.append(0)
            else:
                new_labels.append('-')
            
        placeholder['cleanlab_groundtruth']=new_labels
        
        return placeholder
    
    else:
        print('!DataframeError: expected 4 frames each for X and y, got ', len(x_data), ' for X and ', len(y_data), ' for y')    


        
