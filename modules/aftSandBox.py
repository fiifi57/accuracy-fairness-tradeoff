#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thursday June 27 2024

@authors: kofi.arhin@lehigh.edu, rik224@lehigh.edu
"""
# import relibraries ###########################################################################################################
import sys 
import warnings
warnings.filterwarnings('ignore') 

import os
import re
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification # module to create synthetic datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix
from datetime import datetime as dt

from sklearnex import patch_sklearn 
patch_sklearn()

# create protected class data ################################################################################################
def create_protected_class(n_size, protected_percentages, feature_size, informative_features, target_label):
    # module can be found here: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
    '''
    n_size                  : sample size 
    protected_percentages   : percentage of each protected class 
    feature_size            : number of features or predictors
    informative_features    : number of informative features in set of predictors
    target_label            : name for target variable column
    '''
    X,y = make_classification(
        n_samples=n_size,
        n_features=feature_size,
        n_redundant=0,
        n_informative=informative_features,
        n_classes=3,
        weights=protected_percentages,
    )
    p_columns = ['proc_feature_'+str(x) for x in range(informative_features)] # prefix for the protected class features
    protected_df = pd.DataFrame(X, columns = p_columns) # save data
    protected_df[target_label] = y # save target label in created datarame
    protected_df['protected_id'] = protected_df.index # create an index to track sample size
    
    return protected_df

# create main dataset #########################################################################################################
def create_dataset(sample_sizes, n_features, n_informative, n_classes, class_separations, p_weights, 
                   p_feature_size, p_informative, p_label, save_path, log=True):
    '''
    sample_sizes        : list of sample sizes to create datasets for
    n_features          : number of features or predictors
    n_informative       : number of informative features in set of predictors
    n_classes           : number of classes in dataset target
    class_separations   : list of class separations to create datasets for (low class separation  is similar to low standard deviation for e.g.)
    p_weights           : list of percentages for each protected class
    p_feature_size      : number of features or predictors for protected class
    p_informative       : number of informative features in set of predictors for protected class
    p_label             : name for target variable column for protected class
    save_path           : path to save datasets
    '''
    
    #print('grab a cup of coffee. this might take a while :)')
    for n_samples in sample_sizes:
        #print('working on sample size:', n_samples)
        # create protected class for each sample size
        protected_df = create_protected_class(n_size=n_samples,
                                              protected_percentages=p_weights,
                                              feature_size=p_feature_size,
                                              informative_features=p_informative,
                                              target_label=p_label,
                                              )
        
        for class_sep in class_separations:
            # Generate the synthetic dataset
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_classes=n_classes,
                class_sep=class_sep,
                n_redundant=0,
            )
            
            # Combine all features
            v_columns = ['var_'+str(x) for x in range(n_informative)] # prefix for the main dataset features
            main_df = pd.DataFrame(X, columns=v_columns) # save data in a pandas dataframe
            main_df['accept'] = y   # save target label in created datarame
            main_df['sample_id'] = main_df.index # create an index to track sample size
            
            final_df = pd.concat([main_df, protected_df], axis=1) # combine the main dataset with the protected class dataset

            # save files if log is True
            if log:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                dataset_filename = os.path.join(save_path, f'sample_{n_samples}_sep_{class_sep}.csv')
                final_df.to_csv(dataset_filename, index = False)

# flip labels #############################################################################################################
def flip_labels(target_col, protected_col, protected_classes_to_flip, flip_percentages, data_loc, save_path, log=True):
    '''
    target_col                  : name of target column
    protected_col               : name of protected class column
    protected_classes_to_flip   : list of protected class labels to flip
    flip_percentages            : list of flip percentages to flip  (e.g. 0.1, 0.2, 0.3)
    data_loc                    : path to datasets that need labels flipped
    save_path                   : path to save flipped datasets
    '''
    #print('grab a cup of coffee. this might take a while :)')
    for dataset in os.listdir(data_loc):
        df = pd.read_csv(os.path.join(data_loc, dataset)) # create an instance of each dataset in data folder
        for flip_percentage in flip_percentages:
            y_flipped = df[target_col].copy()
            for protected_class in protected_classes_to_flip:
                
                # Get the indices of the rows where the target is 1 and the protected class is the one we want to flip
                indices = np.where((df[target_col] == 1) & (df[protected_col] == protected_class))[0]
                
                # Randomly select a subset of the indices to flip
                n_flips = int(flip_percentage * len(indices))
                
                # Randomly select n_flips indices to flip
                flip_indices = np.random.choice(indices, n_flips, replace=False)
                
                y_flipped[flip_indices] = 0  # Flip Accept=1 to Accept=0
            df['flipped_accept'] = y_flipped
            
            if log:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    
                # The rsplit tells Python to perform the string splits starting from the right of the string, and the 1 says to perform at most one split 
                flipped_filename = os.path.join(save_path, f'{dataset.rsplit( ".", 1 )[ 0 ]}_flipped_{flip_percentage}.csv')
                df.to_csv(flipped_filename, index=False)

# run predictions ##########################################################################################################
def run_predictions(data_loc, X_features_drop, y_target, scimodels, experiment_name, protected_col, ground_truth_label, results_file, study_num, log=True, oversample=False):
    '''
    data_loc            : path to datasets to run predictions on
    X_features_drop     : features to drop from the dataset before running predictions
    y_target            : target variable
    scimodels           : list of models to run predictions on
    experiment_name     : name of the experiment
    protected_col       : name of protected class column
    ground_truth_label  : name of ground truth label (this contains true values that were flipped in the "flip_labels" function)
    results_file        : path to save results
    '''
    
    # this module speeds up the prediction process for some basic sklearn algorithms. read more here https://pypi.org/project/scikit-learn-intelex/
    from sklearnex import patch_sklearn 
    patch_sklearn()
    
    results_df = pd.DataFrame() # empty dataframe to store results from predictions
    average_probs = [] # empty list to store average probabilities (i.e., predict_proba()) from each model
    std_probs = []   # empty list to store standard deviation of probabilities from each model
    skmodel = [] # empty list to store algorithm or model names
    
    train_sr_all = [] # empty list to store base selection ratio for all samples
    train_sr_0 = []  # empty list to store base selection ratio for protected class 0
    train_sr_1 = []  # empty list to store base selection ratio for protected class 1
    train_sr_2 = []  # empty list to store base selection ratio for protected class 2
    
    train_air_1 = [] # empty list to store base adverse impact ratio for protected class 1
    train_air_2 = [] # empty list to store base adverse impact ratio for protected class 2
    
    gt_train_cm_all = [] # empty list to store groundtruth confusion matrix for all samples in the training set
    gt_train_cm_p0 = [] # empty list to store groundtruth confusion matrix for all samples in the training set
    gt_train_cm_p1 = [] # empty list to store groundtruth confusion matrix for all samples in the training set
    gt_train_cm_p2 = [] # empty list to store groundtruth confusion matrix for all samples in the training set
    
    pred_sr_all = [] # empty list to store predicted selection ratio for all samples
    pred_sr_0 = []  # empty list to store predicted selection ratio for protected class 0
    pred_sr_1 = []  # empty list to store predicted selection ratio for protected class 1
    pred_sr_2 = []  # empty list to store predicted selection ratio for protected class 2
    
    pred_air_1 = [] # empty list to store predicted adverse impact ratio for protected class 1
    pred_air_2 = [] # empty list to store predicted adverse impact ratio for protected class 2
    
    bal_accuracy = [] # empty list to store balanced accuracy scores
    accuracy = [] # empty list to store accuracy scores
    dataset_name = [] # empty list to store dataset names
    ground_truth_accuracy = [] # empty list to store ground truth accuracy scores
    ground_truth_bal_accuracy = [] # empty list to store ground truth balanced accuracy scores
    
    accuracy_p0 = [] # empty list to store balanced accuracy scores for protected class 0
    accuracy_p1 = [] # empty list to store balanced accuracy scores for protected class 1
    accuracy_p2 = [] # empty list to store balanced accuracy scores for protected class 2
    
    gt_accuracy_p0 = [] # empty list to store groundtruth balanced accuracy scores for protected class 0
    gt_accuracy_p1 = [] # empty list to store groundtruth balanced accuracy scores for protected class 1
    gt_accuracy_p2 = [] # empty list to store groundtruth balanced accuracy scores for protected class 2
    
    cm_p0 = [] # empty list to store confusion matrices for protected class 0
    cm_p1 = [] # empty list to store confusion matrices for protected class 1
    cm_p2 = [] # empty list to store confusion matrices for protected class 2
    
    gt_sr_0 = [] # empty list to store ground truth selection ratio for protected class 0
    gt_sr_1 = [] # empty list to store ground truth selection ratio for protected class 1
    gt_sr_2 = [] # empty list to store ground truth selection ratio for protected class 2
    gt_sr_all = [] # empty list to store ground truth selection ratio for all samples
    
    cm_p0_gt = [] # empty list to store confusion matrices for protected class 0 with groundtruth labels
    cm_p1_gt = [] # empty list to store confusion matrices for protected class 1 with groundtruth labels
    cm_p2_gt = [] # empty list to store confusion matrices for protected class 2 with groundtruth labels
    
    
    timestamp = [] # empty list to store timestamps
    
    #print('running predictions for ', len(os.listdir(data_loc)), 'datasets. grab a cup of coffee. this might take a while :)')
    
    for dataset in os.listdir(data_loc):
        df = pd.read_csv(os.path.join(data_loc, dataset)) # create an instance of each dataset in data folder
        
        y = df[y_target] # target variable
        X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.3, stratify=df[protected_col])
        
        if oversample:
            X_train, y_train = oversample_minority(features = X_train, 
                                                   target_values = y_train, 
                                                   target_label='flipped_accept', 
                                                   protected_col=protected_col
                                                   ) 
        else:
            pass
        
        X_train_2 = X_train.drop(X_features_drop, axis=1)
        X_test_2 = X_test.drop(X_features_drop, axis=1)
        
        for scimodel in scimodels:
            skmodel.append(type(scimodel).__name__)
            
            sr_0 = X_train[X_train[protected_col] == 0]['flipped_accept'].mean()
            sr_1 = X_train[X_train[protected_col] == 1]['flipped_accept'].mean()
            sr_2 = X_train[X_train[protected_col] == 2]['flipped_accept'].mean()
            
            gt_train_cm_all.append(confusion_matrix(X_train[ground_truth_label], y_train))
            train_0 = X_train[X_train[protected_col] == 0]
            gt_train_cm_p0.append(confusion_matrix(train_0[ground_truth_label], train_0['flipped_accept']))
            
            train_1 = X_train[X_train[protected_col] == 1]
            gt_train_cm_p1.append(confusion_matrix(train_1[ground_truth_label], train_1['flipped_accept']))
            
            train_2 = X_train[X_train[protected_col] == 2]
            gt_train_cm_p2.append(confusion_matrix(train_2[ground_truth_label], train_2['flipped_accept']))
            
            
            train_sr_all.append(round(X_train['flipped_accept'].mean(), 4))
            
            train_sr_0.append(round(sr_0, 4))
            train_sr_1.append(round(sr_1, 4))
            train_sr_2.append(round(sr_2, 4))
            
            train_air_1.append(round(sr_1/sr_0, 4))
            train_air_2.append(round(sr_2/sr_0, 4))
        
            dataset_name.append(dataset) # save dataset
            
            scimodel.fit(X_train_2, y_train)
            y_pred = scimodel.predict(X_test_2)
            bal_accuracy.append(balanced_accuracy_score(y_test, y_pred))
            accuracy.append(accuracy_score(y_test, y_pred))
            ground_truth_accuracy.append(accuracy_score(X_test[ground_truth_label], y_pred))
            ground_truth_bal_accuracy.append(balanced_accuracy_score(X_test[ground_truth_label], y_pred))
            
            X_test['pred_class'] = y_pred
            X_test['pred_probs'] = np.where(X_test['pred_class'] == 1, scimodel.predict_proba(X_test_2)[:, 1], scimodel.predict_proba(X_test_2)[:, 0])
            average_probs.append(round(X_test['pred_probs'].mean(), 4))
            std_probs.append(round(X_test['pred_probs'].std(), 4))
            
            pred_sr_all.append(round(X_test['pred_class'].mean(), 4))

            tsr_0 = X_test[X_test[protected_col] == 0]['pred_class'].mean()
            tsr_1 = X_test[X_test[protected_col] == 1]['pred_class'].mean()
            tsr_2 = X_test[X_test[protected_col] == 2]['pred_class'].mean()
            
            gt_sr_0.append(round(X_test[X_test[protected_col] == 0]['accept'].mean(), 4))
            gt_sr_1.append(round(X_test[X_test[protected_col] == 1]['accept'].mean(), 4))
            gt_sr_2.append(round(X_test[X_test[protected_col] == 2]['accept'].mean(), 4))
            gt_sr_all.append(round(X_test['accept'].mean(), 4))
            
            pred_sr_0.append(round(tsr_0, 4))
            pred_sr_1.append(round(tsr_1, 4))
            pred_sr_2.append(round(tsr_2, 4))
            
            pred_air_1.append(round(tsr_1/tsr_0, 4))
            pred_air_2.append(round(tsr_2/tsr_0, 4))
            
            test_0 = X_test[X_test[protected_col] == 0]
            accuracy_p0.append(balanced_accuracy_score(test_0[y_target], test_0['pred_class']))
            gt_accuracy_p0.append(balanced_accuracy_score(test_0[ground_truth_label], test_0['pred_class']))
            cm_p0.append(confusion_matrix(test_0[y_target], test_0['pred_class']))
            cm_p0_gt.append(confusion_matrix(test_0[ground_truth_label], test_0['pred_class']))
            
            test_1 = X_test[X_test[protected_col] == 1]
            accuracy_p1.append(balanced_accuracy_score(test_1[y_target], test_1['pred_class']))
            gt_accuracy_p1.append(balanced_accuracy_score(test_1[ground_truth_label], test_1['pred_class']))
            cm_p1.append(confusion_matrix(test_1[y_target], test_1['pred_class']))
            cm_p1_gt.append(confusion_matrix(test_1[ground_truth_label], test_1['pred_class']))
            
            test_2 = X_test[X_test[protected_col] == 2]
            accuracy_p2.append(balanced_accuracy_score(test_2[y_target], test_2['pred_class']))
            gt_accuracy_p2.append(balanced_accuracy_score(test_2[y_target], test_2['pred_class']))
            cm_p2.append(confusion_matrix(test_2[y_target], test_2['pred_class']))
            cm_p2_gt.append(confusion_matrix(test_2[ground_truth_label], test_2['pred_class']))
            
            timestamp.append(dt.now().isoformat())
        
    results_df['skmodel'] = skmodel
    results_df['average_probs'] = average_probs
    results_df['std_probs'] = std_probs
    results_df['train_sr_all'] = train_sr_all
    results_df['train_sr_0'] = train_sr_0
    results_df['train_sr_1'] = train_sr_1
    results_df['train_sr_2'] = train_sr_2
    results_df['train_air_1'] = train_air_1
    results_df['train_air_2'] = train_air_2
    results_df['pred_sr_all'] = pred_sr_all
    results_df['pred_sr_0'] = pred_sr_0
    results_df['pred_sr_1'] = pred_sr_1
    results_df['pred_sr_2'] = pred_sr_2
    results_df['pred_air_1'] = pred_air_1
    results_df['pred_air_2'] = pred_air_2
    results_df['accuracy'] = accuracy
    results_df['bal_accuracy'] = bal_accuracy
    results_df['gt_accuracy'] = ground_truth_accuracy
    results_df['gt_bal_accuracy'] = ground_truth_bal_accuracy
    results_df['gt_sr_0'] = gt_sr_0
    results_df['gt_sr_1'] = gt_sr_1
    results_df['gt_sr_2'] = gt_sr_2
    results_df['gt_sr_all'] = gt_sr_all
    results_df['accuracy_p0'] = accuracy_p0
    results_df['accuracy_p1'] = accuracy_p1
    results_df['accuracy_p2'] = accuracy_p2
    results_df['gt_accuracy_p0'] = gt_accuracy_p0
    results_df['gt_accuracy_p1'] = gt_accuracy_p1
    results_df['gt_accuracy_p2'] = gt_accuracy_p2
    results_df['gt_train_cm_all'] = gt_train_cm_all
    results_df['gt_train_cm_p0'] = gt_train_cm_p0
    results_df['gt_train_cm_p1'] = gt_train_cm_p1
    results_df['gt_train_cm_p2'] = gt_train_cm_p2
    results_df['cm_p0'] = cm_p0
    results_df['cm_p1'] = cm_p1
    results_df['cm_p2'] = cm_p2
    results_df['cm_p0_gt'] = cm_p0_gt
    results_df['cm_p1_gt'] = cm_p1_gt
    results_df['cm_p2_gt'] = cm_p2_gt
    results_df['experiment_name'] = experiment_name
    results_df['dataset_name'] = dataset_name
    results_df['study'] = study_num
    results_df['timestamp'] = timestamp
    
    if log:
        if not os.path.exists(results_file):
            results_df.to_csv(results_file, index=False)
        else:
            results_df.to_csv(results_file, mode='a', header=False, index=False)
    
    #return results_df

# extract numbers from string ################################################################################################
def extract_numbers_from_string(df, string_col):
    '''
    df          : dataframe
    string_col  : column with string values
    
    NB:
    This function extracts numbers from a string column and creates new columns for each number extracted.
    It was purposely created to extract the sample size, class separation, and flip percentage from the "dataset_name" column
    in the run_predictions function.
    '''
    z = df.copy()
    sample_size = []
    class_sep = []
    flip_percentage = []
    
    for index, row in z.iterrows():
        string_nums = re.findall(r'[-+]?(?:\d*\.*\d+)', row[string_col])
        sample_size.append(int(string_nums[0]))
        class_sep.append(float(string_nums[1]))
        flip_percentage.append(float(string_nums[2]))
        
    z['sample_size'] = sample_size
    z['class_sep'] = class_sep
    z['flip_percentage'] = flip_percentage
    #df[['sample_size','class_sep','flip_percentage']] = df[['sample_size','class_sep','flip_percentage']].apply(pd.to_numeric)
    
    return z

# expand confusion matrix ################################################################################################
def expand_matrix_col(df, matrix_cols):
    '''
    df          : dataframe
    matrix_col  : column name with confusion matrix values
    prefix      : prefix for new columns
    '''
    z = df.copy() # create a copy of data to avoid chaning original
    print('data has', len(z.keys()), 'columns')
    count_null = 0
    for matrix_col in matrix_cols:
        tn = []
        fp = []
        fn = []
        tp = []
        
        for index, row in z.iterrows():
            # Given confusion matrix output as a string
            confusion_matrix_str = row[matrix_col]

            # Use regular expressions to extract the numbers
            numbers = re.findall(r'\d+', confusion_matrix_str)
            
            if len(numbers) == 4:
                temp_tn, temp_fp, temp_fn, temp_tp = map(int, numbers)
                tn.append(temp_tn)
                fp.append(temp_fp)
                fn.append(temp_fn)
                tp.append(temp_tp)
            else:
                count_null += 1
                tn.append(np.nan)
                fp.append(np.nan)
                fn.append(np.nan)
                tp.append(np.nan)
    
        z[str(matrix_col) + '_tn'] = tn
        z[str(matrix_col) + '_fp'] = fp
        z[str(matrix_col) + '_fn'] = fn
        z[str(matrix_col) + '_tp'] = tp
    
    print('number of null values:', count_null)
    
    # return dataframe with new columns
    print('data now has', len(z.keys()), 'columns')
    print(z.keys())
    return z

################################################################################################################################################################

def create_fair_impact(data_loc, save_path, drop_cols, maj_group, min_group, protected_class, label, disparate_repair_level, log=True):
    from aif360.algorithms.preprocessing import DisparateImpactRemover
    from aif360.datasets import BinaryLabelDataset
    
    for dataset in os.listdir(data_loc):
        df = pd.read_csv(os.path.join(data_loc, dataset)) # create an instance of each dataset in data folder
        z = df.copy()
        temp_z = z.drop(drop_cols, axis=1)
        dir_df = BinaryLabelDataset(df=temp_z, 
                                    label_names=label, 
                                    protected_attribute_names=[protected_class],
                                    privileged_protected_attributes = maj_group,
                                    unprivileged_protected_attributes = min_group,
                                    favorable_label=1, # accepted
                                    unfavorable_label=0, # rejected
        )
        
        di_remover = DisparateImpactRemover(repair_level=disparate_repair_level)
        dir_df_transformed = di_remover.fit_transform(dir_df)    
        z_transformed = dir_df_transformed.convert_to_dataframe()[0] # create the transformed dataframe
        
        # add dropped columns back
        for i in drop_cols:
            z_transformed[i] = z[i].values
        
        if log:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            dataset_filename = os.path.join(save_path, dataset)
            z_transformed.to_csv(dataset_filename, index = False)

##################################################################################################################################################################

def oversample_minority(features, target_values, target_label, protected_col):
    from imblearn.over_sampling import RandomOverSampler
    random_over = RandomOverSampler(sampling_strategy='minority')
    
    temp_df = features.copy()
    temp_df[target_label] = target_values
    
    temp_dfg1 = temp_df[temp_df[protected_col] == 1]
    temp_dfg2 = temp_df[temp_df[protected_col] == 2]
    temp_dfg0 = temp_df[temp_df[protected_col] == 0]
    
    min_dfs = [temp_dfg0, temp_dfg1, temp_dfg2]
    
    transformed_dfs = []
    count = 1
    
    for data_slice in min_dfs:
        X = data_slice.drop(columns=[target_label, protected_col], axis=1)
        y = data_slice[target_label]
        X_resampled, y_resampled = random_over.fit_resample(X, y)
        
        resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
        resampled_data[target_label] = y_resampled
        if count == 1:
            resampled_data[protected_col] = 1
        else:
            resampled_data[protected_col] = 2
        transformed_dfs.append(resampled_data)
        count += 1
    
    new_data = pd.concat(transformed_dfs, axis=0)
    # shuffle dataset
    new_data = new_data.sample(frac=1).reset_index(drop=True)
    
    X_train = new_data
    y_train = new_data[target_label]
    
    return X_train, y_train