# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 10:36:57 2026

@author: catarinolo
"""

import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix


#Loading the dictionary that links each SNP with the corresponding chromosome
SNP_CHR= pd.read_csv("models/single_omic/SNP_CHR_vcf_IGAP_5e-5.csv", index_col=0)

all_results = []

#Iteration over the pairs of chromosomes (index i). For other chromosomes simply add 
#further indices, in the form of nested loops. It requires a while to complete the training.

for i in range(1,23): 
    
    for j in range (1, 23):
        
    #excluding chromosome 21 as there are no mutations corresponding to it. 
    #The condition i<j allows to consider all the combinations without repetitions.
    
        if i!=21 and j!=21 and i < j: 
        
            #extracting lists of the mutations corresponding to the considered chromosomes
            CHR_i=SNP_CHR[SNP_CHR['#CHROM']==i] 
            CHR_i_list= CHR_i.index.tolist()
    
            CHR_j=SNP_CHR[SNP_CHR['#CHROM']==j] 
            CHR_j_list= CHR_j.index.tolist()
            
            #summing up the two lists
            CHR_list= CHR_i_list + CHR_j_list
            
            #loading the training datasets with all mutations
            SNPs_train= pd.read_csv("datasets/single_omic/SNPs_train_all.csv", index_col=0)
            
            #defining target (in the last columns)
            target = SNPs_train['Mode']
            encoder = LabelEncoder()
            target_encoded = encoder.fit_transform(target)
    
            #selecting the mutations (in the columns)
            SNPs_select= SNPs_train.loc[:, CHR_list]
    
            X= SNPs_select
            y= target_encoded
    
            skf= StratifiedKFold(n_splits = 10, shuffle= True, random_state= 35)
    
            #INITIALIZING VARIABLES
    
            fold_accuracies=[]
            fold_precisions= []
            fold_recalls= []
            fold_f1=[]
            fold_mcc=[]
    
            y_true_list= []
            y_pred_list= []
            
            for train_index, test_index in skf.split(X, y):
               
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                y_true_list.extend(y_test)
                
                #NORMALIZATION
                
                scaler = StandardScaler()
                
                X_train_scaled= scaler.fit_transform(X_train)
                X_test_scaled= scaler.transform(X_test)
                
                # defining the classifier
                clf= RandomForestClassifier(max_features=500, n_estimators=300, random_state=42)
                
                clf.fit(X_train_scaled, y_train) 
                
                y_pred= clf.predict(X_test_scaled) 
                
                y_pred_list.extend(y_pred)
                
                #computing scores for each fold
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                
                precision = precision_score(y_test, y_pred, zero_division= np.nan)
                fold_precisions.append(precision)
                
                recall = recall_score(y_test, y_pred)
                fold_recalls.append(recall)
                
                F1_score = f1_score(y_test, y_pred)
                fold_f1.append(F1_score)
                
                MCC= matthews_corrcoef(y_test, y_pred)
                fold_mcc.append(MCC)
            
            #computing the mean scores over the 10 iterations
            acc_mean = np.mean(fold_accuracies)
            pre_mean = np.nanmean(fold_precisions)
            rec_mean = np.mean(fold_recalls)
            f1_mean = np.mean(fold_f1)
            mcc_mean = np.mean(fold_mcc)
            
            cm = confusion_matrix(y_true_list, y_pred_list)

            cm_ls = cm.tolist()
            
            # Saving the results of all combinations of chromosomes
            all_results.append({
                'Chromosomes': f'CHR_{i}_{j}',
                'Accuracy_mean': acc_mean,
                'Precision_mean': pre_mean,
                'Recall_mean': rec_mean,
                'F1_mean': f1_mean,
                'MCC_mean': mcc_mean,
                'Confusion_matrix': cm_ls
            })

#converting the resulting dict in a DataFrame
all_results = pd.DataFrame(all_results)

# Ordering according to the mean value of accuracy
all_results_sorted = all_results.sort_values(
    by='Accuracy_mean',
    ascending=False
)

# The best combination is the first element of the sorted DataFrame
best_result = all_results_sorted.iloc[0]

# Creating the DataFrame of the mean scores
mean_scores = pd.DataFrame({
    'Mean Score': [
        best_result['Accuracy_mean'],
        best_result['Precision_mean'],
        best_result['Recall_mean'],
        best_result['F1_mean'],
        best_result['MCC_mean']
    ]
}, index=['Accuracy', 'Precision', 'Recall', 'F1 score', 'MCC'])

# Printing the obtained results
print("\n=== BEST PAIR ===")
print(best_result['Chromosomes'])

print("\n=== Mean scores (10-fold CV) ===")
print(mean_scores)

cm_df = pd.DataFrame(
    best_result['Confusion_matrix']
)

print("\n=== Confusion Matrix (10-fold CV) ===")
print(cm_df)

# Here we start from the training dataset selected with the mutation belonging to the 
#best combination, i.e. chromosomes 9, 13, 14, 17, 19
SNPs_selected= pd.read_csv('datasets/single_omic/SNPs_train_selected.csv', index_col=0)

target = SNPs_selected['Mode']
encoder = LabelEncoder()
target_encoded = encoder.fit_transform(target)

X= SNPs_selected.iloc[:,:-1]
y= target_encoded

skf= StratifiedKFold(n_splits = 10, shuffle= True, random_state= 35)

#INITIALIZING VARIABLES
fold_accuracies=[]
fold_precisions= []
fold_recalls= []
fold_f1=[]
fold_mcc=[]

y_true_list= []
y_pred_list= []

for train_index, test_index in skf.split(X, y):
   
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    y_true_list.extend(y_test)
    
    #NORMALIZATION
    
    scaler = StandardScaler()
    
    X_train_scaled= scaler.fit_transform(X_train)
    X_test_scaled= scaler.transform(X_test)
    
    clf = RandomForestClassifier(max_features=500, n_estimators=300, random_state=42) 
    
    clf.fit(X_train_scaled, y_train) 
    
    y_pred= clf.predict(X_test_scaled) 
    
    y_pred_list.extend(y_pred)
        
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    
    precision = precision_score(y_test, y_pred, zero_division= np.nan)
    fold_precisions.append(precision)
    
    recall = recall_score(y_test, y_pred)
    fold_recalls.append(recall)
    
    F1_score = f1_score(y_test, y_pred)
    fold_f1.append(F1_score)
    
    MCC= matthews_corrcoef(y_test, y_pred)
    fold_mcc.append(MCC)
    
    #computing scores for each fold
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    
    precision = precision_score(y_test, y_pred, zero_division= np.nan)
    fold_precisions.append(precision)
    
    recall = recall_score(y_test, y_pred)
    fold_recalls.append(recall)
    
    F1_score = f1_score(y_test, y_pred)
    fold_f1.append(F1_score)
    
    MCC= matthews_corrcoef(y_test, y_pred)
    fold_mcc.append(MCC)

    #computing the mean scores over the 10 iterations
    acc_mean = np.mean(fold_accuracies)
    pre_mean = np.nanmean(fold_precisions)
    rec_mean = np.mean(fold_recalls)
    f1_mean = np.mean(fold_f1)
    mcc_mean = np.mean(fold_mcc)
    
    cm = confusion_matrix(y_true_list, y_pred_list)
    
    cm_ls = cm.tolist()

# DataFrame degli score medi
mean_scores = pd.DataFrame({
    'Mean Score': [
        acc_mean,
        pre_mean,
        rec_mean,
        f1_mean,
        mcc_mean
    ]
}, index=['Accuracy', 'Precision', 'Recall', 'F1 score', 'MCC'])

# Printing the obtained results
print("\n=== BEST COMBINATION: CHROMOSOMES 9, 13, 14, 17, 19 ===")
print("\n=== Mean scores (10-fold CV) ===")
print(mean_scores)

# Confusion matrix aggregata su tutti i fold
cm = confusion_matrix(y_true_list, y_pred_list)

cm_df = pd.DataFrame(
    cm
)

print("\n=== Confusion Matrix (10-fold CV) ===")
print(cm_df)