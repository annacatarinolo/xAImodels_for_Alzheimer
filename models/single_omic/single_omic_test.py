# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:36:08 2026

@author: catarinolo
"""

import shap
import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler

#Loading training and test datasets with the selected mutations
SNPs_train= pd.read_csv("SNPs_train_selected.csv", index_col=0)
SNPs_test= pd.read_csv("SNPs_test_selected.csv", index_col=0)

#Extracting the column from the initial datasets that contains the labels to define the target and
#convert them into numbers (0 for AD, 1 for CN)
target_train = SNPs_train['Mode']
encoder = LabelEncoder()
target_train_encoded = encoder.fit_transform(target_train)

target_test = SNPs_test['Mode']
encoder = LabelEncoder()
target_test_encoded = encoder.fit_transform(target_test)

# Defining the X_train and X_test matrices, on which the algorithm will be trained and tested, respectively.
# Last column, which contains the labels, is removed.
X_train= SNPs_train.iloc[:,:-1]
X_test= SNPs_test.iloc[:,:-1]

#Defining y_train and y_test, which contain the encoded labels, respectively,
#for the patients in the training and test datasets.
y_train= target_train_encoded
y_test= target_test_encoded 

#Defining the scaler
scaler = StandardScaler()

#applico lo scaler ad entrambi i datatset
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

#Converting the arrays created by the scaler into DataFrames, restoring the names of 
#patients (indices) and mutations (features)
X_train_df= pd.DataFrame(X_train_scaled, index= X_train.index, columns= X_train.columns)
X_test_df= pd.DataFrame(X_test_scaled, index= X_test.index, columns= X_test.columns)

#defining the classier
clf = RandomForestClassifier(max_features=500, n_estimators=300, random_state=42) 

#training the classifier on training data
clf.fit(X_train_df, y_train) 

#textin on the test data, storing the predictions in the variable y_pred
y_pred= clf.predict(X_test_df) 

#Calculating the metrics by comparing the labels in the test dataset with those 
#predicted by the classifier
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division= np.nan)

recall = recall_score(y_test, y_pred)

F1_score = f1_score(y_test, y_pred)

MCC= matthews_corrcoef(y_test, y_pred)

scores = pd.Series([accuracy, precision, recall, F1_score, MCC], index= ['accuracy', 'precision', 'recall', 'F1 score', 'MCC'])
print(scores)

#inizialing SHAP 
shap.initjs()

#Initializing the explainer on the model I want to explain
explainer = shap.TreeExplainer(clf)

#calculating the SHAP values, in this case using the training data
shap_values = explainer.shap_values(X_train)

#plotting the SHAP values as both a bar plot (global explanation) and a 
#beeswarm plot (local explanation)
print("Variable Importance Plot - Global Interpretation")
figure = plt.figure()
shap.summary_plot(shap_values[:, :, 0], X_train, max_display=10, plot_type='bar')
print("Variable Importance Plot - Local Interpretation for AD class")
figure = plt.figure()
shap.summary_plot(shap_values[:, :, 0], X_train, max_display=10)