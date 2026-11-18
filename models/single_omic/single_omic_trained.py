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

#importo i dataset di training e di test
SNPs_train= pd.read_csv("SNPs_train_selected.csv", index_col=0)
SNPs_test= pd.read_csv("SNPs_test_selected.csv", index_col=0)

#estraggo la colonna dei dataset iniziali che contiengono le etichette per definire il target e 
#le converto in numeri (0 per AD, 1 per CN)
target_train = SNPs_train['Mode']
encoder = LabelEncoder()
target_train_encoded = encoder.fit_transform(target_train)

target_test = SNPs_test['Mode']
encoder = LabelEncoder()
target_test_encoded = encoder.fit_transform(target_test)

# definisco le matrici X_train e X_test su cui l'algoritmo verrà, rispettivamente, addestrato e testato. 
#Rimuovo l'ultima colonna che contiene le etichette.
X_train= SNPs_train.iloc[:,:-1]
X_test= SNPs_test.iloc[:,:-1]

#definisco y_train e y_test che contengono le etichette codificate, rispettivamente, per i pazienti del dataset di training e di test
y_train= target_train_encoded
y_test= target_test_encoded 

## definisco lo scaler per la normalizzazione
scaler = StandardScaler()

#applico lo scaler ad entrambi i datatset
X_train_scaled= scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

#converto gli array creati dallo scaler in DataFrame, ripristinando i nomi di pazienti(indici) e mutazioni(features)
X_train_df= pd.DataFrame(X_train_scaled, index= X_train.index, columns= X_train.columns)
X_test_df= pd.DataFrame(X_test_scaled, index= X_test.index, columns= X_test.columns)

#definisco il classificatore
clf = RandomForestClassifier(max_features=500, n_estimators=300, random_state=42) 

#addestro il classificatore sui dati di training
clf.fit(X_train_df, y_train) 

#testo sui dati di test, racchiudendo le predizioni nella variabile y_pred
y_pred= clf.predict(X_test_df) 

#calcolo le metriche confrontando le etichette del dataset di test con quelle predette dal classificatore
accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, zero_division= np.nan)

recall = recall_score(y_test, y_pred)

F1_score = f1_score(y_test, y_pred)

MCC= matthews_corrcoef(y_test, y_pred)

scores = pd.Series([accuracy, precision, recall, F1_score, MCC], index= ['accuracy', 'precision', 'recall', 'F1 score', 'MCC'])
print(scores)

#inizializzo SHAP
shap.initjs()

#inizializzo l'explainer sul modello che voglio spiegare
explainer = shap.TreeExplainer(clf)

#calcolo gli SHAP values, in questo caso sui dati di training
shap_values = explainer.shap_values(X_train)

#plotto gli SHAP values sia come bar plot (spiegazione globale) che come beeswarm plot (spiegazione locale)
print("Variable Importance Plot - Global Interpretation")
figure = plt.figure()
shap.summary_plot(shap_values[:, :, 0], X_train, max_display=10, plot_type='bar')
print("Variable Importance Plot - Local Interpretation for AD class")
figure = plt.figure()
shap.summary_plot(shap_values[:, :, 0], X_train, max_display=10)