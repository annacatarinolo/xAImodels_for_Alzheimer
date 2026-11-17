# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:08:06 2026

@author: catarinolo
"""

import shap
import muon as mu
import numpy as np
import pandas as pd
import mudata as md
import mofax as mofa
import seaborn as sns
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix

#Carico i dataset dei valori di espressione genica e delle mutazioni
df_genes = pd.read_csv("de_genes_renamed.csv", index_col=0)
df_snp= pd.read_csv("snp_selected.csv", index_col=0)

# Definisco gli scaler
scaler_ge = StandardScaler()
scaler_snp = StandardScaler()

# Definisco il target
target = df_genes['BL_Diagnosis']

# Definisco le matrici di dati rimuovendo l'ultima colonna che contiene le etichette
X_ge= df_genes.iloc[:,:-1]
X_snp= df_snp.iloc[:,:-1]

# Codifico il target
encoder = LabelEncoder()
target_encoded = encoder.fit_transform(target)
y= target_encoded

# Definisco il classificatore
clf= RandomForestClassifier(random_state=42, n_estimators=200, max_depth=8)
clf_name="RandomForest"

# Divido i pazienti per il training e quelli per il test (10% test, 90% train)
train_idx, test_idx = train_test_split(df_genes.index, test_size=0.1, random_state=42)

# Ricavo i dataset di training e test sia per i geni che per gli snp
ge_train= X_ge.loc[train_idx]
ge_test= X_ge.loc[test_idx]

snp_train= X_snp.loc[train_idx]
snp_test= X_snp.loc[test_idx]

# Normalizzo i dati e li riporto a dataframe per 
# conservare le informazioni su pazienti (righe) e features (colonne)

ge_train_scaled = scaler_ge.fit_transform(ge_train)
ge_test_scaled  = scaler_ge.transform(ge_test)

ge_train_scaled_df = pd.DataFrame(ge_train_scaled, index=ge_train.index, columns=ge_train.columns)
ge_test_scaled_df = pd.DataFrame(ge_test_scaled, index=ge_test.index, columns=ge_test.columns)

snp_train_scaled = scaler_snp.fit_transform(snp_train)
snp_test_scaled  = scaler_snp.transform(snp_test)

snp_train_scaled_df = pd.DataFrame(snp_train_scaled, index=snp_train.index, columns=snp_train.columns)
snp_test_scaled_df = pd.DataFrame(snp_test_scaled, index=snp_test.index, columns=snp_test.columns)

# Divido anche il target in training e test e lo codifico
target_train = target.loc[train_idx]
target_test  = target.loc[test_idx]

y_train= encoder.fit_transform(target_train)
y_test= encoder.transform(target_test)

# Creo gli AnnData sia per geni che per snp, 
# necessari per creare il MuData richiesto da MOFA

adata_ge = AnnData(X=ge_train_scaled_df)
adata_snp = AnnData(X=snp_train_scaled_df)

# Definisco il numero di fattori per il modello MOFA
n_factors= 50

# Creo il MuData di train unendo i due AnnData
mdata = md.MuData({'genes': adata_ge, 'mut': adata_snp})

# Addestro il modello MOFA e lo salvo come file hdf5
mu.tl.mofa(
    mdata,
    n_factors=n_factors,
    convergence_mode='medium',
    outfile=f"mofax/model_final_{n_factors}.hdf5"
)

# Carico il modello precedentemente addestrato e salvato
model = mofa.mofa_model(f"mofax/model_final_{n_factors}.hdf5")
# Visualizzo i campioni del modello
model.get_samples()

# Estraggo la matrice dei pesi addestrati dal modello
trained_weights = model.get_weights(df=True)
trained_weights_ge = model.get_weights(df=True, views='genes')
trained_weights_mut=model.get_weights(df=True, views='mut')

# Estraggo la matrice dei fattori creata dal modello
Z_train = model.get_factors(df=True)

# PRIMO PLOT: barplot del coefficiente di determinazione R^2, che dà indicazioni sulla proporzione di varianza spiegata da ogni fattore.
#Le righe di codice che seguono servono a :
    
#estrarre i valori di R^2 dal modello MOFA sotto forma di DataFrame
r2= model.get_r2()

#impostare i fattori come indici
r2= r2.set_index('Factor')

#rimuovere la colonna del gruppo, che nel nostro caso è solo uno e non serve per il plot
r2= r2.drop('Group', axis=1)

#resettare l'indice e mantenere solo il valore più alto per ciascun fattore in base alla vista corrispondente
r2 = r2.reset_index()
r2 = r2.loc[r2.groupby('Factor')['R2'].idxmax()]

#mettere in ordine gli indici e impostare i fattori
r2 = r2.sort_index()
r2= r2.set_index('Factor')

#stoppare il plot al fattore 30, poichè già dal 29esimo il valore di R^2 è trascurabile
r2 = r2.iloc[:30]

#scalare i valori in scala logaritmica
r2['R2_log'] = np.log1p(r2['R2'])

#colorare i fattori secondo le viste (SNP e geni) e aggiungere la legenda
color_map = {'genes': 'orange', 'mut': 'blue'}
colors= r2['View'].map(color_map)

legend_handles = [
    mpatches.Patch(color='orange', label='genes'),
    mpatches.Patch(color='blue', label='mut')
]

plt.figure(figsize=(8,6))
plt.bar(r2.index, r2['R2_log'], color=colors)
plt.grid(False)
plt.tight_layout()
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('R2_log', fontsize=9)
plt.legend(handles= legend_handles, 
           title='Views', 
           fontsize=12,
           title_fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.xticks(fontsize=8)
plt.yticks(np.arange(0, 4.4, 0.5), fontsize=8)
plt.show()

# Creo gli AnnData di test
adata_ge_test = AnnData(X=ge_test_scaled_df)
adata_snp_test = AnnData(X=snp_test_scaled_df)

# Creo il MuData di test
mdata_test = md.MuData({'genes': adata_ge_test, 'mut': adata_snp_test})
# mdata_test

# Creo il MuData di test sottoforma di DataFrame, servirà nel calcolo della pseudoinversa
mdata_test_all = mdata_test[mdata_test.mod_names[0]].to_df()
for elem in mdata_test.mod_names[1:]:
  if elem != 'genes':
    mdata_test_all = pd.concat((mdata_test_all,mdata_test[elem].to_df()),axis=1)
mdata_test_all

#visualizzo le osservazioni, quindi i pazienti del MuData di test
mdata_test['genes'].obs

# # CALCOLO DELLA PSEUDOINVERSA
# # qui ricavo la proiezione nello spazio dei fattori dei campioni di test
Z_test = np.dot(mdata_test_all, np.linalg.pinv(trained_weights.T) )
Z_test_df= pd.DataFrame(Z_test, index= ge_test.index, columns=Z_train.columns)

# Addestro il classificatore con la matrice dei fattori creata dal modello con i pazienti di training
clf.fit(Z_train, y_train)

# predizione sul dataset di test, che è la 
# nuova matrice dei fattori risultante dalla pseudo-inversione
y_pred = clf.predict(Z_test_df)

# calcolo le metriche
accuracy = accuracy_score(y_test, y_pred)
precision= precision_score(y_test, y_pred)
recall= recall_score(y_test, y_pred)
f1_score= f1_score(y_test, y_pred)
mcc= matthews_corrcoef(y_test, y_pred)
cm= confusion_matrix(y_test, y_pred)

scores = pd.Series([accuracy, precision, recall, f1_score, mcc], index= ['accuracy', 'precision', 'recall', 'F1_score', 'MCC'])
print(scores)

# inizializzo SHAP per l'explainability
shap.initjs()

# Creo l' explainer
explainer = shap.TreeExplainer(clf)

#SECONDO PLOT: barplot dei valori globali di SHAP
# calcolo gli SHAP values per sula matrice dei fattori di test
shap_values = explainer.shap_values(Z_test_df)

# plotto gli SHAP values colorando in maniera diversa 
# i fattori associati ai geni e quelli associati agli snp
feature_names = Z_test_df.columns
shap_mean = np.abs(shap_values).mean(axis=(0, 2))

df_shap = pd.DataFrame({
    'feature': feature_names,
    'importance': shap_mean
})

df_shap = df_shap.sort_values('importance', ascending=False).head(30)
df_shap= df_shap.set_index(df_shap.columns[0])

feature_to_view = r2['View'].to_dict()

df_shap['views'] = df_shap.index.map(feature_to_view)

colors= df_shap['views'].map(color_map)

legend_handles = [
    mpatches.Patch(color='orange', label='genes'),
    mpatches.Patch(color='blue', label='mut')
]

plt.figure(figsize=(8,6))
plt.bar(df_shap.index, df_shap['importance'], color=colors)
plt.grid(False)
plt.tight_layout()
plt.xticks(rotation=45, fontsize=10)
plt.ylabel('SHAP_values', fontsize=9)
plt.legend(handles=legend_handles, title="Views", fontsize=12,
    title_fontsize=14)
plt.show()

#TERZO PLOT: heatmap che associa ai migliori 5 fattori secondo SHAP le corrispondenti features biologiche
# creo la lista dei 5 fattori più importanti secondo il plot SHAP
factors = [1, 6, 10, 24, 29]

# creo un dizionario che associa a questi fattori le 10 features con peso maggiore
top_features = {}

for f in factors:
    col = trained_weights.columns[f-1]
    
    top = trained_weights[col].abs().nlargest(10)
    selected = trained_weights.loc[top.index, col]
    
    top_features[f] = selected

# visualizzo le features comuni tra più fattori
# common_24_29 = set(top_features[24].index).intersection(set(top_features[29].index))
# common_1_6 = set(top_features[6].index).intersection(set(top_features[10].index))

# creo la lista delle features più importanti
top_features_list= top_features[1].index.append(top_features[6].index).append(top_features[29].index).append(top_features[24].index).append(top_features[10].index).drop_duplicates()

# creo il subset per il plot che associa le features  ai fattori
subset = trained_weights.loc[list(top_features_list), ['Factor1','Factor6', 'Factor29', 'Factor24', 'Factor10']]

# creo una funzione per normalizzare i pesi da 0 a 1, quindi in valore assoluto 
def normalize_abs_01(series):
    series = series.astype(float)
    
    abs_series = series.abs()
    max_val = abs_series.max()
    
    if max_val == 0:
        return pd.Series(0, index=series.index)
    
    return abs_series / max_val

# applico la funzione a ciascuna serie del dizionario, quindi a ciascun fattore
subset1_norm=normalize_abs_01(top_features[1])
subset6_norm=normalize_abs_01(top_features[6])
subset10_norm=normalize_abs_01(top_features[10]) 
subset24_norm=normalize_abs_01(top_features[24])
subset29_norm=normalize_abs_01(top_features[29])

# Creo il dataframe con i valori normalizzati e creo la heatmap che associa ad ognuno dei migliori 5 fattori (asse x) 
#le 10 features biologihe più significative (asse y)

df_normalized= pd.concat([subset1_norm, subset6_norm, subset29_norm, subset24_norm, subset10_norm], axis=1)

plt.figure(figsize=(15, 15))
sns.set(font_scale=1.5)

ax= sns.heatmap(df_normalized, cmap='Reds',linewidths=0.2)

ax.set_xticks(range(len(df_normalized.columns)))
ax.set_xticks(np.arange(df_normalized.shape[1]) + 0.5)
ax.set_xticklabels(df_normalized.columns, ha='center')
ax.set_yticklabels(df_normalized.index)

# plt.title("Top features per Factor 1, 6, 29, 24, 10")
plt.tight_layout()
plt.show()