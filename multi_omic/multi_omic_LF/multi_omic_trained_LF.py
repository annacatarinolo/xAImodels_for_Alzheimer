# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:52:59 2026

@author: catarinolo
"""

import pandas as pd
import numpy as np
import torch
import shap
from snp_gene_models_explain import WrappedModel, Multiomics_early
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ge_df= pd.read_csv("de_genes_renamed.csv", index_col=0)
snp_df= pd.read_csv("snp_selected.csv", index_col=0)

ge_tensor=torch.tensor(ge_df.iloc[:,:-1].values)
snp_tensor=torch.tensor(snp_df.iloc[:,:-1].values)

X_ge= ge_df.iloc[:,:-1].values
X_snp= snp_df.iloc[:,:-1].values

target= snp_df['Bl_DX']
encoder= LabelEncoder()
target_encoded= encoder.fit_transform(target)
y= target_encoded

skf= StratifiedKFold(n_splits=5, random_state=35, shuffle=True)

list_shap_values_LF = []
list_shap_values_EF = []
list_test_sets = []

for i, (train_index, test_index) in enumerate(skf.split(X_ge, y)):
    
    X_train_ge, X_test_ge = X_ge[train_index], X_ge[test_index]
    X_train_snp, X_test_snp = X_snp[train_index], X_snp[test_index]
    y_train, y_test = target[train_index], target[test_index]
    
    X_train_ge_df = pd.DataFrame(X_train_ge,columns=ge_df.iloc[:,:-1].columns, 
                                 index=ge_df.iloc[train_index, :].index)
    
    X_train_snp_df = pd.DataFrame(X_train_snp,columns=snp_df.iloc[:,:-1].columns, 
                                 index=snp_df.iloc[train_index, :].index)
    
    X_test_ge_df = pd.DataFrame(X_test_ge,columns=ge_df.iloc[:,:-1].columns, 
                                 index=ge_df.iloc[test_index,:].index)
    
    X_test_snp_df = pd.DataFrame(X_test_snp,columns=snp_df.iloc[:,:-1].columns, 
                                 index=snp_df.iloc[test_index,:].index)
    
    back= pd.concat([X_train_snp_df, X_train_ge_df], axis=1)
    back_tensor= torch.tensor(back.values)
    
    ge_train_tensor=torch.tensor(X_train_ge)
    snp_train_tensor=torch.tensor(X_train_snp)
    ge_test_tensor=torch.tensor(X_test_ge)
    snp_test_tensor=torch.tensor(X_test_snp)
    
    info_test= pd.concat([X_test_snp_df, X_test_ge_df], axis=1)
    info_test_tensor= torch.tensor(info_test.values)
    
    early_model = Multiomics_early(indims=717, layers_dim_output=[64, 64, 64]).double()
    checkp = torch.load("early_best.pth")
    early_model.load_state_dict(checkp)
    early_model.eval()

    new_model = WrappedModel(543).double()
    new_model.eval() 
    new_model(info_test_tensor)
    shap.initjs()

    #explaining early model
    explainer_LF = shap.DeepExplainer(new_model, back_tensor[130:])
    shap_values_LF = explainer_LF.shap_values(info_test_tensor,check_additivity=False).squeeze()
    #per ogni iterazione salviamo gli inidici del test_set e gli shap_values
    list_shap_values_LF.append(shap_values_LF)
    
    explainer_EF = shap.DeepExplainer(early_model, back_tensor[130:])
    shap_values_EF = explainer_EF.shap_values(info_test_tensor,check_additivity=False).squeeze()
    #per ogni iterazione salviamo gli inidici del test_set e gli shap_values
    list_shap_values_EF.append(shap_values_EF)
    
    list_test_sets.append(info_test)

# flatten list of lists, pick the sv for 1 class, stack the result
shap_values_LF = np.vstack(list_shap_values_LF)
shap_values_LF = np.abs(shap_values_LF).mean(0)   
importance_df_LF = pd.DataFrame({
    "shap_values_LF":shap_values_LF 
}, index=info_test.columns)

shap_values_EF = np.vstack(list_shap_values_EF)
shap_values_EF = np.abs(shap_values_EF).mean(0)   
importance_df_EF = pd.DataFrame({
    "shap_values_EF":shap_values_EF 
}, index=info_test.columns)


feature_colors = ['tab:red']*543 + ['tab:blue']*174
shap_mean_abs_LF = np.abs(shap_values_LF).mean(axis=0) 
shap_mean_abs_EF = np.abs(shap_values_EF).mean(axis=0)  # [n_features]
feature_names= info_test.columns.tolist()

order = np.argsort(shap_mean_abs_LF)[::-1]
top_k = 100
top_order = order[:top_k]
shap_mean_abs_LF_sorted = shap_mean_abs_LF[top_order]
shap_mean_abs_EF_sorted = shap_mean_abs_EF[top_order]
feature_names_sorted = np.array(feature_names)[top_order]
colors_sorted = np.array(feature_colors)[top_order]

# y = np.arange(len(shap_mean_abs_sorted))
x_LF = np.arange(len(shap_mean_abs_LF_sorted))
x_EF = np.arange(len(shap_mean_abs_EF_sorted))

fig, axes = plt.subplots(2,1, figsize=(20, 6))
bar_width = 0.8 

# =========================
# LF
# =========================

axes[0].bar(x_LF, shap_mean_abs_LF_sorted, color=colors_sorted, width=bar_width)
axes[0].set_xticks(x_LF)
axes[0].set_xticklabels(feature_names_sorted, rotation=90, fontsize=10) 
axes[0].set_ylim(0, 0.3)
axes[0].set_title('LF', fontsize=22)
axes[0].tick_params(axis='y', labelsize=20)
axes[0].spines['left'].set_linewidth(1.5)
axes[0].spines['bottom'].set_linewidth(1.5)
axes[0].spines['top'].set_linewidth(1.5)
axes[0].spines['right'].set_linewidth(1.5)

# =========================
# EF
# =========================

axes[1].bar(x_EF, shap_mean_abs_EF_sorted, color=colors_sorted, width=bar_width)
axes[1].set_xticks(x_EF)
axes[1].set_xticklabels(feature_names_sorted, rotation=90, fontsize=10) 
axes[1].set_ylim(0, 0.3)
axes[1].set_title('EF', fontsize=22)
axes[1].tick_params(axis='y', labelsize=20)
axes[1].spines['left'].set_linewidth(1.5)
axes[1].spines['bottom'].set_linewidth(1.5)
axes[1].spines['top'].set_linewidth(1.5)
axes[1].spines['right'].set_linewidth(1.5)



red_patch = mpatches.Patch(color='tab:red', label='mutations')
blue_patch = mpatches.Patch(color='tab:blue', label='genes')
axes.legend(handles=[red_patch, blue_patch],  loc='upper right', fontsize=30)

plt.show()
