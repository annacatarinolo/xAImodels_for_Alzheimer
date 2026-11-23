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
from matplotlib.patches import Patch

#loading SNP and genes datasets
ge_df= pd.read_csv("de_genes_renamed.csv", index_col=0)
snp_df= pd.read_csv("snp_selected.csv", index_col=0)

#convert to PyTorch tensors so that they can be processed by the LF models
ge_tensor=torch.tensor(ge_df.iloc[:,:-1].values)
snp_tensor=torch.tensor(snp_df.iloc[:,:-1].values)

X_ge= ge_df.iloc[:,:-1].values
X_snp= snp_df.iloc[:,:-1].values

target= snp_df['Bl_DX']
encoder= LabelEncoder()
target_encoded= encoder.fit_transform(target)
y= target_encoded

skf= StratifiedKFold(n_splits=5, random_state=35, shuffle=True)

#defining the lists in which SHAP values for the two configurations (EF and LF) will be collected
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
    #carico il background per il modello EF
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
    #For each iteration, we save the indices of the test_set and the shap_values
    list_shap_values_LF.append(shap_values_LF)
    
    explainer_EF = shap.DeepExplainer(early_model, back_tensor[130:])
    shap_values_EF = explainer_EF.shap_values(info_test_tensor,check_additivity=False).squeeze()
    #For each iteration, we save the indices of the test_set and the shap_values
    list_shap_values_EF.append(shap_values_EF)
    
    list_test_sets.append(info_test)

# defining the EF and LF shape value arrays as a vertical stack of the lists obtained
shap_values_LF = np.vstack(list_shap_values_LF)   
shap_values_EF = np.vstack(list_shap_values_EF)

# defining the colour mapping and calculate the mean and absolute value of the SHAP values
feature_colors = ['tab:blue']*543 + ['tab:orange']*174
shap_mean_abs_LF = np.abs(shap_values_LF).mean(axis=0) 
shap_mean_abs_EF = np.abs(shap_values_EF).mean(axis=0)
feature_names= info_test.columns.tolist()

#selecting 100 features for the first plot
top_k = 100

order_LF = np.argsort(shap_mean_abs_LF)[::-1]
order_EF = np.argsort(shap_mean_abs_EF)[::-1]

top_order_LF = order_LF[:top_k]
top_order_EF = order_EF[:top_k]

shap_mean_abs_LF_sorted = shap_mean_abs_LF[top_order_LF]
shap_mean_abs_EF_sorted = shap_mean_abs_EF[top_order_EF]

feature_names_LFsorted = np.array(feature_names)[top_order_LF]
feature_names_EFsorted = np.array(feature_names)[top_order_EF]

colors_LF_sorted = np.array(feature_colors)[top_order_LF]
colors_EF_sorted = np.array(feature_colors)[top_order_EF]

x_LF = np.arange(len(shap_mean_abs_LF_sorted))
x_EF = np.arange(len(shap_mean_abs_EF_sorted))

fig, axes = plt.subplots(2, 1,figsize=(30, 20))
bar_width = 0.8 

# =========================
# LF
# =========================

axes[0].bar(x_LF, shap_mean_abs_LF_sorted, color=colors_LF_sorted, width=bar_width)
axes[0].set_xticks(x_LF)
axes[0].set_xticklabels(feature_names_LFsorted, rotation=90, fontsize=10) 
axes[0].set_ylim(0, 0.3)
axes[0].set_title('LF', fontsize=36,
                  loc='center', y=0.92)
axes[0].tick_params(axis='y', labelsize=20)
axes[0].spines['left'].set_linewidth(1.5)
axes[0].spines['bottom'].set_linewidth(1.5)
axes[0].spines['top'].set_linewidth(1.5)
axes[0].spines['right'].set_linewidth(1.5)

# =========================
# EF
# =========================

axes[1].bar(x_EF, shap_mean_abs_EF_sorted, color=colors_EF_sorted, width=bar_width)
axes[1].set_xticks(x_EF)
axes[1].set_xticklabels(feature_names_EFsorted, rotation=90, fontsize=10) 
axes[1].set_ylim(0, 0.3)
axes[1].set_title('EF', fontsize=36,
                  loc='center', y=0.92)
axes[1].tick_params(axis='y', labelsize=20)
axes[1].spines['left'].set_linewidth(1.5)
axes[1].spines['bottom'].set_linewidth(1.5)
axes[1].spines['top'].set_linewidth(1.5)
axes[1].spines['right'].set_linewidth(1.5)

orange_patch = mpatches.Patch(color='tab:blue', label='mutations')
blue_patch = mpatches.Patch(color='tab:orange', label='genes')

for ax in axes:
    ax.legend(
        handles=[orange_patch, blue_patch],
        loc='upper right',
        fontsize=20
    )
    
plt.show()

#For the next plot, the value of top_k has to be changed in 300 (line 101)

top_k = 300

order_LF = np.argsort(shap_mean_abs_LF)[::-1]
order_EF = np.argsort(shap_mean_abs_EF)[::-1]

top_order_LF = order_LF[:top_k]
top_order_EF = order_EF[:top_k]

shap_mean_abs_LF_sorted = shap_mean_abs_LF[top_order_LF]
shap_mean_abs_EF_sorted = shap_mean_abs_EF[top_order_EF]

feature_names_LFsorted = np.array(feature_names)[top_order_LF]
feature_names_EFsorted = np.array(feature_names)[top_order_EF]

colors_LF_sorted = np.array(feature_colors)[top_order_LF]
colors_EF_sorted = np.array(feature_colors)[top_order_EF]

k_values = [10, 20, 50, 100, 300]

snp_early = []
gene_early = []
snp_late = []
gene_late = []

for k in k_values:
   # EF
    snp_early.append(np.sum(colors_EF_sorted[:k] == 'tab:blue'))
    gene_early.append(np.sum(colors_EF_sorted[:k] == 'tab:orange'))

    # LF
    snp_late.append(np.sum(colors_LF_sorted[:k] == 'tab:blue'))
    gene_late.append(np.sum(colors_LF_sorted[:k] == 'tab:orange'))

df = pd.DataFrame({
    ('Early', 'SNP'): snp_early,
    ('Early', 'Genes'): gene_early,
    ('Late', 'SNP'): snp_late,
    ('Late', 'Genes'): gene_late
}, index=[10, 20, 50, 100, 300])

x = np.arange(len(df))
w = 0.35

fig, ax = plt.subplots(figsize=(8,5))

# Early
ax.bar(x - w/2, df[('Early', 'SNP')], w,
       color='tab:blue')

ax.bar(x - w/2, df[('Early', 'Genes')], w,
       bottom=df[('Early', 'SNP')],
       color='tab:orange')

# Late
ax.bar(x + w/2, df[('Late', 'SNP')], w,
       color='tab:blue',
       hatch='//',
       edgecolor='black')

ax.bar(x + w/2, df[('Late', 'Genes')], w,
       bottom=df[('Late', 'SNP')],
       color='tab:orange',
       hatch='//',
       edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(df.index)
ax.set_xlabel('Range of features', fontsize= 12)
ax.set_ylabel('Number of features', fontsize=12)

legend1 = ax.legend(
    handles=[
        Patch(facecolor='tab:blue', label='SNP'),
        Patch(facecolor='tab:orange', label='Genes'),
        Patch(facecolor='white', edgecolor='black', label='Early'),
        Patch(facecolor='white', edgecolor='black', hatch='//', label='Late')
    ],
    loc='upper left',
    fontsize=12
)

ax.add_artist(legend1)

plt.tight_layout()
plt.show()



