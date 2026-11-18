# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 10:17:58 2026

@author: catarinolo
"""

import muon as mu
import numpy as np
import pandas as pd
import mudata as md
import mofax as mofa
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier

#Loading of the training datasets for gene expression values and mutations and y_train
de_genes_train = pd.read_csv("datasets/multi_omic/de_genes_norm_train.csv", index_col=0)
snp_train= pd.read_csv("dataset/multi_omic/snp_select_norm_train.csv", index_col=0)

y_train= pd.read_csv("datasets/multi_omic/y_train.csv", index_col=0)

# Creating the AnnData files for both genes and SNPs,
# which are required to generate the MuData requested by MOFA

adata_ge = AnnData(X=de_genes_train)
adata_snp = AnnData(X=snp_train)

# Defining the number of factors for the MOFA model
n_factors= 50

# Creating the training MuData merging the two AnnData
mdata = md.MuData({'genes': adata_ge, 'mut': adata_snp})

# Training the MOFA model and saving it as an hdf5 file
mu.tl.mofa(
    mdata,
    n_factors=n_factors,
    convergence_mode='medium',
    outfile=f"mofax/model_final_{n_factors}.hdf5"
)

# Loading the model previously trained and saved
model = mofa.mofa_model(f"models/multi_omic_IF/model_final_{n_factors}.hdf5")

# Extracting the factor matrix created by the model
Z_train = model.get_factors(df=True)

# FIRST PLOT: barplot of the coefficient of determination R^2, giving information about the 
#proportion of variance explained by each factor.

#The following lines of code are needed to:
    
#extract the R² values from the MOFA model in the form of a DataFrame
r2= model.get_r2()

#set the factors as indices
r2= r2.set_index('Factor')

#Remove the group column, which in our case is just one and is not needed for the plot
r2= r2.drop('Group', axis=1)

#reset the index and retain only the highest value for each factor based on the corresponding view
r2 = r2.reset_index()
r2 = r2.loc[r2.groupby('Factor')['R2'].idxmax()]

#organise the indices and set the factors
r2 = r2.sort_index()
r2= r2.set_index('Factor')

#Stop the plot at factor 30, as the R² value is already negligible from the 29th onwards
r2 = r2.iloc[:30]

#scale the values on a logarithmic scale
r2['R2_log'] = np.log1p(r2['R2'])

#Colour the factors according to the views (SNPs and genes) and add the legend
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

# Defining the classifier
clf= RandomForestClassifier(random_state=42, n_estimators=200, max_depth=8)
clf_name="RandomForest"

# training the classifier with the factor matrix created by the model with the training patients
clf.fit(Z_train, y_train)