# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:38:36 2026

@author: catarinolo
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, matthews_corrcoef
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.optim as optim
import os
import random
from statistics import mean 
from micro_meta_utils_train import model_train_multiomics_new 
from snp_gene_models import Multiomics_merged, Identity, MLP_snp_f, MLP_gene_f

def set_seed(seed):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
#defining train function for training the model. With "mode" one can choose between 
#fine-tuning or frozen configuration
def train(fold,layers_dim,d_rate, lr, n_epochs, batch_size, X1_train, X2_train,
           y_train,X1_val,X2_val,y_val,best_arch_1,
           best_arch_2,
           mode = 'fine',
           seed = 42,
           device = 'cuda',
           outdir = 'outdir_best',
           model_name = 'merged', 
           task = 'binary',
           early_stop = False,
           verbose = False):
    
    device ='cuda:1' if torch.cuda.is_available() else 'cpu'

    output_size=1
    loss_fn = nn.BCEWithLogitsLoss()
  
    snp_dims, gene_dims= [], []
    
    snp_path = os.path.join(outdir, best_arch_1)
    gene_path = os.path.join(outdir, best_arch_2)
    
    snp_arch = torch.load(snp_path, map_location=device)
    gene_arch = torch.load(gene_path, map_location=device)
    
    for param in snp_arch:
        # print(param)
        # if all(x in param for x in match):
        if param.endswith('.weight'):
            # print(param)
            t = snp_arch[param]
            if t is not None and t.numel() > 0:
                snp_dims.append(t.size(0))
                # snp_dims.append(snp_arch[param].size()[0])
            
    for param in gene_arch:
        
        if param.endswith('.weight'):
            t = gene_arch[param]
            if t is not None and t.numel() > 0:
                gene_dims.append(t.size(0))
                
    # Controlla che snp_dims e gene_dims non siano vuoti
    if len(snp_dims) == 0 or len(gene_dims) == 0:
        raise ValueError("Error: empty snp_dims or gene_dims! Check saved models")
    # print(snp_dims)
    
    # Instantiate the models
    model1 = MLP_snp_f(input_dim_snp, snp_dims[-2],
                            output_dim = output_size).double().to(device) #load the architecture
    
    model2 = MLP_gene_f(input_dim_ge, gene_dims[-2],
                            output_dim=output_size).double().to(device)
    
    print(" - - - - - - - - - - - ")
    print(f"CHECK SNP START DIMENSIONS: {snp_dims}")
    print(f"CHECK GENE START DIMENSIONS: {gene_dims}")
    print(" - - - - - - - - - - - ")
    
    if mode in ['frozen', 'fine']:
        
        model1.load_state_dict(snp_arch)  #carica i pesi
        model2.load_state_dict(gene_arch)
        model1.eval()
        model2.eval()

    outdimA = model1.layer_2.in_features
    outdimB = model2.layer_2.in_features
    
    # here final outputs are concatenated
    outdims = outdimA + outdimB
    
    # Eliminate output layers
    model1.layer_2 = Identity()
    model2.layer_2 = Identity()
    
    ########### SET UP VARIABLES for reproducibility using the set_seed function
    set_seed(seed)
    ##############################################

    model = Multiomics_merged(model1, 
                            model2,
                            outdims,
                            output_size=1, 
                            layers_dim_output=layers_dim, 
                            d_rate=d_rate).double()

    model = model.to(device)
    optimizer = optim.Adam( model.parameters(),lr=lr)

    
    # Debugging to check whether the model is ready
    # print(f"[DEBUG] Model ready on device: {next(model.parameters()).device}")
    
    val_scores, train_scores,  thloss, vhloss, y_pred = model_train_multiomics_new(fold, model, 'merged',
                                    X1_train, X2_train, y_train, X1_val, X2_val, y_val, 
                                    loss_fn, optimizer, writer=None, n_epochs=n_epochs, 
                                    batch_size=batch_size, mode = mode,
                                    device = device, outdir = outdir, task= task, verbose = verbose)
    
    return model, val_scores, train_scores,  thloss, vhloss, y_pred
    # model.load_state_dict(snp_arch)

#loading genes and snp datasets
df_genes = pd.read_csv("datasets\multi_omic\de_genes_renamed.csv",index_col=0)
df_snp = pd.read_csv("datasets\multi-omic\snp_selected.csv",index_col=0)

label = df_genes.loc[:,'BL_Diagnosis']

df_genes = df_genes.iloc[:,:-1]
df_snp = df_snp.iloc[:,:-1]

scaler = StandardScaler()

le = LabelEncoder().fit(label)
y = le.transform(label)

hidden_dim = 32

n_epochs, n_folds, lr, d_rate = 1000, 5, 0.001, 0.1
batch_size, seed, my_random_state = 42, 123, 35 #default 42 & 42 & 35
opt ='adamax' #adamax

layers_dim = [hidden_dim]

acc, prec, rec, mcc = [], [], [], []
bal_acc, auc, conf = [], [], []

skf = StratifiedKFold(n_splits=n_folds,random_state = my_random_state,shuffle=True)

for i,(train_idx,test_idx) in enumerate(skf.split(df_genes,y)):
        
        Xge_train = df_genes.iloc[train_idx,:]
        Xge_test = df_genes.iloc[test_idx,:]
        scaler = scaler.fit(Xge_train)
        Xge_train = scaler.transform(Xge_train)
        Xge_test = scaler.transform(Xge_test)
    
        Xsnp_train = df_snp.iloc[train_idx,:]
        Xsnp_test = df_snp.iloc[test_idx,:]
        scaler = scaler.fit(Xsnp_train)
        Xsnp_train = scaler.transform(Xsnp_train)
        Xsnp_test = scaler.transform(Xsnp_test)
        
        y_train = y[train_idx]
        y_test = y[test_idx]
    
        input_dim_ge = Xge_train.shape[1]
        input_dim_snp = Xsnp_train.shape[1]
        
        genepath = "gene_k"+str(i+1)+"_LAST.pth"
        snppath = "snp_k"+str(i+1)+"_LAST.pth"
        
        model, val_scores, train_scores,  thloss, vhloss, y_pred = train(i, layers_dim, d_rate, lr, n_epochs,
                                                                         batch_size, Xsnp_train, Xge_train, y_train,
                                                                         Xsnp_test, Xge_test, y_test, snppath, genepath)
        acc.append(accuracy_score(y_test, y_pred))
        prec.append(precision_score(y_test, y_pred))
        rec.append(recall_score(y_test, y_pred))
        mcc.append(matthews_corrcoef(y_test, y_pred))
    
        print(f"RIASSUNTO FOLD NUMERO {str(i+1)}\n")
        print(f"Test Accuracy fold :  {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print(f"Test Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
        print(f"Test Recall:    {recall_score(y_test, y_pred) * 100:.2f}%")
        print(f"MCC: {matthews_corrcoef(y_test, y_pred) * 100:.2f}%")
    
print("\nRIASSUNTO MEDIA FOLD")
print(f"Test Accuracy:  {mean(acc) * 100:.2f}%")
print(f"Test Precision: {mean(prec) * 100:.2f}%")
print(f"Test Recall:    {mean(rec) * 100:.2f}%")
print(f"MCC: {mean(mcc) * 100:.2f}%")