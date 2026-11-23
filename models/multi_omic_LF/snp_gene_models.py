# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:03:07 2025

@author: massimo
"""
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from torchsummary import summary  # SOLO PER VISUALIZZARE IL MODELLO
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score,recall_score, matthews_corrcoef, roc_auc_score, balanced_accuracy_score,  confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, normalize
import torch.optim as optim
#from torchviz import make_dot

class snp_Deep(nn.Module):
    
    '''
    Simple neural network with four layers of fully connected neurons and Relus activation functions
    Last one is a Sigmoid or Softmax based on the output_size
    -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
    '''
    def __init__(self, input_size=328, layers_dim = [328, 100], output_size=1, d_rate=0.2):
        super(snp_Deep, self).__init__()
        
        self.output_size = output_size
        self.layers_dim = layers_dim
        self.input_size = input_size
        self.d_rate = d_rate
        
        
        
            
                
        ########### Build the MLP ###########
        # self.snp_net = nn.Sequential()
        self.snp_net = nn.Sequential(nn.Dropout(0.2))
        
        # Start building the dense layer part based on layers dimension 
        for j in range(len(self.layers_dim)):
            if j == 0: # input layer
                self.snp_net.add_module('linear_{}'.format(j), nn.Linear(self.input_size,
                                                                           self.layers_dim[j]))
            else: # hidden layers
                self.snp_net.add_module('linear_{}'.format(j), nn.Linear(self.layers_dim[j-1],
                                                                           self.layers_dim[j]))
                    
            # Complete the layer                
            self.snp_net.add_module('bn_{}'.format(j), nn.BatchNorm1d(self.layers_dim[j]))
            self.snp_net.add_module('act_{}'.format(j),  nn.ReLU())
            self.snp_net.add_module('dropout_{}'.format(j), nn.Dropout(self.d_rate))

        ####### OUTPUT LAYER ########################
        self.output = nn.Linear(self.layers_dim[-1], self.output_size)
        
        # if self.output_size==1:
        #     self.out_act = nn.Sigmoid()
        
        # self.double()

    def forward(self, x):
  
        x = self.snp_net(x)
        
        x = self.output(x)
        
        
        return x.squeeze(1)

class MLP_snp_f(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(input_dim, hid_dim*4)
        self.layer_hidden = nn.Linear(hid_dim*4, hid_dim*2)
        self.layer_hidden2 = nn.Linear(hid_dim*2, hid_dim)
        self.layer_2 = nn.Linear(hid_dim, output_dim)
       
        self.dropout = nn.Dropout(p=0.2)
        # self.batchn = nn.BatchNorm1d(hid_dim*2)
    
    def forward(self, x):
        
        x = th.nn.functional.relu(self.layer_1(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden2(x))
        x = self.dropout(x)
        
        x = self.layer_2(x)
        
        return x

class MLP_snp(nn.Module):
    
    def __init__(self, input_dim_snp, hid_dim_snp, output_dim):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(input_dim_snp, hid_dim_snp)
        self.layer_hidden = nn.Linear(hid_dim_snp, hid_dim_snp)
        self.layer_hidden2 = nn.Linear(hid_dim_snp, hid_dim_snp)
        self.layer_hidden3 = nn.Linear(hid_dim_snp, hid_dim_snp)
        self.layer_hidden4 = nn.Linear(hid_dim_snp, hid_dim_snp)
        self.layer_hidden5 = nn.Linear(hid_dim_snp, hid_dim_snp)
        self.layer_2 = nn.Linear(hid_dim_snp, output_dim)
       
        self.dropout = nn.Dropout(p=0.2)
        
        # for layer in [self.layer_1, self.layer_hidden, self.layer_hidden2]:
        #     nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    
    def forward(self, x):
        
        x = th.nn.functional.relu(self.layer_1(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden2(x))
        # x = self.dropout(x)
        
        # x = th.nn.functional.relu(self.layer_hidden3(x))
        # x = self.dropout(x)
        
        # x = th.nn.functional.relu(self.layer_hidden4(x))
        # # x = self.dropout(x) 
        
        # x = th.nn.functional.relu(self.layer_hidden5(x))
        x = self.layer_2(x)
        
        return x
    
    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.layer_2(x)
    #     return x

class MLP_gene_f(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(input_dim, hid_dim)
        self.layer_hidden = nn.Linear(hid_dim, hid_dim)
        #self.layer_hidden2 = nn.Linear(hid_dim, hid_dim)
        self.layer_2 = nn.Linear(hid_dim, output_dim)
       
        self.dropout = nn.Dropout(p=0.2)
        
    
    def forward(self, x):
        
        x = th.nn.functional.relu(self.layer_1(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden(x))
        #x = self.dropout(x)
        
        #x = th.nn.functional.relu(self.layer_hidden2(x))
        #x = self.dropout(x)
        
        x = self.layer_2(x)
        
        return x
    
class MLP_gene(nn.Module):
    
    def __init__(self, input_dim_ge, hid_dim_gene, output_dim):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(input_dim_ge, hid_dim_gene)
        self.layer_hidden = nn.Linear(hid_dim_gene, hid_dim_gene)
        self.layer_hidden2 = nn.Linear(hid_dim_gene, hid_dim_gene)
        self.layer_hidden3 = nn.Linear(hid_dim_gene, hid_dim_gene)
        self.layer_hidden4 = nn.Linear(hid_dim_gene, hid_dim_gene)
        self.layer_hidden5 = nn.Linear(hid_dim_gene, hid_dim_gene)
        self.layer_2 = nn.Linear(hid_dim_gene, output_dim)
       
        self.dropout = nn.Dropout(p=0.2)
        
        # for layer in [self.layer_1, self.layer_hidden, self.layer_hidden2]:
        #     nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    
    def forward(self, x):
        
        x = th.nn.functional.relu(self.layer_1(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden2(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden3(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden4(x))
        # # x = self.dropout(x) 
        
        # x = th.nn.functional.relu(self.layer_hidden5(x))
        x = self.layer_2(x)
        
        return x
    
    # def forward(self, x):
    #     x = self.forward_features(x)
    #     x = self.layer_2(x)
    #     return x
    
# Convert data to torch tensors
class Data(Dataset):
    
    def __init__(self, X, y):
        
        if isinstance(X, np.ndarray):
            self.X = th.tensor(X, dtype=th.float32)
        elif isinstance(X, pd.DataFrame):
            self.X = th.tensor(X.values, dtype=th.float32)
        elif isinstance(X, th.Tensor):
            self.X = X.float()
        else:
            raise TypeError("Unsupported data type for X")

        if isinstance(y, np.ndarray):
            self.y = th.tensor(y, dtype=th.float32)
        elif isinstance(y, pd.Series):
            self.y = th.tensor(y.values, dtype=th.float32)
        elif isinstance(y, th.Tensor):
            self.y = y.float()
        else:
            raise TypeError("Unsupported data type for y")
        
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

class Multiomics_merged(nn.Module):
    def __init__(self, modelA, modelB, outdims, output_size=1, layers_dim_output=None,
                 d_rate=0.2):
        
        super(Multiomics_merged, self).__init__()
        
        if layers_dim_output is None:
            raise ValueError("layers_dim_output must be provided")
        
        self.modelA = modelA
        self.modelB = modelB
        self.outdims = outdims
        
        self.output_size = output_size
        self.layers_dim_output = list(layers_dim_output)
        self.d_rate = d_rate

        # Aggiungo un paio di Parametri come learning weight
        self.lw_gene = nn.Parameter(th.tensor([1.0])) 
        self.lw_snp = nn.Parameter(th.tensor([1.0]))

        ####### DEFINE what happens after concatenation ############
        self.output_model = nn.Sequential(nn.Dropout(self.d_rate))  
        #self.output_model = nn.Sequential()
        for k in range(len(self.layers_dim_output)):
            if k==0:
                self.output_model.add_module('output_{}'.format(k), nn.Linear(self.outdims, self.layers_dim_output[k]))
            else:
                self.output_model.add_module('output_{}'.format(k), nn.Linear(self.layers_dim_output[k-1], self.layers_dim_output[k]))
            
            # Complete the layer
            #self.output_model.add_module('bn_{}'.format(k), nn.BatchNorm1d(self.layers_dim_output[k]))
            self.output_model.add_module('bn_{}'.format(k), nn.LayerNorm(self.layers_dim_output[k]))
            self.output_model.add_module('act_{}'.format(k), nn.ReLU())
            self.output_model.add_module('dropout_{}'.format(k), nn.Dropout(self.d_rate))
            
        self.output_model.add_module('output_last', nn.Linear(self.layers_dim_output[-1], self.output_size))
        # if self.output_size==1:
        #     self.out_act = nn.Sigmoid()

    def forward(self, x1, x2):
        x1 = self.modelA.forward(x1)
        x2 = self.modelB.forward(x2)
        
        #^ Peso i singoli elementi in base al loro contributo nel risultato finale
        #x1 = x1 * self.lw_snp
        #x2 = x2 * self.lw_gene

        x = th.cat((x1, x2), dim=1)
        #print(self.output_model)
        x = self.output_model(x)

        #print(f"Nuovi pesi gene {self.lw_gene} & snp {self.lw_snp}")
    
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Multiomics_early(nn.Module):
    def __init__(self, indims, output_size=1, layers_dim_output=None,
                 d_rate=0.2, first_layer=512):
        
        super(Multiomics_early, self).__init__()
        
        if layers_dim_output is None:
            raise ValueError("layers_dim_output must be provided")
        
        self.indims = indims
        self.output_size = output_size
        self.layers_dim_output = list(layers_dim_output)
        self.d_rate = d_rate
        self.first_layer = first_layer

        ####### DEFINE what happens after concatenation ############
        self.output_model = nn.Sequential(nn.Linear(self.indims, self.first_layer))
        self.output_model.add_module('act_00', nn.ReLU())
        self.output_model.add_module('dropout_00', nn.Dropout(self.d_rate))
        for k in range(len(self.layers_dim_output)):
            if k==0:
                self.output_model.add_module('output_{}'.format(k), nn.Linear(self.first_layer, self.layers_dim_output[k]))
            else:
                self.output_model.add_module('output_{}'.format(k), nn.Linear(self.layers_dim_output[k-1], self.layers_dim_output[k]))
            
            # Complete the layer
            self.output_model.add_module('bn_{}'.format(k), nn.LayerNorm(self.layers_dim_output[k]))
            self.output_model.add_module('act_{}'.format(k), nn.ReLU())
            self.output_model.add_module('dropout_{}'.format(k), nn.Dropout(self.d_rate))
            
        self.last_situation=nn.Linear(self.layers_dim_output[-1], self.output_size)

    def get_factors(self, x) :
        x = self.output_model(x)
        return x
    
    def forward(self, x):    
        #print(self.output_model)
        x = self.get_factors(x)
        #print(f"Nuovi pesi gene {self.lw_gene} & snp {self.lw_snp}")
        x = self.last_situation(x)
        return x
    
