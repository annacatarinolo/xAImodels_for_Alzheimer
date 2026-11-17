import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, normalize

class MLP_snp_f(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(input_dim, hid_dim*4)
        self.layer_hidden = nn.Linear(hid_dim*4, hid_dim*2)
        self.layer_hidden2 = nn.Linear(hid_dim*2, hid_dim)
        self.layer_2 = nn.Linear(hid_dim, output_dim)
       
        self.dropout = nn.Dropout(p=0.2)

    def get_factors(self, x) :
        x = th.nn.functional.relu(self.layer_1(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden2(x))
        x = self.dropout(x)

    def forward(self, x):
        
        x = th.nn.functional.relu(self.layer_1(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden2(x))
        x = self.dropout(x)
        
        x = self.layer_2(x)
        
        return x

class MLP_gene_f(nn.Module):
    
    def __init__(self, input_dim, hid_dim, output_dim):
        
        super().__init__()
        
        self.layer_1 = nn.Linear(input_dim, hid_dim)
        self.layer_hidden = nn.Linear(hid_dim, hid_dim)
        #self.layer_hidden2 = nn.Linear(hid_dim, hid_dim)
        self.layer_2 = nn.Linear(hid_dim, output_dim)
       
        self.dropout = nn.Dropout(p=0.2)

    def get_factors(self, x) :
        x = th.nn.functional.relu(self.layer_1(x))
        x = self.dropout(x)
        x = th.nn.functional.relu(self.layer_hidden(x))
    
    def forward(self, x):
        
        x = th.nn.functional.relu(self.layer_1(x))
        x = self.dropout(x)
        
        x = th.nn.functional.relu(self.layer_hidden(x))
        #x = self.dropout(x)
        
        #x = th.nn.functional.relu(self.layer_hidden2(x))
        #x = self.dropout(x)
        
        x = self.layer_2(x)
        
        return x

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
            
        #self.output_model.add_module('output_last', nn.Linear(self.layers_dim_output[-1], self.output_size))
        self.last_situation=nn.Linear(self.layers_dim_output[-1], self.output_size)
        # if self.output_size==1:
        #     self.out_act = nn.Sigmoid()

    def get_factors(self, x) :
        x = self.output_model(x)
        return x
    
    def get_concatenated_features(self, x1, x2):
        x1 = self.modelA.forward(x1)
        x2 = self.modelB.forward(x2)
        x = th.cat((x1, x2), dim=1)
        return x

    def forward(self, x1, x2):
        x1 = self.modelA.forward(x1)
        x2 = self.modelB.forward(x2)
        
        #^ Peso i singoli elementi in base al loro contributo nel risultato finale
        #x1 = x1 * self.lw_snp
        #x2 = x2 * self.lw_gene

        x = th.cat((x1, x2), dim=1)
        x= self.get_factors(x)
        #print(self.output_model)
        #x = self.output_model(x)
        x=self.last_situation(x)


        #print(f"Nuovi pesi gene {self.lw_gene} & snp {self.lw_snp}")
    
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

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class WrappedModel(th.nn.Module):
    def __init__(self, dim1):
        super().__init__()
        self.model_gene = MLP_gene_f(174,75,output_dim = 1).double()
        self.model_snp = MLP_snp_f(543,75,output_dim = 1).double()
        self.model_multi = Multiomics_merged(self.model_snp, self.model_gene, outdims=150, layers_dim_output=[32]).double()
 
        self.dim1 = dim1  # numero feature di x1
        
    def forward(self, x):
        x_snp = x[:, :self.dim1]
        x_gene = x[:, self.dim1:]
        checkp = th.load('/Users/catarinolo/Documents/python/project/snp_gene_multiomics_MLR/mofax/bibonet/snp_best.pth')
        filtered_state_snp = {
            k: v for k, v in checkp.items()
            if k in self.model_snp.state_dict()
        }
        self.model_snp.load_state_dict(filtered_state_snp, strict=False)
        #self.model_snp.load_state_dict(checkp)
        checkp = th.load('/Users/catarinolo/Documents/python/project/snp_gene_multiomics_MLR/mofax/bibonet/gene_best.pth')
        filtered_state_gene = {
            k: v for k, v in checkp.items()
            if k in self.model_gene.state_dict()
        }
        self.model_gene.load_state_dict(filtered_state_gene, strict=False)
        self.model_gene.eval()
        self.model_snp.eval()
        self.model_gene.layer_2 = Identity()
        self.model_snp.layer_2 = Identity()
        checkp = th.load('/Users/catarinolo/Documents/python/project/snp_gene_multiomics_MLR/mofax/bibonet/merged_best.pth', map_location=th.device('cpu'))
        self.model_multi.load_state_dict(checkp, strict=False)
        self.model_multi.eval()
        # self.model_multi.last_situation = Identity()
        x_output = self.model_multi(x_snp, x_gene)
 
        return x_output
    
    
    
    
    
    
    
    
    
    