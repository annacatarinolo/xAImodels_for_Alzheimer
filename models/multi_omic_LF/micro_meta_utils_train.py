import torch.nn as nn
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, make_scorer
from sklearn.metrics import roc_curve,auc


import numpy as np
seed = 42
torch.manual_seed(seed)


class Dataset_for_MLP(Dataset):
    def __init__(self, x1, label):
        self.y = label
        self.x1 = x1
        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        y = self.y[idx]
        
        return x1, y

def model_train(i, model, X_train, y_train, X_val, y_val,
                 loss_fn, optimizer, writer=None,
                 n_epochs = 200, batch_size=8, device = 'cpu', verbose =False):

    loader_train = DataLoader(TensorDataset(X_train, y_train), 
                              shuffle=True, 
                              batch_size=batch_size, 
                              pin_memory=True, 
                              drop_last=True, num_workers=64)
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    history_train_loss = []
    history_val_loss = []

    miniloss = 10000
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X_batch = X_batch.to(device).double()
            y_batch = y_batch.to(device).double()
            y_pred = model(X_batch).to(device)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.float())
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        # print progress
        
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val = X_val.to(device)
        # y_val = y_val.float().to(device)
        y_val = y_val.to(device)
        with torch.no_grad():
            # print('OOOOOOOOOO')
            y_pred = model(X_val).to(device)
            # print(y_pred, y_val)
            vloss = loss_fn(y_pred, y_val)
            history_val_loss.append(vloss.to('cpu'))
        
        # writer.add_scalar("Loss/train", np.array(train_loss/nbatches), epoch)
        # writer.add_scalar("Loss/val", np.array(vloss), epoch)

        # writer.add_scalars('run_14h', {'loss_train':np.array(train_loss/nbatches),
        #                             'loss_val':np.array(vloss)}, epoch)

        # acc = (y_pred.round() == y_val).float().mean()
        acc = (y_pred.round() == y_val).double().mean()
        acc = float(acc)
        # acc1 = accuracy_score(y_val.detach().numpy(), y_pred.round().detach().numpy())
        mcc = matthews_corrcoef(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy())  
        f1 = f1_score(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy()) 
        roc_auc = roc_auc_score(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy()) 
        if (epoch % 10 == 0) and verbose:
            
            print('====> Epoch: {} TESTING Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} -- Average loss: {:.4f}'.format(
                                        epoch, acc, mcc, f1, roc_auc, vloss))
  
        
        
        if vloss < miniloss:
            if verbose:
                print ('+++ Saving best model because val_loss: {:.4f} < miniloss: {:.4f} +++'.format(
                                                                    vloss,miniloss))
                print ('+++ This model Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} +++'.format(acc,
                                                                                                                mcc,
                                                                                                                  f1,
                                                                                                                  roc_auc))
            miniloss = vloss
            PATH_BEST = "./BEST_model_metaboDeep_k{}.pth".format(i) # i refers to the kfold
            torch.save(model.state_dict(), PATH_BEST)
            
    PATH_LAST = "./LAST_k{}.pth".format(i) # i refers to the kfold
    torch.save(model.state_dict(), PATH_LAST)
    
    # Return accuracy of best model on val set
    checkpoint = torch.load('./LAST_k{}.pth'.format(i))
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).to(device)
        # vloss = loss_fn(y_pred, y_val.float())
        vloss = loss_fn(y_pred, y_val)

    # acc = (y_pred.round() == y_val).float().mean()
    acc = (y_pred.round() == y_val).double().mean()

    acc = float(acc)
    # acc1 = accuracy_score(y_val.detach().numpy(), y_pred.round().detach().numpy())
    mcc = matthews_corrcoef(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy())
    f1 = f1_score(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy()) 
    roc_auc = roc_auc_score(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy()) 

    return acc, mcc,f1, roc_auc, vloss.to('cpu'), history_train_loss, history_val_loss
  
def model_train_new(fold, model, model_name, X_train, y_train, X_val, y_val,
                 loss_fn, optimizer, writer=None,
                 n_epochs = 200, batch_size=16, device = 'cpu', outdir = './models/', verbose =False):

#    loader_train = DataLoader(TensorDataset(X_train, y_train), 
#                              shuffle=True, 
#                              batch_size=batch_size, 
#                              pin_memory=True, 
#                              drop_last=True)
    device ='cuda' if torch.cuda.is_available() else 'cpu'

    dataset_train = Dataset_for_MLP(X_train,y_train)
    dataset_val = Dataset_for_MLP(X_val,y_val)

    # dataset_train = Dataset_for_CNN(X_train,y_train)
    # dataset_val = Dataset_for_CNN(X_val,y_val)
    
    # print(dataset_train[0][0])
    
    
    loss_fn.to(device)
    # Set loaders for both train and validation data.
    #train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=0, pin_memory=True,drop_last =True) 
    #val_loader = DataLoader(dataset_val,batch_size=len(y_val), shuffle=False, num_workers=0, pin_memory=True,drop_last =True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True) 
    val_loader = DataLoader(dataset_val,batch_size=len(y_val), shuffle=False)
    all_train_loader = DataLoader(dataset_train,batch_size=len(y_val), shuffle=False)

    history_train_loss = []
    history_val_loss = []

    miniloss = 10000
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X_batch = X_batch.to(device).double()
            y_batch = y_batch.to(device).double()
            y_pred = model(X_batch).to(device)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.float())
            loss = loss_fn(y_pred.squeeze(), y_batch.float())
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        # print progress
        
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val, y_val = next(iter(val_loader))
        X_val = X_val.to(device)
        # y_val = y_val.float().to(device)
        y_val = y_val.to(device)
        with torch.no_grad():
            # print('OOOOOOOOOO')
            y_pred = model(X_val).to(device)
            # print(y_pred, y_val)
            vloss = loss_fn(y_pred.squeeze(), y_val.float())
            history_val_loss.append(vloss.to('cpu'))
        
        y_true = y_val.detach().to('cpu').numpy()
        y_pred = y_pred.detach().to('cpu').sigmoid().numpy().round()
        
        # print(y_true.shape,y_pred.shape)
        acc = accuracy_score(y_true,y_pred)
        acc = float(acc)
        # acc1 = accuracy_score(y_val.detach().numpy(), y_pred.round().detach().numpy())
        mcc = matthews_corrcoef(y_true, y_pred)  
        f1 = f1_score(y_true,y_pred) 
        roc_auc = roc_auc_score(y_true, y_pred) 
        if (epoch % 10 == 0) and verbose:
            
            print('====> Epoch: {} TESTING Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} -- Average loss: {:.4f}'.format(
                                        epoch, acc, mcc, f1, roc_auc, vloss))
  
        
        
        if vloss < miniloss:
            if verbose:
                print ('+++ Saving best model because val_loss: {:.4f} < miniloss: {:.4f} +++'.format(
                                                                    vloss,miniloss))
                print ('+++ This model Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} +++'.format(acc,
                                                                                                                mcc,
                                                                                                                  f1,
                                                                                                                  roc_auc))
            miniloss = vloss
            PATH_BEST = "{}/{}_k{}_BEST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
            torch.save(model.state_dict(), PATH_BEST)
            
    PATH_LAST = "{}/{}_k{}_LAST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
    torch.save(model.state_dict(), PATH_LAST)
    
    # Return accuracy of best model on val set
    # checkpoint = torch.load(PATH_BEST)
    checkpoint = torch.load(PATH_LAST)
    model.load_state_dict(checkpoint)

    
    X_train, y_train = next(iter(all_train_loader))
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).to(device)
        vloss = loss_fn(y_pred.squeeze(), y_val.float())
        # y_pred_train = model(torch.from_numpy(X_train).to(device))#.to(device)
        y_pred_train = model(X_train.to(device).to(device))#.to(device)
        # y_pred_train = model(dataset_train[0][0].to(device)).to(device)#.to(device)
        # loss_train = loss_fn(y_pred_train.squeeze(), torch.from_numpy(y_train).float().to(device)).to('cpu')
        loss_train = loss_fn(y_pred_train.squeeze(), y_train.float().to(device)).to('cpu')

    # print('predicting val_set')
     ########### VALIDATION SET   
    y_true = y_val.detach().to('cpu').numpy()
    y_pred_detach = y_pred.detach().to('cpu')
    y_pred_rnd = y_pred_detach.sigmoid().numpy().round()
    y_pred = y_pred_detach.numpy()
    
    
    y_true_val = y_true.copy()
    y_pred_val = y_pred_rnd.copy()
    
    acc = accuracy_score(y_true_val,y_pred_val)
    acc = float(acc)
    mcc = matthews_corrcoef(y_true,y_pred_val)  
    f1 = f1_score(y_true,y_pred_val) 
    roc_auc = roc_auc_score(y_true,y_pred_detach.sigmoid().numpy()) 
    
    val_scores = {'accuracy':acc, 'mcc': mcc, 'f1':f1, 'roc_auc':roc_auc, 'loss': vloss}
    
    # print('predicting train_set')
    ########### TRAIN SET 
    y_true = y_train.numpy().copy()
    y_pred = y_pred_train.detach().to('cpu').sigmoid().numpy().round()
 
    acc_train = (y_pred == y_true).mean()
    mcc_train = matthews_corrcoef(y_true,y_pred)  
    f1_train = f1_score(y_true,y_pred) 
    roc_auc_train = roc_auc_score(y_true,y_pred) 
    
    train_scores = {'accuracy':acc_train, 'mcc': mcc_train, 'f1':f1_train, 'roc_auc':roc_auc_train, 'loss': loss_train}
    # print('done')
    
    #return val_scores, train_scores, history_train_loss, history_val_loss, y_true_val, y_pred_val
    return val_scores, train_scores, history_train_loss, history_val_loss, y_pred_val
  
def model_train_slim(fold, model, model_name, X_train, y_train, X_val, y_val,
                 loss_fn, optimizer, writer=None,
                 n_epochs= 500, batch_size=64, device = 'cpu', outdir = './models/', verbose =False):

#    loader_train = DataLoader(TensorDataset(X_train, y_train), 
#                              shuffle=True, 
#                              batch_size=batch_size, 
#                              pin_memory=True, 
#                              drop_last=True)
    device ='cuda:1' if torch.cuda.is_available() else 'cpu'

    dataset_train = Dataset_for_MLP(X_train,y_train)
    dataset_val = Dataset_for_MLP(X_val,y_val)

    # dataset_train = Dataset_for_CNN(X_train,y_train)
    # dataset_val = Dataset_for_CNN(X_val,y_val)
    
    # print(dataset_train[0][0])
    
    
    loss_fn.to(device)
    # Set loaders for both train and validation data.
    #train_loader = DataLoader(dataset_train, batch_size=32, shuffle=False, num_workers=0, pin_memory=True,drop_last =True) 
    #val_loader = DataLoader(dataset_val,batch_size=len(y_val), shuffle=False, num_workers=0, pin_memory=True,drop_last =True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True) 
    val_loader = DataLoader(dataset_val,batch_size=len(y_val), shuffle=False)
    #all_train_loader = DataLoader(dataset_train,batch_size=len(y_val), shuffle=False)

    history_train_loss = []
    history_val_loss = []

    miniloss = 10000
    model.train()
    for epoch in range(n_epochs):
        
        train_loss = 0.0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()  
    #         print(X_batch.size())
    #         break
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch).to(device)
            #print("Y PRED LEN")
            #print(y_pred.size())
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
    #         break
            # loss = loss_fn(y_pred, y_batch.float())
            loss = loss_fn(y_pred.squeeze(), y_batch.float())
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
    
    nbatches = batch_idx+1

    
    PATH_LAST = "{}/{}_k{}_LAST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
    
    torch.save(model.state_dict(), PATH_LAST)

    # checkpoint = torch.load(PATH_LAST)
    # model.load_state_dict(checkpoint)

    # torch.save(model.state_dict(),PATH_LAST)
    # model = torch.load(PATH_LAST)
        
    X_val, y_val = next(iter(val_loader))
    X_val = X_val.to(device)
    # y_val = y_val.float().to(device)
    y_val = y_val.to(device)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).to(device)
        vloss = loss_fn(y_pred.squeeze(), y_val.float())
    # print('predicting val_set')
     ########### VALIDATION SET   
    y_true = y_val.detach().to('cpu').numpy()
    y_pred_detach = y_pred.detach().to('cpu')
    y_pred_rnd = y_pred_detach.sigmoid().numpy().round()
    y_pred = y_pred_detach.numpy()
    
    y_true_val = y_true.copy()
    y_pred_val = y_pred_rnd.copy()
    
    acc = accuracy_score(y_true_val,y_pred_val)
    acc = float(acc)
    mcc = matthews_corrcoef(y_true,y_pred_val)  
    f1 = f1_score(y_true,y_pred_val) 
    roc_auc = roc_auc_score(y_true,y_pred_detach.sigmoid().numpy()) 
    
    val_scores = {'accuracy':acc, 'mcc': mcc, 'f1':f1, 'roc_auc':roc_auc, 'loss': vloss}
    
    # print('done')
    
    #return val_scores, train_scores, history_train_loss, history_val_loss, y_true_val, y_pred_val
    return val_scores, None, history_train_loss, history_val_loss, y_pred_val, y_pred_detach.sigmoid().numpy()

def model_train_production(model, model_name, X_train, y_train,  loss_fn, optimizer, writer=None,
                 n_epochs = 200, batch_size=8, device = 'cpu', verbose =False):

    loader_train = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size, drop_last=True)


    history_train_loss = []
    # history_val_loss = []

    # miniloss = 10000
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X_batch).to(device)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.long())
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        # print progress
        
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
    
    print('Done. Saving last model as PROD_{}.pth'.format(model_name))     
    PATH_LAST = "./PROD_{}.pth".format(model_name)
    torch.save(model.state_dict(), PATH_LAST)
    
   
    return  history_train_loss

def model_train_multiclass(i, model, model_name, X_train, y_train, X_val, y_val,
                 loss_fn, optimizer, writer=None,
                 n_epochs = 200, batch_size=8, device = 'cpu', 
                 verbose =False):

    loader_train = DataLoader(TensorDataset(X_train, y_train), 
                              shuffle=True, 
                              batch_size=batch_size,
                              drop_last=True,
                              pin_memory = True)
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    history_train_loss = []
    history_val_loss = []

    miniloss = 10000
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X_batch = X_batch.to(device).double()
            y_batch = y_batch.to(device).double()
            y_pred = model(X_batch).to(device)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.long())
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        # print progress
        #######
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val = X_val.to(device)
        # y_val = y_val.float().to(device)
        y_val = y_val.to(device)
        with torch.no_grad():
            # print('OOOOOOOOOO')
            y_pred = model(X_val).to(device)
            # print(y_pred, y_val)
            # vloss = loss_fn(y_pred, y_val.long())
            vloss = loss_fn(y_pred, y_val)
            history_val_loss.append(vloss.to('cpu'))
        
        # writer.add_scalar("Loss/train", np.array(train_loss/nbatches), epoch)
        # writer.add_scalar("Loss/val", np.array(vloss), epoch)

        # writer.add_scalars('run_14h', {'loss_train':np.array(train_loss/nbatches),
        #                             'loss_val':np.array(vloss)}, epoch)

        y_true = y_val.detach().to('cpu').numpy()
        y_pred = nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.detach().to('cpu').numpy()

        # print(y_pred, y_true)
        roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')

        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)

        acc = (y_pred == y_true).mean()
        acc = float(acc)
        # acc1 = accuracy_score(y_val.detach().numpy(), y_pred.round().detach().numpy())
        mcc = matthews_corrcoef(y_true,y_pred)  
        # f1 = f1_score(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy()) 
        f1 = f1_score(y_true,y_pred, average='macro') 
        if (epoch % 10 == 0) and verbose:
            
            print('====> Epoch: {} TESTING Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} -- Average loss: {:.4f}'.format(
                                        epoch, acc, mcc, f1, roc_auc, vloss))
  
        
        
        if vloss < miniloss:
            if verbose:
                print ('+++ Saving best model because val_loss: {:.4f} < miniloss: {:.4f} +++'.format(
                                                                    vloss,miniloss))
                print ('+++ This model Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} +++'.format(acc,
                                                                                                                mcc,
                                                                                                                  f1,
                                                                                                                  roc_auc))
            miniloss = vloss
            PATH_BEST = "./BEST_{}_k{}.pth".format(model_name, i) # i refers to the kfold
            torch.save(model.state_dict(), PATH_BEST)
            
    PATH_LAST = "./LAST_{}_k{}.pth".format(model_name, i) # i refers to the kfold
    torch.save(model.state_dict(), PATH_LAST)
    
    # Return accuracy of best model on val set
    # checkpoint = torch.load('./BEST_{}_k{}.pth'.format(model_name, i))
    checkpoint = torch.load('./LAST_{}_k{}.pth'.format(model_name, i))
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).to(device)
        
        # vloss = loss_fn(y_pred, y_val.long())
        vloss = loss_fn(y_pred, y_val)

        
    y_true = y_val.detach().to('cpu').numpy()
    y_pred = nn.functional.softmax(y_pred, dim=1)
    y_pred = y_pred.detach().to('cpu').numpy()
    
    roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')
    
    # print(y_true.shape, y_pred.shape)
    y_true = y_true.argmax(axis=1)
    y_pred = y_pred.argmax(axis=1)
    acc = (y_pred == y_true).mean()
    acc = float(acc)
    # acc1 = accuracy_score(y_val.detach().numpy(), y_pred.round().detach().numpy())
    mcc = matthews_corrcoef(y_true,y_pred)  
    # f1 = f1_score(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy()) 
    f1 = f1_score(y_true,y_pred, average='macro') 
    
    return acc, mcc,f1, roc_auc, vloss.to('cpu'), history_train_loss, history_val_loss

def model_train_multiclass_new(i, model, model_name, X_train, y_train, X_val, y_val,
                 loss_fn, optimizer, writer=None,
                 n_epochs = 200, batch_size=8, device = 'cpu', outdir = './',
                 verbose =False):

    loader_train = DataLoader(TensorDataset(X_train, y_train), 
                              shuffle=True, 
                              batch_size=batch_size,
                              drop_last=True,
                              pin_memory = True)
    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    history_train_loss = []
    history_val_loss = []

    miniloss = 10000
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        # train_f1 = 0.0
        # train_mcc = 0.0
        # train_roc_auc = 0.0
        
        
        
        for batch_idx, (X_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X_batch = X_batch.to(device).double()
            y_batch = y_batch.to(device).double()
            y_pred = model(X_batch).to(device)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.long())
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
            
            
            # print progress
            ################################################################
            ####### Calculate per batch training metrics ###################
            y_true = y_batch.detach().to('cpu').numpy()
            y_pred = nn.functional.softmax(y_pred, dim=1)
            y_pred = y_pred.detach().to('cpu').numpy()

            # print(y_pred, y_true)
            # train_roc_auc += roc_auc_score(y_true,y_pred, multi_class='ovr')
            # Get the labels
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)

            train_acc += float((y_pred == y_true).mean())
            # train_mcc += matthews_corrcoef(y_true,y_pred)  
            # train_f1 += f1_score(y_true,y_pred, average='macro') 
            
        nbatches = batch_idx+1

        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- Accuracy {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, train_acc/nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val = X_val.to(device)
        # y_val = y_val.float().to(device)
        y_val = y_val.to(device)
        with torch.no_grad():
            # print('OOOOOOOOOO')
            y_pred = model(X_val).to(device)
            # print(y_pred, y_val)
            # vloss = loss_fn(y_pred, y_val.long())
            vloss = loss_fn(y_pred, y_val)
            history_val_loss.append(vloss.to('cpu'))
        
        # writer.add_scalar("Loss/train", np.array(train_loss/nbatches), epoch)
        # writer.add_scalar("Loss/val", np.array(vloss), epoch)

        # writer.add_scalars('run_14h', {'loss_train':np.array(train_loss/nbatches),
        #                             'loss_val':np.array(vloss)}, epoch)

        y_true = y_val.detach().to('cpu').numpy()
        y_pred = nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.detach().to('cpu').numpy()

        # print(y_pred, y_true)
        roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')

        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)

        acc = (y_pred == y_true).mean()
        acc = float(acc)
        # acc1 = accuracy_score(y_val.detach().numpy(), y_pred.round().detach().numpy())
        mcc = matthews_corrcoef(y_true,y_pred)  
        # f1 = f1_score(y_val.detach().to('cpu').numpy(), y_pred.round().detach().to('cpu').numpy()) 
        f1 = f1_score(y_true,y_pred, average='macro') 
        if (epoch % 10 == 0) and verbose:
            
            print('====> Epoch: {} TESTING Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} -- Average loss: {:.4f}'.format(
                                        epoch, acc, mcc, f1, roc_auc, vloss))
  
        
        
        if vloss < miniloss:
            if verbose:
                print ('+++ Saving best model because val_loss: {:.4f} < miniloss: {:.4f} +++'.format(
                                                                    vloss,miniloss))
                print ('+++ This model Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} +++'.format(acc,
                                                                                                                mcc,
                                                                                                                  f1,
                                                                                                                  roc_auc))
            miniloss = vloss
            PATH_BEST = "{}/{}_k{}_BEST.pth".format(outdir, model_name, i) # i refers to the kfold
            torch.save(model.state_dict(), PATH_BEST)
            
    PATH_LAST = "{}/{}_k{}_LAST.pth".format(outdir, model_name, i) # i refers to the kfold
    torch.save(model.state_dict(), PATH_LAST)
    
    # Return accuracy of best model on val set
    checkpoint = torch.load(PATH_LAST)
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        # y_pred = model(X_val).to(device)
        y_pred = model(X_val)
        vloss = loss_fn(y_pred.squeeze(), y_val.float())
        y_pred_train = model(torch.from_numpy(X_train).to(device))
        loss_train = loss_fn(y_pred_train.squeeze(), torch.from_numpy(y_train).float().to(device)).to('cpu')

    ########### VALIDATION SET   
    y_true = y_val.detach().to('cpu').numpy()
    y_pred_detach = y_pred.detach().to('cpu')    
    y_pred_rnd = y_pred_detach.sigmoid().numpy().round()
    y_pred = y_pred_detach.numpy()

    y_true_val = y_true.copy()
    y_pred_val = y_pred_rnd.copy()
    
    accuracy_score(y_true_val,y_pred_val)
    acc = float(acc)
    mcc = matthews_corrcoef(y_true,y_pred_rnd)  
    f1 = f1_score(y_true,y_pred_rnd) 
    roc_auc = roc_auc_score(y_true,y_pred) 
    
    val_scores = {'accuracy':acc, 'mcc': mcc, 'f1':f1, 'roc_auc':roc_auc, 'loss': vloss}

    # roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')
    # y_true = y_true.argmax(axis=1)
    # y_pred = y_pred.argmax(axis=1)
    # y_true_val = y_true.copy()
    # y_pred_val = y_pred.copy()
    # acc = (y_pred == y_true).mean()
    # acc = float(acc)
    # mcc = matthews_corrcoef(y_true,y_pred)  

    # f1 = f1_score(y_true,y_pred, average='macro') 
    
    # val_scores = {'accuracy':acc, 'mcc': mcc, 'f1':f1, 'roc_auc':roc_auc, 'loss': vloss}
    
    ########### TRAIN SET 
    y_true = y_train.copy()
    y_pred = y_pred_train.detach().to('cpu').sigmoid().numpy().round()
 
    acc_train = (y_pred == y_true).mean()
    mcc_train = matthews_corrcoef(y_true,y_pred)  
    f1_train = f1_score(y_true,y_pred) 
    roc_auc_train = roc_auc_score(y_true,y_pred) 
    
    train_scores = {'accuracy':acc_train, 'mcc': mcc_train, 'f1':f1_train, 'roc_auc':roc_auc_train, 'loss': loss_train}
    # print('done')
    
    #return val_scores, train_scores, history_train_loss, history_val_loss, y_true_val, y_pred_val
    return val_scores, train_scores, history_train_loss, history_val_loss, y_pred_val

class MyDataset(Dataset):
    def __init__(self, datasetA, datasetB, labels):
        self.datasetA = datasetA
        self.datasetB = datasetB
        self.labels = labels
        
    def __getitem__(self, index):
        xA = self.datasetA[index]
        xB = self.datasetB[index]
        ylabels = self.labels[index]
        return xA, xB, ylabels
    
    def __len__(self):
        return len(self.datasetA)

def model_train_multiomics_new(fold, model, model_name,
                           X1_train, X2_train, y_train, 
                           X1_val, X2_val, y_val,
                           loss_fn, optimizer, writer=None,
                           n_epochs = 1000, batch_size=64,
                           mode = 'fine',
                           device = 'cpu', 
                           outdir = './models/', 
                           task = 'binary',
                           verbose =False):
    '''
    X1: snp
    X2: genes
    '''
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn.to(device)

    dataset = MyDataset(X1_train, X2_train, y_train)
    # dataset = MyDatasetCNN(X1_train, X2_train, y_train)
    dataset_val = MyDataset(X1_val, X2_val, y_val)
    loader_train = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=False )
    val_loader = DataLoader(dataset_val,batch_size=len(y_val), shuffle=False)

    history_train_loss = []
    history_val_loss = []

    X1_val, X2_val, y_val = next(iter(val_loader))
    X1_val = X1_val.to(device)
    X2_val = X2_val.to(device)
    # X1_val = torch.from_numpy(X1_val).to(device)
    # X2_val = torch.from_numpy(X2_val).to(device)
    y_val = y_val.float().to(device)
    # y_val = torch.from_numpy(y_val).to(device)

    miniloss = 10000
    
    # print('')
    for epoch in range(n_epochs):
        
        model.train()
        if mode == 'frozen':
                ### Freeze modelA and modelB
            for param in model.modelA.parameters():
                param.requires_grad = False
                param.grad = None
            for param in model.modelB.parameters():
                param.requires_grad = False
                param.grad = None

        train_loss = 0.0

        for batch_idx, (X1_batch, X2_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X1_batch = X1_batch.to(device).double()
            X2_batch = X2_batch.to(device).double()
            y_batch = y_batch.to(device).double()
            y_pred = model(X1_batch, X2_batch).to(device)
            # y_pred = model(X1_batch, X2_batch)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            loss = loss_fn(y_pred.squeeze(), y_batch.float())
            train_loss += loss.item()
#         break
            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
        # evaluate accuracy at end of each epoch
        # model.eval()
        # X1_val = torch.from_numpy(X1_val).to(device)
        # X2_val = torch.from_numpy(X2_val).to(device)
        # # y_val = y_val.float().to(device)
        # y_val = torch.from_numpy(y_val).to(device)
        with torch.no_grad():
            # print('OOOOOOOOOO')
            y_pred = model(X1_val, X2_val).to(device)
            # print(y_pred, y_val)
            # vloss = loss_fn(y_pred, y_val.long())
            vloss = loss_fn(y_pred.squeeze(), y_val.float())
            # vloss = loss_fn(y_pred.sigmoid().squeeze(), y_val.float())
            history_val_loss.append(vloss.to('cpu'))
        
        # if task == 'multiclass':
        #     y_true = y_val.detach().to('cpu').numpy()
        #     y_pred = nn.functional.softmax(y_pred, dim=1)
        #     y_pred = y_pred.detach().to('cpu').numpy()
        #     roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')
        #     y_true = y_true.argmax(axis=1)
        #     y_pred = y_pred.argmax(axis=1)
        #     f1 = f1_score(y_true,y_pred, average='macro') 
            
        
        # else:
        y_true = y_val.detach().to('cpu').numpy()
        y_pred = y_pred.detach().to('cpu').sigmoid().numpy().round()
        roc_auc = roc_auc_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)


        
        acc = accuracy_score(y_true,y_pred)
        acc = float(acc)
        mcc = matthews_corrcoef(y_true,y_pred)  

        if (epoch % 10 == 0) and verbose:
            
            print('====> Epoch: {} TESTING Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} -- Average loss: {:.4f}'.format(
                                        epoch, acc, mcc, f1, roc_auc, vloss))

        if vloss < miniloss:
            if verbose:
                print ('+++ Saving best model because val_loss: {:.4f} < miniloss: {:.4f} +++'.format(
                                                                    vloss,miniloss))
                print ('+++ This model Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} +++'.format(acc,
                                                                                                                mcc,
                                                                                                                  f1,
                                                                                                                  roc_auc))
            miniloss = vloss
            PATH_BEST = "{}/{}_k{}_BEST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
            # torch.save(model.state_dict(), PATH_BEST)
            
    PATH_LAST = "{}/{}_k{}_LAST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
    # torch.save(model.state_dict(), PATH_LAST)
    
    # print ('After Training')
    # for i, param in enumerate(model.output_model.parameters()):
    #     if i>0:
    #         break
    #     print(param)
           
    
    # Return accuracy of best model on val set
    # checkpoint = torch.load(PATH_LAST)
    # model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        y_pred = model(X1_val, X2_val).to(device)
        # vloss = loss_fn(y_pred.sigmoid().squeeze(), y_val.float()).to('cpu')
        vloss = loss_fn(y_pred.squeeze(), y_val.float()).to('cpu')
        # y_pred_train = model(torch.from_numpy(X1_train).to(device), torch.from_numpy(X2_train).to(device)).to(device)
        # loss_train = loss_fn(y_pred_train.squeeze(), torch.from_numpy(y_train).float().to(device)).to('cpu')
 

    if task == 'multiclass':
        
        y_true = y_val.detach().to('cpu').numpy()
        y_pred = nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.detach().to('cpu').numpy()
        roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')
        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)
        # print(y_true, y_pred)
        f1 = f1_score(y_true,y_pred, average='macro') 
        
    else:
        
        y_true = y_val.detach().to('cpu').numpy()
        y_pred_val = y_pred.detach().to('cpu').sigmoid().numpy().round()
        f1 = f1_score(y_true,y_pred_val) 
        roc_auc = roc_auc_score(y_true,y_pred_val)


    acc = accuracy_score(y_true,y_pred_val)
    acc = float(acc)
    mcc = matthews_corrcoef(y_true,y_pred_val)  
    f1 = f1_score(y_true,y_pred_val) 
    roc_auc = roc_auc_score(y_true,y_pred_val) 
     
    val_scores = {'accuracy':acc, 'mcc': mcc, 'f1':f1, 'roc_auc':roc_auc, 'loss': vloss}
    
    ###########################

    ########### TRAIN SET 
    # if task == 'multiclass':
    #     y_true = y_train.detach().to('cpu').numpy()
    #     y_pred = nn.functional.softmax(y_pred_train, dim=1)
    #     y_pred = y_pred_train.detach().to('cpu').numpy()
        
    #     roc_auc_train = roc_auc_score(y_true,y_pred, multi_class='ovr')
    
    #     # print(y_true.shape, y_pred.shape)
    #     y_true = y_true.argmax(axis=1)
    #     y_pred = y_pred.argmax(axis=1)
   
    #     f1_train = f1_score(y_true,y_pred, average='macro') 
        
    # else:
    #     y_true = y_train.copy()
    #     y_pred = y_pred_train.detach().to('cpu').sigmoid().numpy().round()
    #     f1_train = f1_score(y_true,y_pred) 
    #     roc_auc_train = roc_auc_score(y_true,y_pred)
        
    
    # acc_train = (y_pred == y_true).mean()
    # mcc_train = matthews_corrcoef(y_true,y_pred)  
    
    
    # train_scores = {'accuracy':acc_train, 'mcc': mcc_train, 'f1':f1_train, 'roc_auc':roc_auc_train, 'loss': loss_train}
    
    
    
    
    
    
    return val_scores, None, history_train_loss, history_val_loss, y_pred_val
    # return val_scores, train_scores, history_train_loss, history_val_loss, y_pred_val

def model_train_multiomics_early(fold, model, model_name, X_train, y_train, X_val, y_val,
                 loss_fn, optimizer, writer=None,
                 n_epochs = 200, batch_size=16, device = 'cpu', outdir = './models/', verbose =False):

    device ='cuda' if torch.cuda.is_available() else 'cpu'

    dataset_train = Dataset_for_MLP(X_train,y_train)
    dataset_val = Dataset_for_MLP(X_val,y_val)
    
    loss_fn.to(device)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, pin_memory=True) 
    val_loader = DataLoader(dataset_val,batch_size=len(y_val), shuffle=False)
    all_train_loader = DataLoader(dataset_train,batch_size=len(y_val), shuffle=False)

    history_train_loss = []
    history_val_loss = []

    miniloss = 10000
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()  
            X_batch = X_batch.to(device).float()
            y_batch = y_batch.to(device).float()
            y_pred = model(X_batch).to(device)
            loss = loss_fn(y_pred.squeeze(), y_batch.float())
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        PATH_LAST = "{}/{}_k{}_BEST_EARLY.pth".format(outdir, model_name, fold+1) # i refers to the kfold
        torch.save(model.state_dict(), PATH_LAST)
        
        if (epoch % 10 == 0) and verbose :
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
        # evaluate accuracy at end of each epoch
        model.eval()
        X_val, y_val = next(iter(val_loader))
        X_val = X_val.to(device)
        # y_val = y_val.float().to(device)
        y_val = y_val.to(device)
        with torch.no_grad():
            # print('OOOOOOOOOO')
            y_pred = model(X_val).to(device)
            # print(y_pred, y_val)
            vloss = loss_fn(y_pred.squeeze(), y_val.float())
            history_val_loss.append(vloss.to('cpu'))
        
        y_true = y_val.detach().to('cpu').numpy()
        y_pred = y_pred.detach().to('cpu').sigmoid().numpy().round()
        
        # print(y_true.shape,y_pred.shape)
        acc = accuracy_score(y_true,y_pred)
        acc = float(acc)
        # acc1 = accuracy_score(y_val.detach().numpy(), y_pred.round().detach().numpy())
        mcc = matthews_corrcoef(y_true, y_pred)  
        f1 = f1_score(y_true,y_pred) 
        roc_auc = roc_auc_score(y_true, y_pred) 
        if (epoch % 10 == 0) and verbose:
            
            print('====> Epoch: {} TESTING Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} -- Average loss: {:.4f}'.format(
                                        epoch, acc, mcc, f1, roc_auc, vloss))
  
        
        
        if vloss < miniloss:
            if verbose:
                print ('+++ Saving best model because val_loss: {:.4f} < miniloss: {:.4f} +++'.format(
                                                                    vloss,miniloss))
                print ('+++ This model Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} +++'.format(acc,
                                                                                                                mcc,
                                                                                                                  f1,
                                                                                                                  roc_auc))
            miniloss = vloss
            PATH_BEST = "{}/{}_k{}_BEST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
            torch.save(model.state_dict(), PATH_BEST)
            
    PATH_LAST = "{}/{}_k{}_LAST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
    torch.save(model.state_dict(), PATH_LAST)
    
    checkpoint = torch.load(PATH_LAST)
    model.load_state_dict(checkpoint)

    
    X_train, y_train = next(iter(all_train_loader))
    
    model.eval()
    with torch.no_grad():
        y_pred = model(X_val).to(device)
        vloss = loss_fn(y_pred.squeeze(), y_val.float())

        y_pred_train = model(X_train.to(device).to(device))#.to(device)
        loss_train = loss_fn(y_pred_train.squeeze(), y_train.float().to(device)).to('cpu')

    # print('predicting val_set')
     ########### VALIDATION SET   
    y_true = y_val.detach().to('cpu').numpy()
    y_pred_detach = y_pred.detach().to('cpu')
    y_pred_rnd = y_pred_detach.sigmoid().numpy().round()
    y_pred = y_pred_detach.numpy()
    
    
    y_true_val = y_true.copy()
    y_pred_val = y_pred_rnd.copy()
    
    acc = accuracy_score(y_true_val,y_pred_val)
    acc = float(acc)
    mcc = matthews_corrcoef(y_true,y_pred_val)  
    f1 = f1_score(y_true,y_pred_val) 
    roc_auc = roc_auc_score(y_true,y_pred_detach.sigmoid().numpy()) 
    
    val_scores = {'accuracy':acc, 'mcc': mcc, 'f1':f1, 'roc_auc':roc_auc, 'loss': vloss}
    
    # print('predicting train_set')
    ########### TRAIN SET 
    y_true = y_train.numpy().copy()
    y_pred = y_pred_train.detach().to('cpu').sigmoid().numpy().round()
 
    acc_train = (y_pred == y_true).mean()
    mcc_train = matthews_corrcoef(y_true,y_pred)  
    f1_train = f1_score(y_true,y_pred) 
    roc_auc_train = roc_auc_score(y_true,y_pred) 
    
    train_scores = {'accuracy':acc_train, 'mcc': mcc_train, 'f1':f1_train, 'roc_auc':roc_auc_train, 'loss': loss_train}
    # print('done')
    
    #return val_scores, train_scores, history_train_loss, history_val_loss, y_true_val, y_pred_val
    return val_scores, train_scores, history_train_loss, history_val_loss, y_pred_val

def model_train_multichrome2(fold, model, model_name,
                           X1_train, X2_train, y_train, 
                           X1_val, X2_val, y_val,
                           loss_fn, optimizer, writer=None,
                           n_epochs = 200, batch_size=32,
                           mode = 'frozen',
                           device = 'cpu', 
                           outdir = './models/', 
                           task = 'binary',
                           verbose =False):
    '''
    X1: snp
    X2: genes
    '''
    device ='cuda' if torch.cuda.is_available() else 'cpu'
    loss_fn.to(device)

    dataset = MyDataset(X1_train, X2_train, y_train)
    dataset_val = MyDataset(X1_val, X2_val, y_val)
    loader_train = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=False )
    val_loader = DataLoader(dataset_val,batch_size=len(y_val), shuffle=False)

    history_train_loss = []
    history_val_loss = []

    X1_val, X2_val, y_val = next(iter(val_loader))
    X1_val = X1_val.to(device)
    X2_val = X2_val.to(device)
    # X1_val = torch.from_numpy(X1_val).to(device)
    # X2_val = torch.from_numpy(X2_val).to(device)
    y_val = y_val.float().to(device)
    # y_val = torch.from_numpy(y_val).to(device)

    miniloss = 10000
    
    print('')
    for epoch in range(n_epochs):
        
        model.train()
        if mode == 'frozen':
                ### Freeze modelA and modelB
            for param in model.modelA.parameters():
                param.requires_grad = False
                param.grad = None
            for param in model.modelB.parameters():
                param.requires_grad = False
                param.grad = None
            
        # # print('Model A')
        # for m in model.modelA.modules():
        #     # print(m)
        #     if m.__class__.__name__.startswith('Dropout'):
        #         m.eval()
        #     if m.__class__.__name__.startswith('Batch'):
        #         m.eval()
        # # print('Model B')       
        # for m in model.modelB.modules():
        #     # print(m)
        #     if m.__class__.__name__.startswith('Dropout'):
        #         m.eval()
        #     if m.__class__.__name__.startswith('Batch'):
        #         m.eval()
           
        train_loss = 0.0

        for batch_idx, (X1_batch, X2_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X1_batch = X1_batch.to(device).double()
            X2_batch = X2_batch.to(device).double()
            y_batch = y_batch.to(device).double()
            y_pred = model(X1_batch, X2_batch).to(device)
            # y_pred = model(X1_batch, X2_batch)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.long())
            loss = loss_fn(y_pred.squeeze(), y_batch.float())
            # loss = loss_fn(y_pred.sigmoid().squeeze(), y_batch.float())
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        # print progress
        
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
        # evaluate accuracy at end of each epoch
        # model.eval()
        # X1_val = torch.from_numpy(X1_val).to(device)
        # X2_val = torch.from_numpy(X2_val).to(device)
        # # y_val = y_val.float().to(device)
        # y_val = torch.from_numpy(y_val).to(device)
        with torch.no_grad():
            # print('OOOOOOOOOO')
            y_pred = model(X1_val, X2_val).to(device)
            # print(y_pred, y_val)
            # vloss = loss_fn(y_pred, y_val.long())
            vloss = loss_fn(y_pred.squeeze(), y_val.float())
            # vloss = loss_fn(y_pred.sigmoid().squeeze(), y_val.float())
            history_val_loss.append(vloss.to('cpu'))
        
        # if task == 'multiclass':
        #     y_true = y_val.detach().to('cpu').numpy()
        #     y_pred = nn.functional.softmax(y_pred, dim=1)
        #     y_pred = y_pred.detach().to('cpu').numpy()
        #     roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')
        #     y_true = y_true.argmax(axis=1)
        #     y_pred = y_pred.argmax(axis=1)
        #     f1 = f1_score(y_true,y_pred, average='macro') 
            
        
        # else:
        y_true = y_val.detach().to('cpu').numpy()
        y_pred = y_pred.detach().to('cpu').sigmoid().numpy().round()
        roc_auc = roc_auc_score(y_true,y_pred)
        f1 = f1_score(y_true,y_pred)


        
        acc = accuracy_score(y_true,y_pred)
        acc = float(acc)
        mcc = matthews_corrcoef(y_true,y_pred)  

        if (epoch % 10 == 0) and verbose:
            
            print('====> Epoch: {} TESTING Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} -- Average loss: {:.4f}'.format(
                                        epoch, acc, mcc, f1, roc_auc, vloss))
  
        
        
        if vloss < miniloss:
            if verbose:
                print ('+++ Saving best model because val_loss: {:.4f} < miniloss: {:.4f} +++'.format(
                                                                    vloss,miniloss))
                print ('+++ This model Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} +++'.format(acc,
                                                                                                                mcc,
                                                                                                                  f1,
                                                                                                                  roc_auc))
            miniloss = vloss
            PATH_BEST = "{}/{}_k{}_BEST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
            torch.save(model.state_dict(), PATH_BEST)
            
    PATH_LAST = "{}/{}_k{}_LAST.pth".format(outdir, model_name, fold+1) # i refers to the kfold
    torch.save(model.state_dict(), PATH_LAST)
    
    # print ('After Training')
    # for i, param in enumerate(model.output_model.parameters()):
    #     if i>0:
    #         break
    #     print(param)
           
    
    # Return accuracy of best model on val set
    checkpoint = torch.load(PATH_LAST)
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        y_pred = model(X1_val, X2_val).to(device)
        # vloss = loss_fn(y_pred.sigmoid().squeeze(), y_val.float()).to('cpu')
        vloss = loss_fn(y_pred.squeeze(), y_val.float()).to('cpu')
        # y_pred_train = model(torch.from_numpy(X1_train).to(device), torch.from_numpy(X2_train).to(device)).to(device)
        # loss_train = loss_fn(y_pred_train.squeeze(), torch.from_numpy(y_train).float().to(device)).to('cpu')
 

    if task == 'multiclass':
        
        y_true = y_val.detach().to('cpu').numpy()
        y_pred = nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.detach().to('cpu').numpy()
        roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')
        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)
        # print(y_true, y_pred)
        f1 = f1_score(y_true,y_pred, average='macro') 
        
    else:
        
        y_true = y_val.detach().to('cpu').numpy()
        y_pred_val = y_pred.detach().to('cpu').sigmoid().numpy().round()
        f1 = f1_score(y_true,y_pred_val) 
        roc_auc = roc_auc_score(y_true,y_pred_val)


    accuracy_score(y_true,y_pred_val)
    acc = float(acc)
    mcc = matthews_corrcoef(y_true,y_pred_val)  
    f1 = f1_score(y_true,y_pred_val) 
    roc_auc = roc_auc_score(y_true,y_pred_val) 
     
    val_scores = {'accuracy':acc, 'mcc': mcc, 'f1':f1, 'roc_auc':roc_auc, 'loss': vloss}
    
    ###########################

    ########### TRAIN SET 
    # if task == 'multiclass':
    #     y_true = y_train.detach().to('cpu').numpy()
    #     y_pred = nn.functional.softmax(y_pred_train, dim=1)
    #     y_pred = y_pred_train.detach().to('cpu').numpy()
        
    #     roc_auc_train = roc_auc_score(y_true,y_pred, multi_class='ovr')
    
    #     # print(y_true.shape, y_pred.shape)
    #     y_true = y_true.argmax(axis=1)
    #     y_pred = y_pred.argmax(axis=1)
   
    #     f1_train = f1_score(y_true,y_pred, average='macro') 
        
    # else:
    #     y_true = y_train.copy()
    #     y_pred = y_pred_train.detach().to('cpu').sigmoid().numpy().round()
    #     f1_train = f1_score(y_true,y_pred) 
    #     roc_auc_train = roc_auc_score(y_true,y_pred)
        
    
    # acc_train = (y_pred == y_true).mean()
    # mcc_train = matthews_corrcoef(y_true,y_pred)  
    
    
    # train_scores = {'accuracy':acc_train, 'mcc': mcc_train, 'f1':f1_train, 'roc_auc':roc_auc_train, 'loss': loss_train}
    
    
    
    
    
    
    return val_scores, None, history_train_loss, history_val_loss, y_pred_val
    # return val_scores, train_scores, history_train_loss, history_val_loss, y_pred_val

def model_train_multiomics(i, model, model_name,
                           X1_train, X2_train, y_train, 
                           X1_val, X2_val, y_val,
                 loss_fn, optimizer, writer=None,
                 n_epochs = 200, batch_size=8, device = 'cpu', task = 'multiclass', verbose =False):
    '''
    X1: metabolites
    X2: microbes
    '''

    dataset = MyDataset(X1_train, X2_train, y_train)
    loader_train = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True )

    history_train_loss = []
    history_val_loss = []

    miniloss = 10000
    
    print('')
    for epoch in range(n_epochs):
        
        model.train()
            ### Freeze modelA and modelB
        for param in model.modelA.parameters():
            param.requires_grad = False
            param.grad = None
        for param in model.modelB.parameters():
            param.requires_grad = False
            param.grad = None
        
        # # print('Model A')
        # for m in model.modelA.modules():
        #     # print(m)
        #     if m.__class__.__name__.startswith('Dropout'):
        #         m.eval()
        #     if m.__class__.__name__.startswith('Batch'):
        #         m.eval()
        # # print('Model B')       
        # for m in model.modelB.modules():
        #     # print(m)
        #     if m.__class__.__name__.startswith('Dropout'):
        #         m.eval()
        #     if m.__class__.__name__.startswith('Batch'):
        #         m.eval()
           
        train_loss = 0.0

        for batch_idx, (X1_batch, X2_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X1_batch = X1_batch.to(device).double()
            X2_batch = X2_batch.to(device).double()
            y_batch = y_batch.to(device).double()
            y_pred = model(X1_batch, X2_batch).to(device)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.long())
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        # print progress
        
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
        # evaluate accuracy at end of each epoch
        model.eval()
        X1_val = X1_val.to(device)
        X2_val = X2_val.to(device)
        # y_val = y_val.float().to(device)
        y_val = y_val.to(device)
        with torch.no_grad():
            # print('OOOOOOOOOO')
            y_pred = model(X1_val, X2_val).to(device)
            # print(y_pred, y_val)
            # vloss = loss_fn(y_pred, y_val.long())
            vloss = loss_fn(y_pred, y_val)
            history_val_loss.append(vloss.to('cpu'))

        if task == 'multiclass':
            y_true = y_val.detach().to('cpu').numpy()
            y_pred = nn.functional.softmax(y_pred, dim=1)
            y_pred = y_pred.detach().to('cpu').numpy()
            roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')
            y_true = y_true.argmax(axis=1)
            y_pred = y_pred.argmax(axis=1)
            f1 = f1_score(y_true,y_pred, average='macro') 
            
        
        else:
            y_true = y_val.detach().to('cpu').numpy()
            y_pred = y_pred.detach().to('cpu').numpy().round()
            roc_auc = roc_auc_score(y_true,y_pred)
            f1 = f1_score(y_true,y_pred)
        
 
        # print(y_pred, y_true)
        
        acc = (y_pred == y_true).mean()
        acc = float(acc)
        mcc = matthews_corrcoef(y_true,y_pred)  
        
        
        
        
        if (epoch % 10 == 0) and verbose:
            
            print('====> Epoch: {} TESTING Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} -- Average loss: {:.4f}'.format(
                                        epoch, acc, mcc, f1, roc_auc, vloss))
  
        
        
        if vloss < miniloss:
            if verbose:
                print ('+++ Saving best model because val_loss: {:.4f} < miniloss: {:.4f} +++'.format(
                                                                    vloss,miniloss))
                print ('+++ This model Accuracy {:.2f} -- MCC {:.2f} -- f1 {:.2f} -- roc_auc {:.2f} +++'.format(acc,
                                                                                                                mcc,
                                                                                                                  f1,
                                                                                                                  roc_auc))
            miniloss = vloss
            PATH_BEST = "./BEST_{}_k{}.pth".format(model_name, i) # i refers to the kfold
            torch.save(model.state_dict(), PATH_BEST)
    
    # print ('After Training')
    # for i, param in enumerate(model.output_model.parameters()):
    #     if i>0:
    #         break
    #     print(param)
           
    PATH_LAST = "./LAST_{}_k{}.pth".format(model_name, i) # i refers to the kfold
    torch.save(model.state_dict(), PATH_LAST)
    
    # Return accuracy of best model on val set
    checkpoint = torch.load('./LAST_{}_k{}.pth'.format(model_name,i))
    model.load_state_dict(checkpoint)

    model.eval()
    with torch.no_grad():
        y_pred = model(X1_val, X2_val).to(device)
        
        # vloss = loss_fn(y_pred, y_val.long())
        vloss = loss_fn(y_pred, y_val)

    if task == 'multiclass':
        
        y_true = y_val.detach().to('cpu').numpy()
        y_pred = nn.functional.softmax(y_pred, dim=1)
        y_pred = y_pred.detach().to('cpu').numpy()
        roc_auc = roc_auc_score(y_true,y_pred, multi_class='ovr')
        y_true = y_true.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)
        # print(y_true, y_pred)
        f1 = f1_score(y_true,y_pred, average='macro') 
        
    else:
        
        y_true = y_val.detach().to('cpu').numpy()
        y_pred = y_pred.detach().to('cpu').numpy().round()
        f1 = f1_score(y_true,y_pred) 
        roc_auc = roc_auc_score(y_true,y_pred)
        
    
    
    
    
    # print(y_true.shape, y_pred.shape)
    # 
    
    acc = (y_pred == y_true).mean()
    acc = float(acc)

    mcc = matthews_corrcoef(y_true,y_pred)  
 
    
    
     

    return acc, mcc,f1, roc_auc, vloss.to('cpu'), history_train_loss, history_val_loss, y_true, y_pred

def model_train_production_multiomics(model,
                                      X1_train, X2_train, 
                                      y_train,  loss_fn, optimizer, writer=None,
                                      n_epochs = 200, batch_size=8, device = 'cpu', verbose =False):

    dataset = MyDataset(X1_train, X2_train, y_train)
    loader_train = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True )


    history_train_loss = []
    # history_val_loss = []

    # miniloss = 10000
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (X1_batch, X2_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X1_batch = X1_batch.to(device)
            X2_batch = X2_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X1_batch, X2_batch).to(device)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.long())
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        # print progress
        
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
    
    print('Done. Saving last model as PROD_multiomicsDeep.pth')     
    PATH_LAST = "./PROD_multiomicsDeep.pth"
    torch.save(model.state_dict(), PATH_LAST)
    
   
    return  history_train_loss

def model_train_production_multiomics_merged(model,
                                      X1_train, X2_train, 
                                      y_train,  loss_fn, optimizer, writer=None,
                                      n_epochs = 200, batch_size=8, device = 'cpu', verbose =False):

    dataset = MyDataset(X1_train, X2_train, y_train)
    loader_train = DataLoader(dataset, shuffle=True, batch_size=batch_size, drop_last=True )


    history_train_loss = []
    # history_val_loss = []

    # miniloss = 10000
    
    for epoch in range(n_epochs):
        model.train()
        ### Freeze modelA and modelB
        for param in model.modelA.parameters():
            param.requires_grad = False
            param.grad = None
        for param in model.modelB.parameters():
            param.requires_grad = False
            param.grad = None
        train_loss = 0.0

        for batch_idx, (X1_batch, X2_batch, y_batch) in enumerate(loader_train):
            optimizer.zero_grad()  
#         print(X_batch.size())
#         break
            X1_batch = X1_batch.to(device)
            X2_batch = X2_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(X1_batch, X2_batch).to(device)
        
            # print(y_pred.size(), y_batch.size())
            # print(y_pred, y_batch)
#         break
            # loss = loss_fn(y_pred, y_batch.long())
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()
#         break

            loss.backward()
            optimizer.step()
        
        nbatches = batch_idx+1

        # print progress
        
        
        if (epoch % 10 == 0) and verbose :
            # acc = (y_pred.round() == y_batch).float().mean()
            # acc = float(acc)
            # # acc1 = accuracy_score(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # mcc = matthews_corrcoef(y_batch.detach().numpy(), y_pred.round().detach().numpy())
            # print('====> Epoch: {} TRAINING Accuracy {:.2f} -- MCC {:.2f} -- Average loss: {:.4f} -- nbatches: {}'.format(
                                        # epoch, acc, mcc, train_loss / nbatches, nbatches))
            print('====> Epoch: {} TRAINING  Average loss: {:.4f} -- nbatches: {}'.format(
                                        epoch, train_loss / nbatches, nbatches))
        history_train_loss.append(train_loss / nbatches)
        
    
    print('Done. Saving last model as PROD_multiomicsDeep.pth')     
    PATH_LAST = "./PROD_multiomicsDeep.pth"
    torch.save(model.state_dict(), PATH_LAST)
    
   
    return  history_train_loss

class EarlyStopping:

    """Early stops the training if validation loss doesn't improve after a given patience."""
    def _init_(self, patience=7, verbose=False, delta=0.0001, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def _call_(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss