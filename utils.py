import os
import torch
import torch.nn as nn
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

import torch
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian

from functorch import make_functional, jvp, grad, hessian, jacfwd, jacrev
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from scipy.interpolate import interp1d
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.impute import KNNImputer
import warnings
from torch.autograd.functional import jacobian
import copy
from scipy.signal import periodogram
# import torch.autograd.functional as F
import time
import torch.nn.functional as FF
from functorch import make_functional, vmap, vjp, jvp, jacrev
def generate_data(arr: np.array, window_size = 24, num_labels = 24):
    data, labels = [], []

    for i in range(len(arr) - window_size - num_labels + 1):
        data.append(arr[i:i + window_size])
        labels.append(arr[i + window_size:i + window_size + num_labels])

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def ntk(model, x_i, y_i, x, y, window_length=24,loss_fn=nn.MSELoss(),batch_size=512):
    def fnet_single(params, x):
        return fnet(params, x.unsqueeze(0)).squeeze(0)
    def empirical_ntk(fnet_single, params, x2):
        # # Compute J(x1)
        # jac1 = vmap(jacrev(fnet_single), (None, 0))(params, x1)
        # jac1 = [j.flatten(2) for j in jac1]
        
        # Compute J(x2)
        jac2 = vmap(jacrev(fnet_single), (None, 0))(params, x2)
        jac2 = [j.flatten(2) for j in jac2]
        
        # # Compute J(x1) @ J(x2).T
        # result = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
        # result = result.sum(0)
        return torch.cat(jac2,-1)
    # y_i_pred = model(x_i).reshape(-1)
    y_pred = model(x).reshape(-1)
    loss = loss_fn(y_pred, y.reshape(-1))

    grad_theta_y = torch.autograd.grad(loss, model.parameters())
    grad_theta_y = [g.contiguous() for g in grad_theta_y]
    grad_theta_y_vector = torch.nn.utils.parameters_to_vector(grad_theta_y).detach()
    fnet, params = make_functional(model)
    n_samples = x_i.size(0)
    ntk_kernel = torch.zeros([n_samples, window_length], device=x.device)

    for batch_start in range(0, n_samples, batch_size):
        batch_end = min(batch_start + batch_size, n_samples)
        x_batch = x_i[batch_start:batch_end]
        jacobian_y_i_pred_batch = empirical_ntk(fnet_single, params, x_batch)
        ntk_kernel_batch = torch.matmul(jacobian_y_i_pred_batch, grad_theta_y_vector.view(-1,1))
        batch_size_in = batch_end - batch_start
        ntk_kernel[batch_start:batch_end] = ntk_kernel_batch.squeeze().reshape(batch_size_in, -1)
    return ntk_kernel

def my_get_yh_v_product(model: nn.Module, X, y, v, loss_fn=nn.MSELoss()):
    """
    Compute the product of the Jacobian matrix of the gradient of the loss function with respect to the predictions (mu),
    with respect to the target values (y) for the whole training set, and a given vector v.
    """
    y = y.reshape(-1)
    y.requires_grad_(True)
    pred = model(X).reshape(-1)
    loss = loss_fn(y, pred)

    # First order gradient with respect to predictions (mu)
    grad = torch.autograd.grad(outputs=loss, inputs=pred, create_graph=True)
    grad = torch.cat([x.flatten() for x in grad], dim=0)

    # Compute the product Jv
    # Jv = torch.zeros_like(y).to(X.device)
    # for i, g in enumerate(grad):
    #     Jv += g * v[i]
    Jv = torch.matmul(grad.view(-1,1).T, v.view(-1,1)).reshape(-1)
    # Compute the gradient of Jv with respect to y
    Jv_grad = torch.autograd.grad(outputs=Jv.sum(), inputs=y,retain_graph=True)
    Jv_grad = torch.cat([x.flatten() for x in Jv_grad], dim=0)

    return Jv_grad

def compute_phi_trace(model, xx, yy, x,y, learning_rate, epochs,train_criterion,device = 'cuda',seed = 1):
    setup_seed(seed)
    x_i,y_i,x,y = xx.to(device),yy.to(device),x.to(device),y.to(device)
    window_length = x_i.shape[1]
    phi_trace = 0.0

    loss_function = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    with trange(epochs) as t:
        for i in t:
            y_i_pred = model(x_i).reshape(-1)
    
            loss_i = loss_function(y_i_pred, y_i.reshape(-1))

            local_importance = ntk(model,x_i,y_i, x,y,window_length)
    
            global_importance = my_get_yh_v_product(model, x_i,y_i,local_importance.reshape(-1),loss_fn=train_criterion)

                        
            phi_trace -= learning_rate * global_importance
        
    
            optimizer.zero_grad()
            loss_i.backward()
            optimizer.step()
            t.set_description(f'Training (loss: {loss_i.item():.4f})')
    phi_trace = phi_trace.cpu().detach().numpy().reshape(-1,window_length)
    return phi_trace,model

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


class dataset(Dataset):
    def __init__(self, data ,label):

        self.seq = data
        self.label = label
        

    def __getitem__(self, idx):
        
        data_idx = torch.Tensor(self.seq[idx])
        label_idx = torch.Tensor(self.label[idx])
        return data_idx, label_idx

    def __len__(self):

        return len(self.seq)

def train_model(model,train_data,train_label,val_data,val_label,epochs = 300,lr= 0.1,batch_size=64,loss_function=nn.MSELoss()):
    X = torch.Tensor(train_data).to(device)
    y = torch.Tensor(train_label).to(device)
    val_X = torch.Tensor(val_data).to(device)
    val_y = torch.Tensor(val_label).to(device)
    trainset = dataset(X,y)
    valset = dataset(val_X,val_y)
    train_loader = DataLoader(trainset, shuffle=False, batch_size=batch_size)
    val_loader = DataLoader(valset, shuffle=False, batch_size=batch_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_min = np.inf
    count = 0
    for i in range(epochs):
        losses = []
        for (data,label) in train_loader:
            input_seq = data.to(device)
            label = label.to(device)
            mu = model(input_seq)
            loss = loss_function(label.reshape(-1),mu.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        for (data,label) in val_loader:
            input_seq = data.to(device)
            label = label.to(device)
            with torch.no_grad():
                mu = model(input_seq)
            loss = loss_function(label.reshape(-1),mu.reshape(-1))
            if loss<=loss_min:
                count = 0
                loss_min = loss
            else:
                count = count+1
        if count>=10:
            break

    return(model)


def test_model(model,test_data,test_label):
    with torch.no_grad():
        y_pred = model(torch.Tensor(test_data).to(device))
    # Convert PyTorch tensors to NumPy arrays
    x_test_np = test_data
    y_test_np = test_label
    y_pred_np = y_pred.cpu().numpy()
    result = MSE(y_test_np.reshape(-1),y_pred_np.reshape(-1))
    return result