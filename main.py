import torch
from torch.autograd import grad
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset
from scipy.special import digamma
from torch.autograd.functional import hessian
from torch.autograd.functional import jacobian
from functorch import make_functional, jvp, grad, hessian, jacfwd, jacrev
import statsmodels.api as sm
import warnings
from torch.autograd.functional import jacobian
import copy
from scipy.signal import periodogram
import time
import torch.nn.functional as FF
from functorch import make_functional, vmap, vjp, jvp, jacrev
from utils import *
from DLinear import *
import argparse
warnings.filterwarnings("ignore")


def replace_top_percent(array1, array2, array3):
    array2 = np.copy(array2)

    if len(array1) != array2.shape[0] or len(array1) != array3.shape[0] or array2.shape != array3.shape:
        raise ValueError("Input arrays must have the same length or shape.")

    positive_values = array1[array1 > 0]

    threshold = np.percentile(positive_values, 90)

    top_percent_indices = np.where(array1 >= threshold)

    array2[top_percent_indices] = array3[top_percent_indices]

    return array2

def main(args):
    setup_seed(1)
    name = args.name
    learning_rate = args.learning_rate
    epochs = args.epochs
    device = args.device
    repeat_times = args.repeat_times
    seq_len,pred_len = 24,24
    model = DLinear(seq_len,pred_len).to(device)
    train_data = np.load('./imputated_folder/'+name+'/train_data_mean.npy')
    train_label = np.load('./imputated_folder/'+name+'/train_label_mean.npy')
    val_data = np.load('./imputated_folder/'+name+'/val_data.npy')
    val_label = np.load('./imputated_folder/'+name+'/val_label.npy')
    test_data = np.load('./imputated_folder/'+name+'/test_data.npy')
    test_label = np.load('./imputated_folder/'+name+'/test_label.npy')

    create_directory_if_not_exists('./estimation_folder')
    create_directory_if_not_exists('./estimation_folder/'+name)

    create_directory_if_not_exists('./result_folder')
    create_directory_if_not_exists('./result_folder/'+name)

    phi_trace,model = compute_phi_trace(model, torch.Tensor(train_data), torch.Tensor(train_label), torch.Tensor(val_data), torch.Tensor(val_label), learning_rate, epochs,train_criterion = nn.MSELoss(),device=device)
    np.save('./estimation_folder/'+name+'/phi_trace_mean.npy',phi_trace)
    result = test_model(model,test_data,test_label)
    result_dict['mean'] = result

    if name == 'ELE':
        name_list = ['saits','brits','gpvae','usgan']
    else:
        name_list = ['saits','brits','mrnn','gpvae','usgan']


    for im_name in name_list:
        print(im_name)
        temp_result = []
        for i in range(1,repeat_times):
            print(i)
            setup_seed(1)
            train_label_im = np.load('./imputated_folder/'+name+'/train_label_'+im_name+'_'+str(i)+'_.npy')
            model = DLinear(seq_len,pred_len).to(device)
            loss_function = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            model = train_model(model,train_data,train_label_im,val_data,val_label,epochs = epochs,lr= learning_rate,batch_size=len(train_data),loss_function=loss_function)
            result = test_model(model,test_data,test_label)
            temp_result.append(result)
        print(np.mean(temp_result))
        result_dict[im_name] = [np.mean(temp_result),np.std(temp_result)]
    
    with open('./result_folder/'+name+'/result_dict.json', 'w') as file:
        json.dump(result_dict, file)
    
    im_result_dict = {}

    for im_name in name_list:
        print(im_name)
        temp_result = []
        for i in range(1,repeat_times):
            print(i)
            setup_seed(1)
            
            train_label_im = np.load('./imputated_folder/'+name+'/train_label_'+im_name+'_'+str(i)+'_.npy')
            eps = train_label-train_label_im

            phi_trace = np.load('./estimation_folder/'+name+'/phi_trace_mean.npy')
            phi_trace = phi_trace.reshape(phi_trace.shape[0], phi_trace.shape[1], 1)

            effect = phi_trace*eps
            new_y = replace_top_percent(effect,train_label,train_label_im)


            model = DLinear(seq_len,pred_len).to(device)
            loss_function = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            model = train_model(model,train_data,new_y,val_data,val_label,epochs = epochs,lr= learning_rate,batch_size=len(train_data),loss_function=loss_function)
            result = test_model(model,test_data,test_label)
            temp_result.append(result)
        print(np.mean(temp_result))
        im_result_dict[im_name+'+en'] = [np.mean(temp_result),np.std(temp_result)]
    with open('./result_folder/'+name+'/im_result_dict.json', 'w') as file:
        json.dump(im_result_dict, file)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name", help="name of the data",default='GEF',choices=['GEF','ETTH1','ETTH2','ELE','Traffic','Air'])
    parser.add_argument("-lr","--learning_rate", type=int,help="learning rate",default=0.1)
    parser.add_argument("-e","--epochs", type=int,help="training epochs",default=300)
    parser.add_argument("-d","--device",help="device to run",default='cuda')
    parser.add_argument("-rt","--repeat_times", type=int,help="times of repeating experiment",default=3)
    args = parser.parse_args()
    main(args)



