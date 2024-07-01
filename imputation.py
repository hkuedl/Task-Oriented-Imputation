import torch
import torch.nn as nn
import pandas as pd
import random
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from pypots.imputation import SAITS,BRITS,GPVAE,MRNN,USGAN
TF_ENABLE_ONEDNN_OPTS=0
import argparse
from utils import *

def random_subarray(array, period=24):
    array1 = np.copy(array)
    array2 = np.copy(array)
    length = len(array)
    num_to_select = int(length * 0.4)
    selected_length = 0

    replaced_indices = []  
    replaced_intervals = []  

    while selected_length < num_to_select:
        period_lengths = [period // 10, period // 4, period // 5, period // 2, period, period * 2, period * 4, period * 5]
        subarray_length = random.choice(period_lengths)

        if subarray_length + selected_length > num_to_select:
            subarray_length = num_to_select - selected_length

        start_idx = random.randint(0, length - subarray_length)
        end_idx = start_idx + subarray_length - 1

        overlapping = False
        for interval in replaced_intervals:
            if (start_idx >= interval[0] and start_idx <= interval[1]) or (end_idx >= interval[0] and end_idx <= interval[1]):
                overlapping = True
                break

        if overlapping:
            continue  

        array1[start_idx:end_idx + 1] = np.mean(array[start_idx:end_idx + 1])
        array2[start_idx:end_idx + 1] = np.nan

        replaced_indices.extend(range(start_idx, end_idx + 1))  
        replaced_intervals.append((start_idx, end_idx))  

        selected_length += subarray_length

    return array1, array2, replaced_indices  
def replace_nan_with_array(array1, array2):
    if array1.shape != array2.shape:
        raise ValueError("error")

    result_array = np.copy(array1)
    nan_indices = np.isnan(array1)
    result_array[nan_indices] = array2[nan_indices]
    
    return result_array


def main(args):
    setup_seed(0)
    name = args.name
    repeat_times = args.repeat_times
    device = args.device
    data = pd.read_csv(args.data_path,index_col=0)
    data.index = pd.to_datetime(data.index)
    # deine timestep
    time_1 = pd.Timestamp(args.time_1)
    time_2 = pd.Timestamp(args.time_2)
    time_3 = pd.Timestamp(args.time_3)
    # split the data
    train = np.array(data.loc[time_1:time_2].iloc[:,-1:]).reshape(-1)
    val = np.array(data.loc[time_2:time_3].iloc[:,-1:])
    test = np.array(data.loc[time_3:].iloc[:,-1:])

    train_mean,train_nan,nan_index = random_subarray(train,24)
    train_mean = train_mean.reshape(-1,1)
    train_nan = train_nan.reshape(-1,1)
    scaler = StandardScaler()
    scaler.fit(train_mean)
    train_mean = scaler.transform(train_mean)
    train_nan = scaler.transform(train_nan)
    val =  scaler.transform(val)
    test = scaler.transform(test)

    train_data_mean,train_label_mean = generate_data(train_mean)
    train_data_nan,train_label_nan = generate_data(train_nan)
    val_data,val_label = generate_data(val)
    test_data,test_label = generate_data(test)
    

    create_directory_if_not_exists('./imputated_folder')
    create_directory_if_not_exists('./imputated_folder/'+name)

    np.save('./imputated_folder/'+name+'/train_data_mean.npy', train_data_mean)
    np.save('./imputated_folder/'+name+'/train_label_mean.npy', train_label_mean)
    np.save('./imputated_folder/'+name+'/val_data.npy', val_data)
    np.save('./imputated_folder/'+name+'/val_label.npy', val_label)
    np.save('./imputated_folder/'+name+'/test_data.npy', test_data)
    np.save('./imputated_folder/'+name+'/test_label.npy', test_label)

    im_dataset = {"X": train_label_nan}

    for i in range(repeat_times):
        setup_seed(i+1)
        model = SAITS(n_steps=24, n_features=1, n_layers=2, d_model=64, d_ffn=32, n_heads=4, d_k=16, d_v=16, dropout=0.1, epochs=100,patience=10,device = device)
        model.fit(im_dataset)
        imputation = model.impute(im_dataset)
        im_result = replace_nan_with_array(train_label_nan,imputation)
        np.save('./imputated_folder/'+name+'/train_label_saits_'+str(i+1)+'_.npy',im_result)

        model = BRITS(n_steps=24, n_features=1, rnn_hidden_size=64,epochs=100,patience=10,device = device)
        model.fit(im_dataset)
        imputation = model.impute(im_dataset)
        im_result = replace_nan_with_array(train_label_nan,imputation)
        np.save('./imputated_folder/'+name+'/train_label_brits_'+str(i+1)+'_.npy',im_result)

        model = GPVAE(n_steps=24, n_features=1, latent_size=64,epochs=100,patience=10,device = device)
        model.fit(im_dataset)
        imputation = model.impute(im_dataset)[:,0,:,:]
        im_result = replace_nan_with_array(train_label_nan,imputation)
        np.save('./imputated_folder/'+name+'/train_label_gpvae_'+str(i+1)+'_.npy',im_result)

        if name != 'ELE':
            model = MRNN(n_steps=24, n_features=1, rnn_hidden_size=64,epochs=100,patience=10,device = device)
            model.fit(im_dataset)
            imputation = model.impute(im_dataset)
            im_result = replace_nan_with_array(train_label_nan,imputation)
            np.save('./imputated_folder/'+name+'/train_label_mrnn_'+str(i+1)+'_.npy',im_result)

        model = USGAN(n_steps=24, n_features=1, rnn_hidden_size=64,epochs=100,patience=10,device = device)
        model.fit(im_dataset)
        imputation = model.impute(im_dataset)
        im_result = replace_nan_with_array(train_label_nan,imputation)
        np.save('./imputated_folder/'+name+'/train_label_usgan_'+str(i+1)+'_.npy',im_result)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--data_path", help="data path",default='./data/GEF.csv')
    parser.add_argument("-n","--name", help="name of the data",default='GEF',choices=['GEF','ETTH1','ETTH2','ELE','Traffic','Air'])
    parser.add_argument("-rt","--repeat_times", type=int,help="times of repeating experiment",default=3)
    parser.add_argument("-d","--device",help="device to run",default='cuda')
    args = parser.parse_args()
    with open("./time_settings.json", "r") as file:
        time_settings = json.load(file)
    if args.name in time_settings:
        args.time_1, args.time_2, args.time_3 = time_settings[args.name]
    main(args)



