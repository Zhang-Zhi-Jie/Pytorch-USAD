# -*- encoding: utf-8 -*-
'''
Filename         :data_preprocess.py
Description      :
Time             :2020/05/21 11:51:11
Author           :ZhiJie Zhang
Version          :1.0
'''
import numpy as np 
import pickle
import pandas as pd
import sklearn.preprocessing as preproc 
import torch
import torch.utils.data as Data
import os,shutil

def read_npy(file):
    """Read data from .npy file.

    Arguments
    ---------
    file: file path

    Returns
    -------
    data: a numpy array
    """
    data = np.load(file)
    return data


def read_pkl(file):
    """Read data from .pkl file.

    Arguments
    ---------
    file: file path

    Returns
    -------
    data: a numpy array
    """
    data = pickle.load(open(file, 'rb'))
    return data

def read_csv(file):
    """Read data from .csv file.

    Arguments
    ---------
    file: file path

    Returns
    -------
    df: a dataframe
    """
    df = pd.read_csv(file)
    return df

def dataframe_to_numpy(df):
    """Convert a dataframe into a numpy array.

    Arguments
    ---------
    df: a dataframe

    Returns
    -------
    data: a numpy array
    """
    data = np.array(df)
    return data

def numpy_to_dataframe(data, index):
    """Convert a numpy array into a dataframe.

    Arguments
    ---------
    data: a numpy array
    index: id

    Returns
    -------
    df: a dataframe
    """
    df = pd.DataFrame(data=data, index=index)
    return df

def min_max_scaling(data):
    """min-max scaling let data be within the range of [0, 1].
    
    Arguments
    ---------
    data: a numpy data

    Returns
    -------
    data_scaling: a numpy data after min-max scaling
    """
    data_scaling = preproc.minmax_scale(data)
    return data_scaling

def variance_scaling(data):
    """variance scaling let data has a mean of 0 and a variance of 1.
    
    Arguments
    ---------
    data: a numpy data

    Returns
    -------
    data_scaling: a numpy data after variance scaling
    """
    data_scaling = preproc.StandardScaler().fit_transform(data)
    return data_scaling

def data_to_dataLoader(x, label, batch_size, shuffle):
    """convert data into pytorch dataloader
    
    Arguments
    ---------
    x: a numpy data
    label: a numpy label
    batch_size: 
    
    Returns
    -------
    dataLoader: a pytorch dataloader
    """
    x = torch.from_numpy(x).float()
    label = torch.from_numpy(label).float()
    dataset = Data.TensorDataset(x, label)
    dataLoader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataLoader

def split_data(x, y, w_size, s_size):
    """
    
    Arguments
    ---------
    window_size: time-series window size

    Returns
    -------
    """
    sData = []
    sLabel = []
    for t in range(0, x.shape[0], s_size):
        if t+w_size < x.shape[0]:
            sData.append(x[t:t+w_size, :])
            if 1 in y[t:t+w_size]:
                sLabel.append(1)
            else:
                sLabel.append(0)
    return np.array(sData), np.array(sLabel)

            


    

    

    
    
    
    
