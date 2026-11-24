# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 10:08:51 2026

@author: rpollet
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn

base_dir = 'where_is_the_folder/models/'

## open the dataset
df = ... 

# model architecture : Multi-Layer Perceptron (MLP)
class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, activation_func):
        super(MLPModel, self).__init__() 
        # list used to store the successive layers of the network
        layers = []
        # number of input features for the first layer
        in_features = input_size 
        # build each hidden layer
        for h in hidden_layers:
            # linear transformation from the previous layer size to the current hidden size
            layers.append(nn.Linear(in_features, h)) 
            # activation function applied after the linear layer
            layers.append(activation_func) 
            in_features = h 
        # final output layer
        layers.append(nn.Linear(in_features, output_size))  
        self.model = nn.Sequential(*layers)
    # forward pass through the network
    def forward(self, x):
        return self.model(x) 

# custom scaled hyperbolic tangent activation function
class ScaledTanh(nn.Module):
    # scaling factor
    def __init__(self, A=4/3, alpha=1.7159):
        super().__init__()
        self.A = A
        self.alpha = alpha

    # comput the scaled tanh-like activation 
    def forward(self, x):
        return self.A * ((torch.exp(self.alpha * x) - 1) / (torch.exp(self.alpha * x) + 1))
    
def cmed_no3(df, psal, pres, doxy, temp, lon, lat) :
    # Multi-Layer Perceptron to predict nitrates (µmol/kg)
    # Neural Network training by Romane Pollet from work by Marine Fourrier in LOV

    # input:
    # df : pandas.DataFrame
    # psal: array-like, practical salinity (PSU)
    # pres: array-like, pressure (dbar)
    # doxy: array-like, dissolved oxgyen (µmol/kg)
    # temp: array-like, temperature (°C)
    # lon: array-like, longitude (°E)
    # lat: array-like, latitude (°N)

    # output:
    # out: nitrates (NO3)
    #      - no3_pred: predicted no3
    #      - no3_std: ensemble standard deviation

    # calculate the input parameter required for the models
    df['pres_transf'] = pres / 2e4 + (1 / (1 + np.exp(-pres / 300))** 3)

    # input combination
    features_model = ['pres_transf', 'longitude','latitude', 'psal', 'temp','doxy']

    # load input and ouput scaler
    in_scaler = torch.load(base_dir+"input_scaler_no3.pt", weights_only=False)
    out_scaler = torch.load(base_dir+"output_scaler_no3.pt", weights_only=False)
    mean = pd.Series(in_scaler['mean'], index=features_model)
    std  = pd.Series(in_scaler['std'],  index=features_model)

    X_all = pd.DataFrame({
        'pres_transf' : df['pres_transf'].values, 'longitude' : np.asarray(lon),
        'latitude' : np.asarray(lat), 'psal' : np.asarray(psal), 
        'temp' : np.asarray(temp), 'doxy' : np.asarray(doxy)})[features_model]

    # normalisation of input parameters
    X_all_scaled = (2/3) * ((X_all - mean) / std)
    xtest_tensor = torch.tensor(X_all_scaled.values, dtype=torch.float32)

    # load neural network models and its architecture
    checkpoint = torch.load(base_dir+"cmed_no3_models.pt", weights_only=False)
    models = [] 
    for model_info in checkpoint:
        m = MLPModel(
            input_size=model_info['input_size'],
            hidden_layers=[model_info['n1'], model_info['n2']],
            output_size=model_info['output_size'],
            activation_func=ScaledTanh())
        m.load_state_dict(model_info['model_state_dict'])
        m.eval()
        models.append(m)

    # prediction for each model
    predictions = []
    with torch.no_grad():
        for m in models : 
            pred = m(xtest_tensor).detach().cpu().numpy()
            predictions.append(pred)

    # denormalisation of output 
    predictions = np.stack(predictions, axis=0)
    predictions_real = (3/2) * predictions * out_scaler['std'] + out_scaler['mean']

    # mean and standard deviation of models
    y_pred = np.mean(predictions_real, axis=0)
    y_std = np.std(predictions_real, axis=0)

    # add predictions and uncertainties on dataframe
    df['no3_pred'] = y_pred.squeeze()
    df['no3_std'] = y_std.squeeze()

    return df

# run nitrate models on df 
# warning: change the name columns
df = cmed_no3(df, df.psal, df.pres, df.doxy, df.temp, df.lon, df.lat)


def cmed_po4(df, psal, pres, doxy, temp, lon, lat):
    # Multi-Layer Perceptron to predict phosphates (µmol/kg)
    # Neural Network training by Romane Pollet from work by Marine Fourrier in LOV

    # input:
    # df : pandas.DataFrame
    # psal: array-like, practical salinity (PSU)
    # pres: array-like, pressure (dbar)
    # doxy: array-like, dissolved oxgyen (µmol/kg)
    # temp: array-like, temperature (°C)
    # lon: array-like, longitude (°E)
    # lat: array-like, latitude (°N)

    # output:
    # out: phosphates concentration (PO4) 
    #      - po4_pred: predicted PO4
    #      - po4_std: ensemble standard deviation

    # calculate the input parameter required for the models
    df['pres_transf'] = pres / 2e4 + (1 / (1 + np.exp(-pres / 300))** 3)

    # input combination
    features_model = ['pres_transf', 'longitude','latitude','psal', 'temp','doxy']

    # load input and output scaler 
    in_scaler = torch.load(base_dir+"input_scaler_po4.pt", weights_only=False)
    out_scaler = torch.load(base_dir+"output_scaler_po4.pt", weights_only=False)
    mean = pd.Series(in_scaler['mean'], index=features_model)
    std  = pd.Series(in_scaler['std'],  index=features_model)

    X_all = pd.DataFrame({
        'pres_transf' : df['pres_transf'].values , 'longitude' : np.asarray(lon),
        'latitude' : np.asarray(lat), 'psal' : np.asarray(psal),
        'temp': np.asarray(temp), 'doxy' : np.asarray(doxy)})[features_model]

    # normalisation of input parameters 
    X_all_scaled = (2/3) * ((X_all - mean) / std)
    xtest_tensor = torch.tensor(X_all_scaled.values, dtype=torch.float32)

    # load neural network models and its architecture
    checkpoint = torch.load(base_dir+"cmed_po4_models.pt", weights_only=False)
    models = [] 
    for model_info in checkpoint:
        m = MLPModel(
            input_size=model_info['input_size'],
            hidden_layers=[model_info['n1'], model_info['n2']],
            output_size=model_info['output_size'],
            activation_func=ScaledTanh())
        m.load_state_dict(model_info['model_state_dict'])
        m.eval()
        models.append(m)

    # prediction for each model
    predictions = []
    with torch.no_grad():
        for m in models : 
            pred = m(xtest_tensor).detach().cpu().numpy()
            predictions.append(pred)

    # denormalisation of output 
    predictions = np.stack(predictions, axis=0) 
    predictions_real = (3/2) * predictions * out_scaler['std'] + out_scaler['mean']

    # mean and standard deviation of models
    y_pred = np.mean(predictions_real, axis=0)
    y_std = np.std(predictions_real, axis=0)

    # add predictions and uncertainties on dataframe
    df['po4_pred'] = y_pred.squeeze()
    df['po4_std'] = y_std.squeeze()

    return df

# run phosphate models on df 
# warning: change the name columns
df = cmed_po4(df, df.psal, df.pres, df.doxy, df.temp, df.lon, df.lat)

def cmed_sioh(df, date, psal, pres, doxy, temp, lon, lat):
    # Multi-Layer Perceptron to predict silicates (µmol/kg)
    # Neural Network training by Romane Pollet from work by Marine Fourrier in LOV

    # input:
    # df : pandas.DataFrame
    # psal: array-like, practical salinity (PSU)
    # pres: array-like, pressure (dbar)
    # doxy: array-like, dissolved oxgyen (µmol/kg)
    # temp: array-like, temperature (°C)
    # lon: array-like, longitude (°E)
    # lat: array-like, latitude (°N)

    # output:
    # out: silicates concentration (SiOH4)
    #      - sioh_pred: predicted SiOH
    #      - sioh_std: ensemble standard deviation

    date = pd.Series(pd.to_datetime(date))

    # calculate the input parameter required for the models
    df['pres_transf'] = pres / 2e4 + (1 / (1 + np.exp(-pres / 300))** 3) 
    
    # input combination
    features_model = ['pres_transf', 'longitude','latitude', 'psal', 'temp', 'doxy']

    # load input and output scaler 
    in_scaler = torch.load(base_dir+"input_scaler_sioh.pt", weights_only=False)
    out_scaler = torch.load(base_dir+"output_scaler_sioh.pt", weights_only=False)
    mean = pd.Series(in_scaler['mean'], index=features_model)
    std  = pd.Series(in_scaler['std'],  index=features_model)

    X_all = pd.DataFrame({
        'pres_transf' : df['pres_transf'].values, 'longitude' : np.asarray(lon),
        'latitude': np.asarray(lat), 'psal' : np.asarray(psal),
        'temp': np.asarray(temp),  'doxy' : np.asarray(doxy)})[features_model]

    # normalisation of input parameters
    X_all_scaled = (2/3) * ((X_all - mean) / std)
    xtest_tensor = torch.tensor(X_all_scaled.values, dtype=torch.float32)

    # load neural network models and its architecture
    checkpoint = torch.load(base_dir+"cmed_sioh_models.pt", weights_only=False)
    models = [] 
    for model_info in checkpoint:
        m = MLPModel(
            input_size=model_info['input_size'],
            hidden_layers=[model_info['n1'], model_info['n2']],
            output_size=model_info['output_size'],
            activation_func=ScaledTanh())
        m.load_state_dict(model_info['model_state_dict'])
        m.eval()
        models.append(m)

    # prediction for each model
    predictions = []
    with torch.no_grad():
        for m in models : 
            pred = m(xtest_tensor).detach().cpu().numpy()
            predictions.append(pred)

    # denormalisation of output
    predictions = np.stack(predictions, axis=0) #(n_models, N,1)
    predictions_real = (3/2) * predictions * out_scaler['std'] + out_scaler['mean']

    # mean and standard deviation of models
    y_pred = np.mean(predictions_real, axis=0)
    y_std = np.std(predictions_real, axis=0)

    # add predictions and uncertainties on dataframe
    df['sioh_pred'] = y_pred.squeeze()
    df['sioh_std'] = y_std.squeeze()

    return df

# run silicate models on df 
# warning: change the name columns
df = cmed_sioh(df, df.psal, df.pres, df.doxy, df.temp, df.lon, df.lat)