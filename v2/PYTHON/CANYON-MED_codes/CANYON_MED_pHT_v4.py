import numpy as np
import pandas as pd
import math
from datetime import datetime

def CANYON_MED_pHT_v4(date, lat, lon, pres, temp, psal, doxy):
    # Multi-layer perceptron to predict total pH (total scale at insitu PTS)
    #
    # Neural network training by Marine Fourrier from work by Raphaelle Sauzede, LOV;
    # as Python function by Florian Ricour & Marine Fourrier, LOV
    #
    #
    # input:
    # date  - date (UTC) as string ('yyyy-mm-dd')
    # lat   - latitude / °N  [-90 90]
    # lon   - longitude / °E [-180 180] or [0 360]
    # pres  - pressure / dbar
    # temp  - in-situ temperature / °C
    # psal  - salinity
    # doxy  - dissolved oxygen / umol kg-1
    #
    # output:
    # out   - pHT (total scale at insitu PTS)
    #
	# check value:  8.0965
	# for 09-Apr-2014, 35° N, 18° E, 500 dbar, 13.5 °C, 38.6 psu, 160 umol O2 kg-1
    # for example for one and two outputs
	# CANYON_MED_pHT_v4(['2014-04-09'], [35], [18], [500], [13.5], [38.6], [160])
	# CANYON_MED_pHT_v4(['2014-04-09','2014-04-09'], [35,35], [18,18], [500,500], [13.5,13.5], [38.6,38.6], [160,160])
    #
    # 13.06.2023

    # No input checks! Assumes informed use, e.g., same dimensions for all
    # inputs, ...
  
    basedir = "/drive/notebooks/"  # relative or absolute path to CANYON-MED folder
    
 
    # create panda dataframe with input data
    df = pd.DataFrame({'lat' : np.array(lat), 'lon' : np.array(lon), 'date' : np.array(date),
                      'temp' : np.array(temp), 'psal' : np.array(psal), 'doxy' : np.array(doxy),
                      'pres' : np.array(pres)})
    
    
    def calculate_decimal_year(date):
        date = datetime.strptime(date, '%Y-%m-%d')
        year = date.year
        base_date = datetime(year, 1, 1)
        total_seconds = (date - base_date).total_seconds()
        decimal_year = year + total_seconds / (365.0 * 24 * 60 * 60)
        return decimal_year
    
    # convert year to decimal year
    df['date'] = df['date'].apply(lambda x : calculate_decimal_year(x))
    df.rename(columns = {'date' : 'dec_year'})
    
    # convert lon in -180; 180
    df['lon'] = df['lon'].apply(lambda x : x - 360 if x>180 else x)

    # convert pres 
    df['pres'] = df['pres'].apply(lambda x : (x/2e4) + (1/((1 + np.exp(-x/300))**3)))
    
    # input sequence:
    #     lat,   lon,    dec_year,    temp,   sal,    oxygen, P

    moy_F = np.array(pd.read_table(basedir + "CANYON-MED_weights/moy_ph_F.txt", sep=" {3}", header=None, engine = 'python'))
    std_F = np.array(pd.read_table(basedir + "CANYON-MED_weights/std_ph_F.txt", sep=" {3}", header=None, engine = 'python'))

    ne = 7  # Number of inputs

    # NORMALISATION OF THE PARAMETERS
    data_N = df.iloc[:, :ne].copy()

    for i in range(ne):
        data_N.iloc[:, i] = (2 / 3) * ((df.iloc[:, i] - moy_F[:, i]) / std_F[:, i])

    data_N = np.array(data_N)

    n_list = 5
    rx = data_N.shape[0]
    
    ph_outputs_s = []
    
    # function ### see Eq. XXX in paper XXX
    def custom_MF(x):
        tmp = 1.7159 * ((np.exp((4/3)*x) - 1)/(np.exp((4/3)*x) + 1))
        return(tmp)

    for i in range(1,n_list+1):
        b1 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_b1_F_{i}.txt", header=None)
        b2 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_b2_F_{i}.txt", header=None)
        b3 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_b3_F_{i}.txt", header=None)
        IW = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_IW_F_{i}.txt", sep="\s+", header=None)
        LW1 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_LW1_F_{i}.txt", sep="\s+", header=None)
        LW2 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_LW2_F_{i}.txt", sep="\s+", header=None)
        b1 = np.array(b1)
        b2 = np.array(b2)
        b3 = np.array(b3)

         # Calculate a
        a = custom_MF(np.dot(data_N, IW.T).T + b1)

        # Calculate b
        b = custom_MF(np.dot(LW1, a) + b2)
        
        # Calculate y
        y = np.dot(LW2, b) + b3
        y = y.T
        
        # Calculate ph_outputs
        ph_outputs = 1.5 * y * std_F[0][ne] + moy_F[0][ne]
        ph_outputs_s.append(ph_outputs)
    
    # reshape array
    ph_outputs_s1 = np.array(ph_outputs_s).T

    # load weights
    moy_G = np.array(pd.read_table(basedir + "CANYON-MED_weights/moy_ph_G.txt", sep=" {3}", header=None, engine = 'python'))
    std_G = np.array(pd.read_table(basedir + "CANYON-MED_weights/std_ph_G.txt", sep=" {3}", header=None, engine = 'python'))

    # NORMALISATION OF THE PARAMETERS
    data_N = df.iloc[:, :ne].copy()

    for i in range(ne):
        data_N.iloc[:, i] = (2 / 3) * ((df.iloc[:, i] - moy_G[:, i]) / std_G[:, i])

    data_N = np.array(data_N)
    
    ph_outputs_s = []
    
    for i in range(1,n_list+1):
        b1 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_b1_G_{i}.txt", header=None)
        b2 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_b2_G_{i}.txt", header=None)
        b3 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_b3_G_{i}.txt", header=None)
        IW = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_IW_G_{i}.txt", sep="\s+", header=None)
        LW1 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_LW1_G_{i}.txt", sep="\s+", header=None)
        LW2 = pd.read_csv(basedir + f"CANYON-MED_weights/poids_ph_LW2_G_{i}.txt", sep="\s+", header=None)
        b1 = np.array(b1)
        b2 = np.array(b2)
        b3 = np.array(b3)

         # Calculate a
        a = custom_MF(np.dot(data_N, IW.T).T + b1)

        # Calculate b
        b = custom_MF(np.dot(LW1, a) + b2)
        
        # Calculate y
        y = np.dot(LW2, b) + b3
        y = y.T
        
        # Calculate ph_outputs
        ph_outputs = 1.5 * y * std_G[0][ne] + moy_G[0][ne]
        ph_outputs_s.append(ph_outputs)
    
    # flatten array because it's nice #2
    ph_outputs_s2 = np.array(ph_outputs_s).T
    
    # concat F and G data
    ph_outputs_s = np.hstack((np.squeeze(ph_outputs_s1, axis = 0), np.squeeze(ph_outputs_s2, axis = 0)))
    
    # neural network
    mean_nn = np.mean(ph_outputs_s, axis=1)
    std_nn = np.std(ph_outputs_s, axis=1, ddof = 1)

    lim_inf = mean_nn - std_nn
    lim_sup = mean_nn + std_nn

    ph_t = ph_outputs_s.copy()
    
    for i in range(ph_outputs_s.shape[0]):
        ph_t[i,:] = np.where(ph_t[i,:]<lim_inf[i], np.nan, ph_t[i,:])
        ph_t[i,:] = np.where(ph_t[i,:]>lim_sup[i], np.nan, ph_t[i,:])

    ph_out = np.nanmean(ph_t, axis=1)

    out = ph_out
    return out
