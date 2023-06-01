import numpy as np
import pandas as pd
import math

def CANYON_MED_PO4_v4(date, lat, lon, pres, temp, psal, doxy):
    # Multi-layer perceptron to predict alkalinity / umol kg-1
    #
    # Neural network training by Marine Fourrier from work by Raphaelle Sauzede, LOV;
    # as R function by Marine Fourrier, LOV
    #
    #
    # input:
    # date  - date (UTC) as string ("yyyy-mm-dd HH:MM")
    # lat   - latitude / °N  [-90 90]
    # lon   - longitude / °E [-180 180] or [0 360]
    # pres  - pressure / dbar
    # temp  - in-situ temperature / °C
    # psal  - salinity
    # doxy  - dissolved oxygen / umol kg-1
    #
    # output:
    # out   - alkalinity / umol kg-1
    #
	# check value:  0.3029 umol kg-1
	# for 09-Apr-2014, 35° N, 18° E, 500 dbar, 13.5 °C, 38.6 psu, 160 umol O2 kg-1
    #
    #
    # Marine Fourrier, LOV
    # 01.06.2023

    # No input checks! Assumes informed use, e.g., same dimensions for all
    # inputs, ...
  
    basedir = "D:/Documents/Thèse/Docs/science/PAPIER_CANYON_MED/CODES/CANYON-MED/v2/PYTHON/"  # relative or absolute path to CANYON-MED folder

    # input preparation
    date = pd.to_datetime(date)
    dec_year = date.year + (date - pd.Timestamp(f"{date.year}-01-01 00:00")).days / 365
    lon[lon > 180] = lon[lon > 180] - 360

    # input sequence:
    #     lat,   lon,    dec_year,    temp,   sal,    oxygen, P

    data = pd.DataFrame({"lat": lat, "lon": lon, "dec_year": dec_year, "temp": temp, "psal": psal, "doxy": doxy, "pres": pres / 2e4 + 1 / ((1 + np.exp(-pres / 300)) ** 3)})

    moy_F = pd.read_table(basedir + "CANYON-MED_weights/moy_phos_F.txt", sep="")
    std_F = pd.read_table(basedir + "CANYON-MED_weights/std_phos_F.txt", sep="")

    ne = 7  # Number of inputs

    # NORMALISATION OF THE PARAMETERS
    data_N = data.iloc[:, :ne].copy()

    for i in range(ne):
        data_N.iloc[:, i] = (2 / 3) * ((data.iloc[:, i] - moy_F.iloc[:, i]) / std_F.iloc[:, i])

    data_N = np.array(data_N)

    n_list = 5

    phos_outputs_s = pd.DataFrame(np.full((data.shape[0], 10), np.nan))

    rx = data_N.shape[0]
    for i in range(n_list):
        b1 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_b1_F_{i}.txt", sep="")
        b2 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_b2_F_{i}.txt", sep="")
        b3 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_b3_F_{i}.txt", sep="")
        IW = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_IW_F_{i}.txt", sep="")
        LW1 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_LW1_F_{i}.txt", sep="")
        LW2 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_LW2_F_{i}.txt", sep="")
        b1 = np.array(b1)
        b2 = np.array(b2)
        b3 = np.array(b3)

        # hidden layers
        a = 1.715905 * np.tanh((2. / 3) * (data_N @ IW.T + b1 @ np.ones((rx, 1)).T))
        b = 1.715905 * np.tanh((2. / 3) * (a @ LW1.T + b2 @ np.ones((rx, 1)).T))
        y = b @ LW2.T + b3 @ np.ones((rx, 1)).T
        phos_outputs = 1.5 * y * std_F.iloc[0, ne + 1] + moy_F.iloc[0, ne + 1]

        phos_outputs_s[i] = phos_outputs

    moy_G = pd.read_table(basedir + "CANYON-MED_weights/moy_phos_G.txt", sep="")
    std_G = pd.read_table(basedir + "CANYON-MED_weights/std_phos_G.txt", sep="")

    ne = 7  # Number of inputs

    # NORMALISATION OF THE PARAMETERS
    data_N = data.iloc[:, :ne].copy()

    for i in range(ne):
        data_N.iloc[:, i] = (2 / 3) * ((data.iloc[:, i] - moy_G.iloc[:, i]) / std_G.iloc[:, i])

    data_N = np.array(data_N)

    rx = data_N.shape[0]
    for i in range(n_list):
        b1 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_b1_G_{i}.txt", sep="")
        b2 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_b2_G_{i}.txt", sep="")
        b3 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_b3_G_{i}.txt", sep="")
        IW = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_IW_G_{i}.txt", sep="")
        LW1 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_LW1_G_{i}.txt", sep="")
        LW2 = pd.read_table(basedir + f"CANYON-MED_weights/poids_phos_LW2_G_{i}.txt", sep="")
        b1 = np.array(b1)
        b2 = np.array(b2)
        b3 = np.array(b3)

        # hidden layers
        a = 1.715905 * np.tanh((2. / 3) * (data_N @ IW.T + b1 @ np.ones((rx, 1)).T))
        b = 1.715905 * np.tanh((2. / 3) * (a @ LW1.T + b2 @ np.ones((rx, 1)).T))
        y = b @ LW2.T + b3 @ np.ones((rx, 1)).T
        phos_outputs = 1.5 * y * std_G.iloc[0, ne + 1] + moy_G.iloc[0, ne + 1]

        phos_outputs_s[i + 5] = phos_outputs

    mean_nn = np.mean(phos_outputs_s, axis=0)
    std_nn = np.std(phos_outputs_s, axis=0)

    lim_inf = mean_nn - std_nn
    lim_sup = mean_nn + std_nn

    phos_t = phos_outputs_s.copy()
    phos_t[(phos_t < lim_inf) | (phos_t > lim_sup)] = np.nan

    phos_out = np.nanmean(phos_t, axis=0)

    out = phos_out
    return out
