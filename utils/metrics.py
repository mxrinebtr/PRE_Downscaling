import numpy as np
from matplotlib import pyplot as plt
import math
import xarray as xr
import skimage
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import seaborn as sns

def SSIM(ds1, ds2) :
    Coordinates = {
        'time':(['time'], ds1.time.values)
    } 
    
    T_2M = []
    TOT_PR = []
    RELHUM_2M = []
    for j in range(len(ds1.time)) :
        T_2M.append(ssim(ds1.T_2M.isel(time = j), ds2.T_2M.isel(time = j)))
        TOT_PR.append(ssim(ds1.TOT_PR.isel(time = j), ds2.TOT_PR.isel(time = j)))
        RELHUM_2M.append(ssim(ds1.RELHUM_2M.isel(time = j), ds2.RELHUM_2M.isel(time = j)))
                
    Variables = {
        'T_2M':(['time'], T_2M),
        'RELHUM_2M':(['time'], RELHUM_2M),
        'TOT_PR':(['time'], TOT_PR)
    }
    SSIM_ds = xr.Dataset(Variables, Coordinates)
    return SSIM_ds


def discrete_Hellinger(ds1, ds2) :
    Coordinates = {
        'time':(['time'], ds1.time.values)
    } 

    T_2M = []
    TOT_PR = []
    RELHUM_2M = []
    for k in range(len(ds1.time)) :
        Array_T = np.square(np.sqrt(ds1.T_2M.isel(time = k).values) - np.sqrt(ds2.T_2M.isel(time = k).values))
        Array_PR = np.square(np.sqrt(ds1.TOT_PR.isel(time = k).values) - np.sqrt(ds2.TOT_PR.isel(time = k).values))
        Array_HUM = np.square(np.sqrt(ds1.RELHUM_2M.isel(time = k).values) - np.sqrt(ds2.RELHUM_2M.isel(time = k).values))
        T_2M.append(math.sqrt(np.sum(Array_T))*1/math.sqrt(2))
        RELHUM_2M.append(math.sqrt(np.sum(Array_HUM))*1/math.sqrt(2))
        TOT_PR.append(math.sqrt(np.sum(Array_PR))*1/math.sqrt(2))
    Variables = {
        'T_2M':(['time'], T_2M),
        'RELHUM_2M':(['time'], RELHUM_2M),
        'TOT_PR':(['time'], TOT_PR)
    }
    H_ds = xr.Dataset(Variables, Coordinates)
    return H_ds


def Hellinger(ds1,ds2) :
    Coordinates = {
        'time':(['time'], ds1.time.values)
    } 
    T_2M = []
    TOT_PR = []
    RELHUM_2M = []
    bins_T = np.linspace(270,320,3000)
    bins_HUM = np.linspace(0,100,3000)
    bins_PR = np.linspace(0,0.06,10000)
    
    for k in range(len(ds1.time)) :
        pdf1, bin_out = np.histogram(ds1.T_2M.isel(time = k).values, bins_T, density = True) 
        pdf2, bin_out = np.histogram(ds2.T_2M.isel(time = k).values, bins_T, density = True)
        int_T = np.trapz(np.square(np.sqrt(pdf1) - np.sqrt(pdf2)),bins_T[1:])*1/2
        
        pdf1, bin_out = np.histogram(ds1.TOT_PR.isel(time = k).values, bins_PR, density = True) 
        pdf2, bin_out = np.histogram(ds2.TOT_PR.isel(time = k).values, bins_PR, density = True)
        int_PR = np.trapz(np.square(np.sqrt(pdf1) - np.sqrt(pdf2)),bins_PR[1:])*1/2
        
        pdf1, bin_out = np.histogram(ds1.RELHUM_2M.isel(time = k).values, bins_HUM, density = True) 
        pdf2, bin_out = np.histogram(ds2.RELHUM_2M.isel(time = k).values, bins_HUM, density = True)
        int_HUM = np.trapz(np.square(np.sqrt(pdf1) - np.sqrt(pdf2)),bins_HUM[1:])*1/2
        
        T_2M.append(math.sqrt(int_T))
        RELHUM_2M.append(math.sqrt(int_HUM))
        TOT_PR.append(math.sqrt(int_PR))
        
    Variables = {
        'T_2M':(['time'], T_2M),
        'RELHUM_2M':(['time'], RELHUM_2M),
        'TOT_PR':(['time'], TOT_PR)
    }
    H_ds = xr.Dataset(Variables, Coordinates)
    return H_ds


def Perkins(ds1, ds2) :
    Coordinates = {
        'time':(['time'], ds1.time.values)
    } 
        
    T_2M = []
    TOT_PR = []
    RELHUM_2M = []
    bins_T = np.linspace(270,320,3000)
    bins_HUM = np.linspace(0,100,3000)
    bins_PR = np.linspace(0,0.06,10000)
    
    for k in range(len(ds1.time)) :
        pdf1, bin_out = np.histogram(ds1.T_2M.isel(time = k).values, bins_T, density = True) 
        pdf2, bin_out = np.histogram(ds2.T_2M.isel(time = k).values, bins_T, density = True)
        T_2M.append(np.trapz(np.minimum(pdf1, pdf2), bins_T[1:]))
        
        pdf1, bin_out = np.histogram(ds1.TOT_PR.isel(time = k).values, bins_PR, density = True) 
        pdf2, bin_out = np.histogram(ds2.TOT_PR.isel(time = k).values, bins_PR, density = True)
        TOT_PR.append(np.trapz(np.minimum(pdf1, pdf2), bins_PR[1:]))

        pdf1, bin_out = np.histogram(ds1.RELHUM_2M.isel(time = k).values, bins_HUM, density = True) 
        pdf2, bin_out = np.histogram(ds2.RELHUM_2M.isel(time = k).values, bins_HUM, density = True)
        RELHUM_2M.append(np.trapz(np.minimum(pdf1, pdf2), bins_HUM[1:]))
        
    Variables = {
        'T_2M':(['time'], T_2M),
        'RELHUM_2M':(['time'], RELHUM_2M),
        'TOT_PR':(['time'], TOT_PR)
    }
    P_ds = xr.Dataset(Variables, Coordinates)
    return P_ds


def corr(ds, dim, lag) :
    if(dim == "time") :
        new_ds = np.array([ds.T_2M.values[i].flatten() for i in range(len(ds.T_2M))])
        corr = np.corrcoef(new_ds)
        corr_ds = []
        mat= []
        for i in range(lag+1) : 
            for j in range(len(new_ds)-lag):
                corr_ds.append(corr[j+i,j])
                mat.append(i)

    data = {'lag':  mat,
    'corr': corr_ds
    }

    df = pd.DataFrame(data)
    sns.lineplot(data=df, x="lag", y="corr")
    return 0


def multi_plot(ds_array, method, error) :# Here the ds array will have the first ds as the og image, and the other will be the downscaled images
    time = ds_array[0].time.values
    figure, axis = plt.subplots(1, 2)
    
    for i in range(len(ds_array)) :
        axis[0].plot(time,ds_array[i].T_2M.values, label = method[i])
    axis[0].set_title("Temperature error " + error)
    axis[0].legend()
    
    for i in range(len(ds_array)) :
        axis[1].plot(time,ds_array[i].RELHUM_2M.values, label = method[i])
    axis[1].set_title("Relative humidity " + error)
    axis[1].legend()
    
    return 0


def multi_scatterplot(ds_array, method, var1, var2):
    l = math.ceil(len(ds_array)/2)
    figure, axis = plt.subplots(l,2)
    for i in range(l) :
        ds_array[i*2].plot.scatter(var1, var2, marker='o', s = 0.7, alpha = 0.05, ax = axis[i,0])
        axis[i,0].set_title("Scatterplot of " + var1 + " and " + var2 +", "+ method[i])
        
        ds_array[i*2+1].plot.scatter(var1, var2, marker='o', s = 0.7, alpha = 0.05, ax = axis[i,1])
        axis[i,1].set_title("Scatterplot of " + var1 + " and " + var2 +", "+ method[i+1])
        
    return 0