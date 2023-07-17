import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import genextreme

def dataset_with_maximas_over_week(dataset) :
    """
    returns a xarray with coordinates : rlat, rlon and week; with variable : max_pr_week
    """
    weekly_data = dataset.resample(time='1W').max(dim='time')
    tot_pr_weekly = weekly_data['TOT_PR']
    max_pr_weekly = tot_pr_weekly.groupby('time.week').max(dim='time')
    year = dataset.time.dt.year
    result = xr.Dataset({'max_pr_week': max_pr_weekly})
    result = result.assign_coords(year=year)  # Ajouter l'année en tant que coordonnée
    print(result)
    #return result

def dataset_with_maximas_over_month(dataset):
    """
    returns a xarray with coordinates: rlat, rlon and month; with variable: max_pr_month
    """
    monthly_data = dataset.resample(time='1M').max(dim='time')
    tot_pr_monthly = monthly_data['TOT_PR']
    max_pr_monthly = tot_pr_monthly.groupby('time.month').max(dim='time')
    result = xr.Dataset({'max_pr_month': max_pr_monthly})
    return result


def maxima_with_week_and_year(dataset):
    group_data=dataset.groupby('year')
    list_year_data=[]
    for year, year_data in group_data:
        weekly_data = year_data.resample(time='1W').max(dim='time') ##j'ai 5 semaines et le max par semaine !! il faut remplacer time par week
        tot_pr=weekly_data['TOT_PR']
        max_pr_week = tot_pr.groupby('time.week').max(dim='time')
        max_pr_week = max_pr_week.assign_coords(year=year)  # Ajouter l'année en tant que coordonnée
        list_year_data.append(max_pr_week)
    return xr.concat(list_year_data,dim='week')


def maxima_with_week_and_year_temp(dataset):
    group_data=dataset.groupby('year')
    list_year_data=[]
    for year, year_data in group_data:
        weekly_data = year_data.resample(time='1W').max(dim='time') ##j'ai 5 semaines et le max par semaine !! il faut remplacer time par week
        tot_pr=weekly_data['T_2M']
        max_pr_week = tot_pr.groupby('time.week').max(dim='time')
        max_pr_week = max_pr_week.assign_coords(year=year)  # Ajouter l'année en tant que coordonnée
        list_year_data.append(max_pr_week)
    return xr.concat(list_year_data,dim='week')


def gev_parameters(dataset):
    """
    returns an xarray with, for each rlon and rlat : the three parameters of the GEV for a dataset that contains 'max_pr_week' variable
    """
    # Dimensions du xarray
    rlon_dim = len(dataset['rlon'])
    rlat_dim = len(dataset['rlat'])
    
    # Résultats
    shape = np.zeros((rlat_dim, rlon_dim))
    loc = np.zeros((rlat_dim, rlon_dim))
    scale = np.zeros((rlat_dim, rlon_dim))
    
    # Boucle for pour calculer les paramètres de la GEV pour chaque point
    for i in range(rlat_dim):
        for j in range(rlon_dim):
            data = dataset['max_pr_week'][:, i, j].values
            shape[i, j], loc[i, j], scale[i, j] = genextreme.fit(data)
    
    # Création d'un nouvel xarray pour stocker les résultats
    gev_params_2 = xr.Dataset(
        {
            'shape': (['rlat', 'rlon'], shape),
            'location': (['rlat', 'rlon'], loc),
            'scale': (['rlat', 'rlon'], scale),
        },
        coords={'rlat': dataset['rlat'], 'rlon': dataset['rlon']}
    )
    return gev_params_2



def gev_parameters_for_week(dataset):
    """
    returns an xarray with, for each rlon and rlat : the three parameters of the GEV for a dataset that contains 'max_pr_week' variable
    """
    # Dimensions du xarray
    rlon_dim = len(dataset['rlon'])
    rlat_dim = len(dataset['rlat'])
    
    # Résultats
    shape = np.zeros((rlat_dim, rlon_dim))
    loc = np.zeros((rlat_dim, rlon_dim))
    scale = np.zeros((rlat_dim, rlon_dim))
    
    # Boucle for pour calculer les paramètres de la GEV pour chaque point
    for i in range(rlat_dim):
        for j in range(rlon_dim):
            data = dataset['max_pr_year'][:, i, j].values
            shape[i, j], loc[i, j], scale[i, j] = genextreme.fit(data)
    
    # Création d'un nouvel xarray pour stocker les résultats
    gev_params_2 = xr.Dataset(
        {
            'shape': (['rlat', 'rlon'], shape),
            'location': (['rlat', 'rlon'], loc),
            'scale': (['rlat', 'rlon'], scale),
        },
        coords={'rlat': dataset['rlat'], 'rlon': dataset['rlon']}
    )
    return gev_params_2