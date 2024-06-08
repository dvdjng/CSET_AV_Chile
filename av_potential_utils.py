# -*- coding: utf-8 -*-
"""
Created on May 08, 2023

@author: David Jung
"""

import warnings
warnings.filterwarnings("ignore")
import sys
import os
import pandas as pd
import numpy as np
import geopandas as gpd
from geopandas import GeoSeries
import requests


import plotly.express as px
import time 
import pytz
import datetime

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import pvlib
from pvlib import location
from pvlib import tracking
from pvlib.bifacial.pvfactors import pvfactors_timeseries
from pvlib import temperature
from pvlib import pvsystem

import pyet

from tqdm import tqdm




sys.path.append('../lib/')


### DATA BASE ###
def get_area_shp(shp,unit,column_name="area"):
    
    if unit == "m2":
        x = 1
    elif unit == "ha":
        x = 10000
    elif unit == "km2":
        x = 1000*1000
    else:
        print("wrong unit, choose 'm2', 'ha' or 'km2'")
        
    crs = shp.crs
    shp.to_crs(epsg = 32719)
    shp[column_name] = (shp.geometry.area / x).round(2)
    shp.to_crs(crs)


def get_TMY(lat_4326, long_4326, tz='America/Santiago'):
    
    try: 
        # API request            
        data_pvgis  = pvlib.iotools.get_pvgis_tmy(lat_4326, long_4326, outputformat='json', usehorizon=True, userhorizon=None, startyear=None, endyear=None, url='https://re.jrc.ec.europa.eu/api/v5_2/', map_variables=None, timeout=20)
        
        # Get altitude and tmy data
        tmy_pvg_r = data_pvgis[0]
        altitude = data_pvgis[2].get("location").get("elevation")

        # adapt to local timezone
        timezone = pytz.timezone(tz)
        dt = datetime.datetime.utcnow()
        offset_seconds = timezone.utcoffset(dt).seconds
        offset_hours = offset_seconds / 3600.0
        if offset_hours > 12:
            offset_hours = offset_hours - 24
        
        if offset_hours < 0:
            offset_hours = int(-offset_hours)
            # Extract the first rows accordin to the time offset
            first_rows = tmy_pvg_r.head(offset_hours)
            tmy_pvg_r  = tmy_pvg_r.drop(tmy_pvg_r.index[:offset_hours])
            # Concatenate the modified DataFrame and the first three rows
            tmy_pvg = pd.concat([tmy_pvg_r, first_rows])

            tmy_pvg["time"] = pd.date_range(start = "2022-01-01 00:00", end="2022-12-31 23:00", freq="h", tz=tz)   
        else:
            print("TMY download only for South America")
   
        # Information on PVGIS Download
        if (tmy_pvg["G(h)"].sum() < 1):
            download_info = "missing solar data"
        elif (tmy_pvg["WS10m"].sum() < 1) | (tmy_pvg["T2m"].sum() < 1):
            download_info = "missing climate data"
        else:
            download_info = "ok"

        # process tmy data: Rename for pvlib
        cols_to_use = ["time", "T2m", "G(h)", "Gb(n)", "Gd(h)", "IR(h)", "WS10m", "RH", "SP"] 
        pvlib_column_names = ["time", "temp_air", "ghi", "dni", "dhi", "lwr_u", "wind_speed", "rh", "sp" ] 
        tmy_pvg = tmy_pvg[cols_to_use]
        tmy_pvg.columns = pvlib_column_names

        location = Location(lat_4326, long_4326, tz, altitude)
        # Get solar azimuth and zenith to store in tmy
        solar_position = location.get_solarposition(times=tmy_pvg.index)
        tmy_pvg["azimuth"] = solar_position["azimuth"] 
        tmy_pvg["zenith"] = solar_position["zenith"] 
        tmy_pvg["apparent_zenith"] = solar_position["apparent_zenith"] 
        tmy_pvg = tmy_pvg.reset_index(drop=True)
        
    except requests.HTTPError as err:
        download_info = err
        tmy_pvg = None
        altitude = None

    return tmy_pvg, altitude, download_info


def get_TMYs_from_gdf(gdf, tz='America/Santiago', directory="data\\PVGIS_TMY\\", tmy_all=None):
    
    if tmy_all is None:
        tmy_all = []
    gdf = gdf.to_crs(epsg = 4326)
    gdf["lat_4326"] = gdf.geometry.y
    gdf["long_4326"] = gdf.geometry.x
    
    for i in tqdm(gdf.index):
        id = i
        lat_4326 = gdf.loc[i,"lat_4326"]
        long_4326 = gdf.loc[i,"long_4326"]
        tmy_pvg, altitude, download_info = get_TMY(lat_4326, long_4326, tz=tz)
        if download_info == "ok":
            gdf.loc[i,"altitude"] = altitude
            gdf.loc[i,"PVGIS_dl"] = download_info
            gdf.loc[i,"GHI_KWh/a"] = tmy_pvg["ghi"].sum()/1000

            # add data info
            tmy_pvg["info"] = np.nan
            tmy_pvg["info_values"] = np.nan
            info = gdf.columns.values.tolist()
            for j in range(len(info)):
                tmy_pvg.loc[j,"info"] = info[j]
            
            info_values = gdf.loc[i].tolist()
            for j in range(len(info_values)):
                tmy_pvg.loc[j,"info_values"] = info_values[j]

            # Save as csv
            outFileName = "AV_Potential_id_" + str(id)
            tmy_pvg.to_csv(directory+outFileName+".csv", sep=',',encoding='latin-1', index=False)

            # Store information in df
            tmy_all.append(tmy_pvg)
        else:
            gdf.loc[i,"GHI_KWh/a"] = None
            gdf.loc[i,"altitude"] = altitude
            gdf.loc[i,"PVGIS_dl"] = download_info


    print("PVGIS Download finished: "+ str(gdf[gdf['PVGIS_dl'] != "ok"].count().area)+ " locations without (complete) TMY data")
    gdf.to_csv("data_used_for_PVGIS_dl.csv", sep=',',encoding='latin-1', index=False)
    
    return gdf, tmy_all

    """"
    def get_elevation(lat, long):
        
        Get elevation data for a specific latitude and longitude coordinate from open-elevation.com.

        Args:
        - lat (float): Latitude coordinate.
        - long (float): Longitude coordinate.

        Returns:
        - elevation (float): Elevation value in meters.
        
        query = ('https://api.open-elevation.com/api/v1/lookup'
                f'?locations={lat},{long}')
        r = requests.get(query).json()  # json object, various ways you can extract value
        elevation = pd.json_normalize(r, 'results')['elevation'].values[0]
        return elevation

    # Apply function to get elevation (DO NOT APPLY TO A LARGE DATASET FOR TESTING)
    for i in range(0,len(gdf)):
        long = gdf.loc[i,"geometry"].x
        lat = gdf.loc[i,"geometry"].y
        
        gdf.loc[i,"altitude"] = get_elevation(lat, long)

    # Plot shapefile of Chilean regions and dataset with altitude values
    ax = gdf_cl.plot(figsize=(10, 10), color="white", edgecolor="lightgrey")
    gdf.plot(ax=ax, column='altitude', legend=True)

    plt.title("Altitude of points in the dataset")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    """

### TECHNO ECONOMIC SIMULATION ###
def calc_GCR(track, pvrow_azimuth, pvrow_tilt, n_pvrows, pvrow_width, pvrow_pitch, pvrow_height, tmy_data, albedo): # pvrow_tilt with tracking == True is equal to max tilt
    
    #Definition of PV array
    gcr = pvrow_width / pvrow_pitch
    axis_azimuth = pvrow_azimuth + 90
    
    pvarray_parameters = {
        'n_pvrows': n_pvrows,
        'axis_azimuth': axis_azimuth,
        'pvrow_height': pvrow_height,
        'pvrow_width': pvrow_width,
        'gcr': gcr
    }

    # Create an ordered PV array
    pvarray = OrderedPVArray.init_from_dict(pvarray_parameters) # ground is not initalized: https://github.com/SunPower/pvfactors/blob/master/pvfactors/geometry/pvarray.py#L12

    if track == True:
        # pv-tracking algorithm to get pv-tilt
        orientation = tracking.singleaxis(tmy_data['apparent_zenith'],
                                        tmy_data['azimuth'],
                                        max_angle=pvrow_tilt,
                                        backtrack=True,
                                        gcr=gcr
                                        )
        tmy_data['surface_azimuth'] = orientation['surface_azimuth']
        tmy_data['surface_tilt'] = orientation['surface_tilt'] 
    else:
        tmy_data['surface_azimuth'] = np.where((tmy_data["apparent_zenith"] > 0 ) & (tmy_data["apparent_zenith"] < 90), pvrow_azimuth,np.nan)
        tmy_data['surface_tilt'] = np.where((tmy_data["apparent_zenith"] > 0 ) & (tmy_data["apparent_zenith"] < 90), pvrow_tilt,np.nan)
    
    # Create engine using the PV array
    engine = PVEngine(pvarray) 

    # Fit engine to data: which will update the pvarray object as well
    engine.fit(tmy_data.index, tmy_data.dni, tmy_data.dhi,
            tmy_data.zenith, tmy_data.azimuth,
            tmy_data.surface_tilt, tmy_data.surface_azimuth,
            albedo= albedo)


    a = pd.DataFrame()
    for i in range(0,len(pvarray.ts_ground.all_ts_surfaces)):
        a[str(i)+"_0,0"] = pvarray.ts_ground.all_ts_surfaces[i].coords.as_array[0][0]
        a[str(i)+"_1,0"] = pvarray.ts_ground.all_ts_surfaces[i].coords.as_array[1][0]

    # set x_min and x_max so that only area under PV array is considered (between second and second to last row)
    a[a < pvrow_pitch] = pvrow_pitch
    a[a > pvrow_pitch * (n_pvrows -2)] = pvrow_pitch * (n_pvrows -2)

    # sum up the shadow and ilum lenghts
    for i in range(0,len(pvarray.ts_ground.all_ts_surfaces)):
        a[str(i)] = a[str(i)+"_1,0"] - a[str(i)+"_0,0"]

    print("shadow ratio is calculated between x = "+str(pvrow_pitch)+" m and "+str(pvrow_pitch * (n_pvrows -2))+" m")

    shadow = 0
    for i in range(0,pvarray.ts_ground.n_ts_shaded_surfaces):
        shadow += a[str(i)]

    light = 0
    for i in range(pvarray.ts_ground.n_ts_shaded_surfaces,len(pvarray.ts_ground.all_ts_surfaces)):
        light += a[str(i)]


    sl = pd.DataFrame()
    sl["lenght_shadow"] = shadow
    sl["lenght_ilum"] = light
    sl["sum"] = sl["lenght_ilum"]+sl["lenght_shadow"]
    sl["shadow_ratio"] = 1 / sl["sum"] * sl["lenght_shadow"]
    return sl, pvarray



def calc_PV(tmy_data, albedo, track, pvrow_azimuth, pvrow_tilt, n_pvrows, pvrow_width, pvrow_pitch, pvrow_height, bifaciality, losses= None): # pvrow_tilt with tracking == True is equal to max tilt
    
    #Definition of PV array
    gcr = pvrow_width / pvrow_pitch
    gcr = 0.2
    if pvrow_azimuth > 269:
        axis_azimuth = pvrow_azimuth + 90 - 360
    else:
        axis_azimuth = pvrow_azimuth + 90
    
    pvarray_parameters = {
        'n_pvrows': n_pvrows,
        'axis_azimuth': axis_azimuth,
        'pvrow_height': pvrow_height,
        'pvrow_width': pvrow_width,
        'gcr': gcr
    }

    # Create an ordered PV array
    #pvarray = OrderedPVArray.init_from_dict(pvarray_parameters) # ground is not initalized: https://github.com/SunPower/pvfactors/blob/master/pvfactors/geometry/pvarray.py#L12

    if track == True:
        # pv-tracking algorithm to get pv-tilt
      
        orientation = tracking.singleaxis(
            apparent_zenith=tmy_data['apparent_zenith'],
            apparent_azimuth=tmy_data['azimuth'],
            axis_tilt=0,
            axis_azimuth=axis_azimuth,
            max_angle=pvrow_tilt,
            backtrack=True,  
            gcr=gcr)  

        tmy_data['surface_azimuth'] = orientation['surface_azimuth']
        tmy_data['surface_tilt'] = orientation['surface_tilt'] 
    else:
        tmy_data['surface_azimuth'] = np.where((tmy_data["apparent_zenith"] > 0 ) & (tmy_data["apparent_zenith"] < 90), pvrow_azimuth,np.nan)
        tmy_data['surface_tilt'] = np.where((tmy_data["apparent_zenith"] > 0 ) & (tmy_data["apparent_zenith"] < 90), pvrow_tilt,np.nan)
    
    
    irrad = pvfactors_timeseries(tmy_data['azimuth'],
                             tmy_data['apparent_zenith'],
                             tmy_data['surface_azimuth'],
                             tmy_data['surface_tilt'],
                             axis_azimuth,
                             tmy_data.index,
                             tmy_data['dni'],
                             tmy_data['dhi'],
                             gcr,
                             pvrow_height,
                             pvrow_width,
                             albedo,
                             n_pvrows=n_pvrows,
                             index_observed_pvrow=2
                             )

    # turn into pandas DataFrame
    irrad = pd.concat(irrad, axis=1)

    # using bifaciality factor and pvfactors results, create effective irradiance
    effective_irrad_bifi = irrad['total_abs_front'] + (irrad['total_abs_back']
                                                    * bifaciality)

    # get cell temperature using the Faiman model    - Here heat coefficients could be implemented
    temp_cell = temperature.faiman(effective_irrad_bifi, temp_air=25,
                                wind_speed=1)

    # using the pvwatts_dc model and parameters detailed above,
    # set pdc0 and return DC power for both bifacial and monofacial
    pdc0 = 1000
    gamma_pdc = -0.0043
    pdc_bifi = pvsystem.pvwatts_dc(effective_irrad_bifi,
                                temp_cell,
                                pdc0,
                                gamma_pdc=gamma_pdc
                                ).fillna(0)
    
    pac0 = 1000
    results_ac = pvlib.inverter.pvwatts(
            pdc=pdc_bifi, 
            pdc0=pac0, 
            eta_inv_nom=0.961,
            eta_inv_ref=0.9637)
    
    if losses is None:
        # Standard losses
        losses = pvlib.pvsystem.pvwatts_losses(
                soiling=5, 
                shading=3, 
                snow=0, 
                mismatch=2, 
                wiring=2, 
                connections=0.5, 
                lid=1.5, 
                nameplate_rating=1, 
                age=0, 
                availability=3)

    results_ac_real = results_ac * (1-losses/100)

    return results_ac_real


def calc_ET(tmy, lat, ele, apv_shading):
    lat_rad = pyet.utils.check_lat(pyet.utils.deg_to_rad(lat))
    # Check if latitude is in correct format
    
    # resample climate data to daily values 
    tmy.index = tmy["time"]
    tmax = tmy["temp_air"].resample("D").max()
    tmin = tmy["temp_air"].resample("D").min()
    tmean = ( tmax + tmin ) / 2
    rhmax = tmy["rh"].resample("D").max()
    rhmin = tmy["rh"].resample("D").min()
    rh = tmy["rh"].resample("D").mean()
    wind = tmy["wind_speed"].resample("D").mean()
    rs = tmy["ghi"].resample("D").sum() * 0.0036 * (1-apv_shading)
    #rs_apv = tmy["ghi"].resample("D").sum() * 0.0036 * (1-apv_shading)

    # ET calculation with pyet
    ev_pm_fao56 = pyet.pm_fao56(tmean, wind=wind, rs=rs, tmax=tmax, tmin=tmin, rh=rh, rhmin=rhmin, rhmax=rhmax, elevation=ele, lat=lat_rad)
    #ev_pm_fao56_apv = pyet.pm_fao56(tmean, wind=wind, rs=rs_apv, tmax=tmax, tmin=tmin, rh=rh, rhmin=rhmin, rhmax=rhmax, elevation=ele, lat=lat_rad)

    tmy = tmy.reset_index(drop=True)

    return ev_pm_fao56


def calc_LCOE(E_G, CAPEX, OPEX, wacc, degre = 0.005, inflation = 0.03, N = 25):
    """
    Calculate the Levelized Cost of Electricity (LCOE) for a simulated PV system.

    Parameters:
    - E_G (float): Annual electricity generation (kWh).
    - CAPEX (float): Capital expenditure.
    - OPEX (float): Operational expenditure.
    - wacc (float): Weighted average cost of capital.
    - degre (float): Annual degradation rate (default: 0.005).
    - inflation (float): Annual inflation rate (default: 0.03).
    - N (int): Number of years for simulation (default: 25).

    Returns:
    - LCOE (float): Calculated Levelized Cost of Electricity (USD/kWh).
    """
    cashflow= pd.DataFrame(index=range(0,N))
    cashflow["year"] = range(1,N+1)
    cashflow["OPEX_des"] = (OPEX * (1+inflation)**cashflow.year) / (1+wacc)**cashflow.year
    cashflow["OPEX_des_infl"]= (OPEX * (1+inflation)**cashflow.year) / (1+inflation)**cashflow.year
    cashflow["EG_des"] = (E_G * (1-degre)**cashflow.year) / (1+wacc)**cashflow.year
    cashflow["EG_des_infl"] = (E_G * (1-degre)**cashflow.year) / (1+inflation)**cashflow.year
    LCOE =  (CAPEX + cashflow["OPEX_des"].sum() ) / cashflow["EG_des"].sum()

    return LCOE


### MCDM ###
"""
def calc_fuzzy

def calc_suitability
"""



### BACKUP ###

def clustering(gdf, num_cluster):
    """ Needs an geodataframe and the number of clusters as input
    
    [description]
    """
    
    gdf = gdf.to_crs('epsg:32719')
    gdf["longitude"] = gdf.geometry.centroid.x
    gdf["latitude"] = gdf.geometry.centroid.y
    gdf_kmeans = gdf[["latitude", "longitude"]]

    # Find clusters
    kmeans = KMeans(n_clusters=num_cluster, random_state=0, n_init=25)
    kmeans.fit_predict(gdf_kmeans)

    # Label cluster centers
    centers = kmeans.cluster_centers_

    # get distortion

    # Get cluster center
    gdf_kmeans["cluster"] = kmeans.labels_
    gdf["cluster"] = kmeans.labels_

    # Create Geo-Dataframe for Clusters
    df_cluster = pd.DataFrame(index=range(0,num_cluster))
    df_cluster["centers"] = centers.tolist()
    df_cluster["area"] = gdf.groupby('cluster')['area'].sum()
    for i in df_cluster.index:
        df_cluster.loc[i,"longitude"] = df_cluster.loc[i,"centers"][1]
        df_cluster.loc[i,"latitude"] = df_cluster.loc[i,"centers"][0]

    gdf_cluster = gpd.GeoDataFrame(df_cluster, geometry=gpd.points_from_xy(df_cluster["longitude"],df_cluster["latitude"]))
    gdf_cluster = gdf_cluster.set_crs('epsg:32719')
    #gdf_cluster = gdf_cluster.to_crs('epsg:32719')
    gdf_cluster = gdf_cluster[["area","longitude","latitude","geometry"]]

    cluster_distance = gdf_cluster.sindex.nearest(gdf.geometry.centroid, return_distance = True, return_all = False)
    gdf["dist"] = cluster_distance[1]
    gdf["clust"] = cluster_distance[0][1]

    gdf_cluster["cluster_id"] = gdf_cluster.index
    gdf_cluster = gdf_cluster.to_crs(epsg = 4326)
    gdf_cluster["long_4326"] = gdf_cluster.geometry.x
    gdf_cluster["lat_4326"] = gdf_cluster.geometry.y
    gdf_cluster = gdf_cluster.to_crs(epsg = 32719)
    gdf_cluster= gdf_cluster[["cluster_id", "area", "longitude","latitude", "long_4326", "lat_4326","geometry"]]

    plot_geolocation_by_cluster(gdf_kmeans, cluster='cluster', 
                            title= f'K-Means: Fruticulture locations grouped into {i+1} clusters',
                            centers=centers)

    return gdf, gdf_cluster


def plot_geolocation_by_cluster(df, 
                                cluster=None, 
                                title=None, 
                                centers=None,
                                filename=None):
    '''
    Function to plot latitude and longitude coordinates
    #####################
    Args:
        df: pandas dataframe 
            Contains id, latitude, longitude, and color (optional).
        cluster: (optional) column (string) in df 
            Separate coordinates into different clusters
        title: (optional) string
        centers: (optional) array of coordinates for centers of each cluster
        filename: (optional) string  
    #####################
    Returns:
        Plot with lat/long coordinates 
    '''
    
    # Transform df into geodataframe
    geo_df = gpd.GeoDataFrame(df.drop(['longitude', 'latitude'], axis=1),
                           crs={'init': 'epsg:32719'},
                           geometry=[Point(xy) for xy in zip(df.longitude, df.latitude)])
      
    # Set figure size
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    
    # Import NYC Neighborhood Shape Files
    #regions = gpd.read_file('./data_nyc/shapefiles/neighborhoods_nyc.shp')
    #nyc_full.plot(ax=ax, alpha=0.4, edgecolor='darkgrey', color='lightgrey', label=nyc_full['nta_name'], zorder=1)
    
    # Plot coordinates from geo_df on top of NYC map
    if cluster is not None:
        
        geo_df.plot(ax=ax, column=cluster, alpha=0.5, 
                    cmap='viridis', linewidth=0.8, zorder=2)
        
        if centers is not None:
            centers_gseries = GeoSeries(map(Point, zip(centers[:,1], centers[:,0])))
            centers_gseries.plot(ax=ax, alpha=1, marker='X', color='red', markersize=100, zorder=3)
        
        plt.title(title)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.show()
        
        if filename is not None:
            fig.savefig(f'{filename}', bbox_inches='tight', dpi=300)
    else:
        geo_df.plot(ax=ax, alpha=0.5, cmap='viridis', linewidth=0.8, legend=True, zorder=2)
        
        plt.title(title)
        plt.xlabel('longitude')
        plt.ylabel('latitude')
        plt.show()
        
    fig.clf()