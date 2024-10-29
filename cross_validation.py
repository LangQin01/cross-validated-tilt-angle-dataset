# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:44:18 2024

@author: 1
"""

import numpy as np
import pandas as pd
from sunpy.coordinates.sun import carrington_rotation_number


# import original datasets. "dpd" for DPD dataset, "rh" for WJL dataset
data_dpd = np.loadtxt('C:/Users/1/Desktop/Quenching/code/dpd_ldcm_nosep.txt',dtype='float')
data_rh = np.loadtxt('C:/Users/1/Desktop/Quenching/code/RH_nosep_labeled.txt',dtype='float')

# sort by date
order = np.argsort(data_rh[:,0])
data_rh = data_rh[order,:]
# add a column for NOAA number
data_rh = np.insert(data_rh,14,np.zeros(data_rh.shape[0]).T,axis=1)
# add two columns for Carrington rotation number and marker of WJL dataset
data_dpd = np.insert(data_dpd,12,np.zeros(data_dpd.shape[0]).T,axis=1)
data_dpd = np.insert(data_dpd,13,np.zeros(data_dpd.shape[0]).T,axis=1)

# caculate the Carrington rotation number using Sunpy package
for i in range(data_dpd.shape[0]):
    date = str(data_dpd[i,0])[0:4]+'-'+str(data_dpd[i,0])[4:6]+'-'+str(data_dpd[i,0])[6:8]
    data_dpd[i,13] = int(str(carrington_rotation_number(date))[0:4])

# 1: with counterpart, 2: without counterpart
data_dpd1 = np.zeros((data_dpd.shape[0],data_dpd.shape[1]))
data_rh1 = np.zeros((data_rh.shape[0],data_rh.shape[1]))

# matching counterparts by Carrington rotation,latitude and longitude
# set an area threshold of 8 units
k1 = 0   
coord2 = np.ones(data_dpd.shape[0])
for i in range(data_dpd.shape[0]):
    for j in range(data_rh.shape[0]):
        if data_dpd[i,13]==data_rh[j,7] and abs(data_dpd[i,1]-data_rh[j,1])<5 \
            and abs(data_dpd[i,2]-data_rh[j,2])<10 and data_dpd[i,3] > 8:
            data_dpd1[k1,:] = data_dpd[i,:]
            coord2[i] = 0
            data_rh1[k1,0:15] = data_rh[j,:]
            data_rh1[k1,14] = data_dpd[i,7] # assign NOAA number to WJL dataset
            data_rh[j,14] = data_dpd[i,7]
            data_dpd[i,12] = data_rh[j,13] # assign marker to DPD dataset
            data_dpd1[k1,12] = data_rh[j,13]
            k1 += 1

# delete zeros        
data_dpd1 = data_dpd1[0:k1,:]
data_rh1 = data_rh1[0:k1,:]


# data without counterparts
coord2 = np.array(coord2,dtype=bool)
data_dpd2 = data_dpd[coord2]
data_rh2 = data_rh[data_rh[:,14]==0] # no NOAA number = no counterpart

# exclude repeated ARs
coord = []
for i in range(data_dpd1.shape[0]):
    if (data_dpd1[i,2] == data_dpd1[i-1,2]) \
        or (data_rh1[i,2] == data_rh1[i-1,2]):
        coord.append(i)
 
data_dpd1 = np.delete(data_dpd1,coord,axis=0)
data_rh1 = np.delete(data_rh1,coord,axis=0)       

coord = []
for i in range(data_dpd1.shape[0]):
    if (data_dpd1[i,2] == data_dpd1[i-1,2]) \
        or (data_rh1[i,2] == data_rh1[i-1,2]):
        coord.append(i)

data_dpd1 = np.delete(data_dpd1,coord,axis=0)
data_rh1 = np.delete(data_rh1,coord,axis=0)

for k in range(1913,2199):
    coord1 = data_rh1[:,7] == k
    coord2 = data_dpd[:,13] == k  
    
    if np.sum(coord1) == 0 or np.sum(coord2) == 0:
        continue
    
    lat = data_dpd[coord2,1]
    lon = data_dpd[coord2,2]
    lat_index = (np.sin(lat/180*np.pi)+np.sin(np.pi/3))/3**0.5*1248
    lon_index = lon/360*3600

    sspot = np.zeros((1248, 3600))
    a1=lat_index.astype('int')
    a2=lon_index.astype('int')
    for i in range(len(a1)):
        sspot[a1[i]-5:a1[i]+5,a2[i]-5:a2[i]+5]=1
    
    path = 'C:/Users/1/Desktop/Quenching/code/img label/CR '+str(k)+'.npy' 
    img_label = np.load(path)
    
    print(k)
    
    for l in data_rh1[data_rh1[:,7]==k,13]:
        ar = np.zeros((1248,3600))
        ar[img_label == l] = 1
        num1 = np.sum(ar)
        ar2 = ar-sspot
        ar2[ar2 == -1] = 0 
        num2 = np.sum(ar-ar2)
        if num2 > 100:
            coord = (data_rh1[:,7]==k)*(data_rh1[:,13]==l)
            data_rh1 = np.delete(data_rh1,coord,axis=0)
            data_dpd1 = np.delete(data_dpd1,coord,axis=0)


# check the consistency of two tilt values
n = 181
sigma = np.zeros(n)
mtilt_rh = np.zeros(n)
mtilt_dpd = np.zeros(n)
x_tilt = np.zeros(n)
coord_within = []
coord_out = []

for i in range(n):
    coord=[]
    for j in range(data_rh1.shape[0]):
        if data_rh1[j,4]>=-90+i*1-10 and data_rh1[j,4]<=-90+i*1+10:
            coord.append(j)
    
    x_tilt[i] = -90+1*i 
    mtilt_rh[i] = np.mean(data_rh1[coord,4])
    mtilt_dpd[i] = np.mean(data_dpd1[coord,5])
    sigma[i] = np.std(data_dpd1[coord,5])
    
for i in range(data_dpd1.shape[0]):
    for j in range(len(x_tilt)):
        if abs(data_rh1[i,4]-x_tilt[j])<1.35:
            if abs(data_dpd1[i,5]-x_tilt[j]) < 1*sigma[j]:
                coord_within.append(i)
            else:
                coord_out.append(i)

# exclude repeated data
coord_within = list(set(coord_within))
coord_out = list(set(coord_out))

for k in coord_within:
    if k in coord_out:
        coord_out.remove(k)


# cross-validated tilt angle dataset
date_dpd = data_dpd1[:,0]
lat_dpd = data_dpd1[:,1]
lon_dpd = data_dpd1[:,2]
llat_dpd = data_dpd1[:,8]
llon_dpd = data_dpd1[:,9]
flat_dpd = data_dpd1[:,10]
flon_dpd = data_dpd1[:,11]
area_dpd = data_dpd1[:,3]
tilt_dpd = data_dpd1[:,5]

date_wjl = data_rh1[:,0]
lat_wjl = data_rh1[:,1]
lon_wjl = data_rh1[:,2]
Plat_wjl = data_rh1[:,8]
Plon_wjl = data_rh1[:,9]
Nlat_wjl = data_rh1[:,10]
Nlon_wjl = data_rh1[:,11]
area_wjl = data_rh1[:,3]
flux_wjl = data_rh1[:,6]
tilt_wjl = data_rh1[:,4]

NOAA = data_rh1[:,14]
CR = data_rh1[:,7]

marker = np.zeros(data_dpd1.shape[0])
marker[coord_within] = 1

table = np.vstack((date_dpd,lat_dpd,lon_dpd,area_dpd,tilt_dpd,date_wjl,lat_wjl,
                   lon_wjl,area_wjl,flux_wjl,tilt_wjl,NOAA,CR,marker))
table = table.T

columns = ['date_dpd','lat_dpd','lon_dpd','area_dpd','tilt_dpd','date_wjl',
           'lat_wjl','lon_wjl','area_wjl','flux_wjl','tilt_wjl','number','CR','consistency']

# the cross_validated dataset
df = pd.DataFrame(table,columns = columns)

df.to_excel("cross_validated_tilt_angle_dataset.xlsx",index=False)












