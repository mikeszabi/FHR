# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:00:28 2020

@author: szabo
"""
import os
import glob
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

N=1200

base_dir=r'D:\DATA\FHR'
data_dir=os.path.join(base_dir,'Zoli-20200327T075837Z-001','Zoli','0')

measure_data_files=glob.glob(os.path.join(data_dir,'*.npy'))


for i_file in range(len(measure_data_files)):
    measure_data=np.load(measure_data_files[i_file])
    
    
    ######
    #
    #mdf=pd.DataFrame(measure_data[:,:1200].copy())
    #roll_mdf=mdf.rolling(axis=1,window=20,center=True)
    #rolling_max = roll_mdf.max().abs()
    #
    #mdf[rolling_max>0.9]=np.NaN
    #mdf.iloc[:,:10]=np.NaN
    #mdf.iloc[:,-10:]=np.NaN
    #
    #plt.plot(mdf.iloc[6,:N])
    #plt.plot(rolling_max.iloc[6,:N])
    
    
    ### concatenating
    
    conc_measure_data=measure_data[0,:1200].copy()
    for i in range(measure_data.shape[0]-1):
        conc_measure_data=np.hstack((conc_measure_data,measure_data[i+1,-85:-2]))
        
    conc_measure_data=np.expand_dims(conc_measure_data, axis=0)
    
    
    #### rolling window
    mdf=pd.DataFrame(conc_measure_data.copy())
    roll_mdf=mdf.rolling(axis=1,window=20,center=True)
    rolling_max = roll_mdf.max().abs()
    
    mdf[rolling_max>0.9]=np.NaN
    mdf.iloc[:,:10]=np.NaN
    mdf.iloc[:,-10:]=np.NaN
    
    #
    #plt.plot(mdf.iloc[0,:N])
    #plt.plot(rolling_max.iloc[0,:N])
    
    ### fhr
    weighted_sum = (measure_data[:,-2]*measure_data[:,-1]).sum()
    weights = measure_data[:,-1].sum()
    
    fhr = np.round(100*weighted_sum/weights)/100
    
    #import scipy.fftpack
    
    
    N=1200
    T=0.003 # sec
    ts=mdf.iloc[0,:N].dropna()
    #N=len(ts)
    
    #fig, ax = plt.subplots()
    #ax.plot(np.linspace(0,N*T,N),ts)
    #plt.show()
    
    
    yf = np.fft.fft(ts)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
    ax.text(0.8,0.8,str(fhr),horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
    ax.set_ylim(0,0.05)
    plt.show()
    
    #fig, ax = plt.subplots()
    #ax.plot(xf, 2.0/N * np.angle(yf[:N//2]))
    #plt.show()