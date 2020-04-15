# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:20:36 2020

@author: szabo
"""

import os
import numpy as np
import pandas as pd
import time

from tools.file_helper import filelist_in_depth 

from matplotlib import pyplot as plt

N=1200

T=0.003 # sec

base_dir=r'D:\DATA\FHR'
data_dir=os.path.join(base_dir,'Zoli','1')

measure_data_files=filelist_in_depth(base_dir,level=2,date_sort=False,included_extenstions = ['*.npy'])


for i_file in range(len(measure_data_files)):
    i_file=2
    
    measure_data=np.load(measure_data_files[i_file])
    
    fig, ax = plt.subplots(2,1)
    ax[0].plot(measure_data[:,1200])
    ax[1].plot(measure_data[:,1201])
    
    mdf=pd.DataFrame(measure_data[:,:1200].copy())
    roll_mdf=mdf.rolling(axis=1,window=20,center=True)
    rolling_max = roll_mdf.max().abs()
    
    mdf[rolling_max>0.9]=np.NaN
    mdf.iloc[:,:10]=np.NaN
    mdf.iloc[:,-10:]=np.NaN
    
    for i_sample in range(mdf.shape[0]):
        
        i_sample=0
        fig, ax = plt.subplots()
        ax.plot(mdf.iloc[i_sample,:N])
        ax.plot(rolling_max.iloc[i_sample,:N])
        
        fhr=measure_data[i_sample,N]
        print(fhr)
        #fhr = np.round(100*weighted_sum/weights)/100
    
        ts=mdf.iloc[i_sample,:N].dropna()
        
        t=time.time()
        yf = np.fft.fft(ts)
        print(time.time()-t)
        
        xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
        
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        ax.text(0.8,0.8,str(fhr),horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
        ax.set_ylim(0,0.05)
        ax.set_xlim(0,1.0/(2.0*T))
        plt.show()
