# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:20:36 2020

@author: szabo
"""

import os
import glob
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

N=1200

T=0.003 # sec

base_dir=r'D:\DATA\FHR'
data_dir=os.path.join(base_dir,'Zoli-20200327T075837Z-001','Zoli','0')

measure_data_files=glob.glob(os.path.join(data_dir,'*.npy'))


for i_file in range(len(measure_data_files)):
    i_file=1
    
    measure_data=np.load(measure_data_files[i_file])
    
    mdf=pd.DataFrame(measure_data[:,:1200].copy())
    roll_mdf=mdf.rolling(axis=1,window=20,center=True)
    rolling_max = roll_mdf.max().abs()
    
    mdf[rolling_max>0.9]=np.NaN
    mdf.iloc[:,:10]=np.NaN
    mdf.iloc[:,-10:]=np.NaN
    
    for i_sample in range(mdf.shape[0]):
        
        
        plt.plot(mdf.iloc[i_sample,:N])
        plt.plot(rolling_max.iloc[i_sample,:N])
        
        fhr=measure_data[i_sample,N]
        print(fhr)
        #fhr = np.round(100*weighted_sum/weights)/100
    
        ts=mdf.iloc[i_sample,:N].dropna()
        
        yf = np.fft.fft(ts)
        xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
        
        fig, ax = plt.subplots()
        ax.plot(xf, 2.0/N * np.abs(yf[:N//2]))
        ax.text(0.8,0.8,str(fhr),horizontalalignment='center',verticalalignment='center', transform=ax.transAxes)
        ax.set_ylim(0,0.05)
        plt.show()
