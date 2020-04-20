# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:53:12 2020

@author: MikeSzabolcs
"""

import numpy as np
N=1200
N2=int(N/2)

#T=0.003 # sec

def proc_data_fft(measure_data):
    # filter data with confidence
        
    measure_data = measure_data[measure_data[:,N+1]>0.5,:]
    measure_data = measure_data[measure_data[:,N]<1.5,:]
    # fft

    yf = np.fft.fft(measure_data[:,:N])
    xfft = 2.0/N * np.abs(yf[:,:N2])
    
    measure_data=np.hstack([xfft,
                            np.expand_dims(measure_data[:,N],axis=1),
                            np.expand_dims(measure_data[:,N+1],axis=1)])
    
    return measure_data


def myGenerator(data_files,batch_size=32, end_alert=False):
    i_file=0
    i_current=0
    remaining_data_chunk=np.empty([0,N2+2])
    while True:
        measure_data=np.load(data_files[i_file])
        # filter data with confidence
        
        measure_data = proc_data_fft(measure_data)
        
        len_data=measure_data.shape[0]        
        i_current=remaining_data_chunk.shape[0]
#        print(measure_data.shape)
        
        while True:
            n_rows2load = batch_size-i_current % batch_size
                
            if i_current+n_rows2load > len_data:
                new_data_chunk=measure_data[i_current:len_data]
                remaining_data_chunk=np.vstack([remaining_data_chunk,new_data_chunk])
                # Go to the next file
                i_file+=1
                i_current=0
                if i_file == len(data_files):
                    print('end of file list')
                    i_file=0 #so that fileIndex wraps back and loop goes on indefinitely
                    remaining_data_chunk=np.empty([0,N2+2])
                    if end_alert:
                        yield np.empty([0,]),np.empty([0,])
                break
            else:                   
                new_data_chunk=measure_data[i_current:i_current+n_rows2load]
                i_current+=n_rows2load
                xy=np.vstack([remaining_data_chunk,new_data_chunk])
                x=xy[:,0:N2]               
                x=x.reshape(x.shape[0],x.shape[1],1)
                y=xy[:,N2]
                remaining_data_chunk=np.empty([0,N2+2])
                yield x,y

