# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:53:12 2020

@author: MikeSzabolcs
"""

import numpy as np


batch_size = 65

from matplotlib import pyplot as plt


def myGenerator(data_files,batch_size=32):
    i_file=0
    i_current=0
    remaining_data_chunk=np.empty([0,1202])
    while True:
        measure_data=np.load(data_files[i_file])
        # filter data with confidence
        # todo: data preprocessing
        measure_data = measure_data[measure_data[:,1201]>0.5,:]
        
        len_data=measure_data.shape[0]
        #print(len_data)
        i_current=remaining_data_chunk.shape[0]
        
        while True:
            n_rows2load = batch_size-i_current % batch_size
                
            if i_current+n_rows2load > len_data:
                new_data_chunk=measure_data[i_current:len_data]
                i_current=0
                remaining_data_chunk=np.vstack([remaining_data_chunk,new_data_chunk])
                # Go to the next file
                i_file+=1
                if i_file == len(data_files):
                    print('end of file list')
                    i_file=0 #so that fileIndex wraps back and loop goes on indefinitely
                    yield np.empty([1,]),np.empty([1,])
                    break
            else:                   
                new_data_chunk=measure_data[i_current:i_current+n_rows2load]
                i_current+=n_rows2load
                xy=np.vstack([remaining_data_chunk,new_data_chunk])
                x=xy[:,0:1200]
                x=x.reshape(x.shape[0],x.shape[1],1)
                y=xy[:,1200]
                yield x,y
                remaining_data_chunk=np.empty([0,1202])

train_generator=myGenerator(training_data_files,batch_size=batch_size)
test_generator=myGenerator(testing_data_files,batch_size=batch_size)
validation_generator=myGenerator(validation_data_files,batch_size=batch_size)

y_c=np.empty([1,])
for x,y in validation_generator:
    #x,y = next(train_generator)
    #print(x.shape[0])
    if y.shape[0]==1:
        break
    y_c=np.hstack([y_c,y])    
    
hist, bins = np.histogram(y_c,1000)
plt.plot(bins[1:],hist)