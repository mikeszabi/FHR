# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 21:53:12 2020

@author: MikeSzabolcs
"""

import numpy as np
from matplotlib import pyplot as plt


batch_size = 128



def myGenerator(data_files,batch_size=32):
    i_file=0
    i_current=0
    remaining_data_chunk=np.empty([0,1202])
    while True:
        measure_data=np.load(data_files[i_file])
        # filter data with confidence
        # todo: data preprocessing
        measure_data = measure_data[measure_data[:,1201]>0.5,:]
        measure_data = measure_data[measure_data[:,1200]<1.5,:]
        # filter
        
        len_data=measure_data.shape[0]        
        i_current=remaining_data_chunk.shape[0]

        
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
                    remaining_data_chunk=np.empty([0,1202])
                    yield np.empty([0,]),np.empty([0,])
                break
            else:                   
                new_data_chunk=measure_data[i_current:i_current+n_rows2load]
                i_current+=n_rows2load
                xy=np.vstack([remaining_data_chunk,new_data_chunk])
                x=xy[:,0:1200]
                x=x.reshape(x.shape[0],x.shape[1],1)
                y=xy[:,1200]
                remaining_data_chunk=np.empty([0,1202])
                yield x,y

#####

train_generator=myGenerator(training_data_files,batch_size=batch_size)
test_generator=myGenerator(testing_data_files,batch_size=batch_size)
validation_generator=myGenerator(validation_data_files,batch_size=batch_size)

generators=[train_generator,test_generator,validation_generator]
fig, ax = plt.subplots( nrows=3, ncols=1 )  # create figure & 1 axis

ax[0].set_title('Train')
ax[1].set_title('Test')
ax[2].set_title('Validation')


for i, generartor in enumerate(generators):
    y_c=np.empty([0,])
    for x,y in generartor:
        #x,y = next(train_generator)
        #print(x.shape[0])
        if y.shape[0]==0:
            break
        y_c=np.hstack([y_c,y])    
    
    print('{0:2d} : {1}'.format(i,len(y_c)))
        
    hist, bins = np.histogram(y_c,1000)
    ax[i].plot(bins[1:],hist)
    ax[i].text(0,0,str(len(y_c)),fontsize=12)
    ax[i].set_xlim(0,1)
    #print(y_c.max())
plt.show()
fig.savefig('sets.png')

############


for file in measure_data_files:
    cur_generator=myGenerator([file],batch_size=1)
    y_c=np.empty([1,])
    for x,y in cur_generator:
        #x,y = next(train_generator)
        #print(x.shape[0])
        if y.shape[0]==0:
            break
        y_c=np.hstack([y_c,y]) 
    print('{0} : {1:2d}'.format(os.path.basename(file),y_c.shape[0]))
    print('{0:.2f} - {1:.2f}'.format(np.percentile(y_c,10),np.percentile(y_c,90)))
        