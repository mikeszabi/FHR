# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:24:08 2020

@author: szabo
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Reshape

batch_size = 32
nb_classes = 10
nb_epoch = 12

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='acc', patience=1)
]

def myGenerator(data_files,batch_size=32):
    i_file=0
    i_current=0
    remaining_data_chunk=np.empty([0,1202])
    while True:
        measure_data=np.load(data_files[i_file])
        # filter data with confidence
        measure_data = measure_data[measure_data[:,1201]>0.5,:]
        
        len_data=measure_data.shape[0]
        print(len_data)
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
                    i_file=0 #so that fileIndex wraps back and loop goes on indefinitely
                break
            else:                   
                new_data_chunk=measure_data[i_current:i_current+n_rows2load]
                i_current+=n_rows2load
                xy=np.vstack([remaining_data_chunk,new_data_chunk])
                x=xy[:,0:1200]
                x=x.reshape(x.shape[0],x.shape[1],1)
                y=xy[:,1201]
                yield x,y
                remaining_data_chunk=np.empty([0,1202])

train_generator=myGenerator(training_data_files,batch_size=batch_size)


# define model
model = Sequential()
#model.add(Reshape((1200, 1), input_shape=(1200,)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(1200,1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(model.summary())
model.fit_generator(train_generator, steps_per_epoch = 100, epochs = 10, verbose=2, callbacks=[], validation_data=None)



genesis=myGenerator(training_data_files,batch_size=32)

a=next(genesis)

for x,y in genesis:
    b=2
    print(x.shape[0])