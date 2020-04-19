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

from matplotlib import pyplot as plt

batch_size = 256


callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=r'./models/best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.TensorBoard()
]

def myGenerator(data_files,batch_size=32, end_alert=True):
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
                    if end_alert:
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


train_generator=myGenerator(training_data_files,batch_size=batch_size)
validation_generator=myGenerator(validation_data_files,batch_size=batch_size)


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

model.fit_generator(train_generator, steps_per_epoch = 1000, epochs = 10, 
                    validation_data=validation_generator, validation_steps=150,
                    verbose=2, callbacks=callbacks_list)


model.reset_metrics()
#genesis=myGenerator(validation_data_files,batch_size=32)
#
#a=next(genesis)
#
#for x,y in genesis:
#    b=2
#    #print(x.shape[0])
    
    
# Save the model
model.save(r'./models/best_model.h5')

# Recreate the exact same model purely from the file
new_model = tf.keras.models.load_model(r'./models/best_model.h5')

test_generator=myGenerator(testing_data_files,batch_size=256)


for x,y in test_generator:
    
    x,y = next(train_generator)
    new_predictions = new_model.predict(x)
    plt.scatter(y,new_predictions,marker='o')
    #print(x.shape[0])
    