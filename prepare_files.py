# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:17:20 2020

@author: szabo
"""
import os
import json
import numpy as np
from tools.file_helper import filelist_in_depth, get_training_and_testing_sets
from matplotlib import pyplot as plt

import data_generator

base_dir=r'E:\DATA\FHR\Zoli'
#data_dir=os.path.join(base_dir,'Zoli','1')
model_dir=r'./model/20200421'
batch_size=128

measure_data_files=filelist_in_depth(base_dir,level=2,date_sort=False,included_extenstions = ['*.npy'])

training_data_files, testing_data_files, validation_data_files=get_training_and_testing_sets(measure_data_files,split=[0.7,0.2,0.1])

print('train: {0}, test: {1}, validation: {2}'.format(len(training_data_files),len(testing_data_files),len(validation_data_files)))
#measure_data=np.load(measure_data_files[0])


#####

train_generator=data_generator.myGenerator(training_data_files,batch_size=batch_size,end_alert=True)
test_generator=data_generator.myGenerator(testing_data_files,batch_size=batch_size,end_alert=True)
validation_generator=data_generator.myGenerator(validation_data_files,batch_size=batch_size,end_alert=True)

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
fig.savefig(os.path.join(model_dir,'sets.png'))

############

with open(os.path.join(model_dir,'validation_data.json'), 'w', encoding='utf-8') as f:
    json.dump(validation_data_files, f, ensure_ascii=False, indent=4)
with open(os.path.join(model_dir,'training_data.json'), 'w', encoding='utf-8') as f:
    json.dump(training_data_files, f, ensure_ascii=False, indent=4)
with open(os.path.join(model_dir,'testing_data.json'), 'w', encoding='utf-8') as f:
    json.dump(testing_data_files, f, ensure_ascii=False, indent=4)

#with open(os.path.join(model_dir,'validation_data.json'), 'r', encoding='utf-8') as f:
#    validation_data_files = json.load(f)
#### save sets if ok

## check by files
#for file in measure_data_files:
#    cur_generator=data_generator.myGenerator([file],batch_size=1)
#    y_c=np.empty([1,])
#    for x,y in cur_generator:
#        #x,y = next(train_generator)
#        #print(x.shape[0])
#        if y.shape[0]==0:
#            break
#        y_c=np.hstack([y_c,y]) 
#    print('{0} : {1:2d}'.format(os.path.basename(file),y_c.shape[0]))
#    print('{0:.2f} - {1:.2f}'.format(np.percentile(y_c,10),np.percentile(y_c,90)))
        