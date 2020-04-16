# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 21:17:20 2020

@author: szabo
"""
import os
from tools.file_helper import filelist_in_depth, get_training_and_testing_sets

base_dir=r'E:\DATA\FHR\Zoli'
#data_dir=os.path.join(base_dir,'Zoli','1')

measure_data_files=filelist_in_depth(base_dir,level=2,date_sort=False,included_extenstions = ['*.npy'])

training_data_files, testing_data_files, validation_data_files=get_training_and_testing_sets(measure_data_files,split=[0.7,0.2,0.1])

#measure_data=np.load(measure_data_files[0])
