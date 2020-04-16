# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:14:54 2017

@author: SzMike
"""

import os
#import exifread
import glob
from math import floor
from random import shuffle


def walklevel(root_dir, level=1):
    root_dir = root_dir.rstrip(os.path.sep)
    assert os.path.isdir(root_dir)
    num_sep = root_dir.count(os.path.sep)
    for root, dirs, files in os.walk(root_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]
            
def filelist_in_depth(base_dir,level=1,date_sort=False,included_extenstions = ['*.jpg', '*jpeg','*.bmp', '*.png', '*.gif','*.ppm']):
    filelist_indir=[]
    for root, dirs, files in walklevel(base_dir, level=level):
        for ext in included_extenstions:
            files_indir = glob.glob(os.path.join(root, ext))
            if date_sort:
                files_indir.sort(key=os.path.getmtime)
            filelist_indir.extend(files_indir)
    return filelist_indir


def get_training_and_testing_sets(file_list,split=0.7):
    shuffle(file_list)
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

#def get_exif_data(image_file,disp=False):
#    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
#
#    with open(image_file, 'rb') as f:
#        exif_data = exifread.process_file(f)
#    
#    if disp:
#        for k in sorted(exif_data.keys()):
#            if k not in ['JPEGThumbnail', 'TIFFThumbnail', 'Filename', 'EXIF MakerNote']:
#                print( '%s = %s' % (k, exif_data[k]) )
#                
#    return exif_data
    
#def _get_if_exist(data, key):
#    if key in data:
#        return data[key]
#		
#    return None
#	
#def _convert_to_degress(value):
#    """Helper function to convert the GPS coordinates stored in the EXIF to degress in float format"""
#    d0 = value[0][0]
#    d1 = value[0][1]
#    d = float(d0) / float(d1)
#
#    m0 = value[1][0]
#    m1 = value[1][1]
#    m = float(m0) / float(m1)
#
#    s0 = value[2][0]
#    s1 = value[2][1]
#    s = float(s0) / float(s1)
#
#    return d + (m / 60.0) + (s / 3600.0)
#
#def get_lat_lon(exif_data):
#    """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""
#    lat = None
#    lon = None
#
#    if "GPSInfo" in exif_data:		
#        gps_info = exif_data["GPSInfo"]
#
#        gps_latitude = _get_if_exist(gps_info, "GPSLatitude")
#        gps_latitude_ref = _get_if_exist(gps_info, 'GPSLatitudeRef')
#        gps_longitude = _get_if_exist(gps_info, 'GPSLongitude')
#        gps_longitude_ref = _get_if_exist(gps_info, 'GPSLongitudeRef')
#
#        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
#            lat = _convert_to_degress(gps_latitude)
#            if gps_latitude_ref != "N":                     
#                lat = 0 - lat
#
#            lon = _convert_to_degress(gps_longitude)
#            if gps_longitude_ref != "E":
#                lon = 0 - lon
#
#    return lat, lon

def check_folder(folder='.',create=True):
    if os.path.exists(folder):
        return True
    else:
        if create:
            os.makedirs(folder)
            return True
        else:
            return False