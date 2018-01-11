#!/usr/bin/python3
# -*- coding: utf-8 -*-
import shutil
import os.path
from subprocess import run

names = ['train', 'members', 'songs', 'song_extra_info']
p7zip = False
data_path = "Data/"
zips_path = data_path + "zips/"
for n in names:
    if os.path.exists(data_path + n + '.csv'):
        print(n + ' OK')
    else:
        print(n + ' missing')
        file_name = n + '.csv'
        zip_name = file_name + '.7z'
        if not os.path.exists(zips_path) or not os.path.exists(zips_path + zip_name):
            print(zips_path + ' or ' + zips_path + zip_name + ' missing')
            print('ERROR: missing members csv and zip files. Download and unzip ' + n + ' data from \'https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data\' into Data directory.')
        else:
            if not p7zip:
                print('\ntry install p7zip-full')
                run('sudo apt-get install -y p7zip-full', shell=True)
                p7zip = True
            print('\nunzip ' + zip_name)
            run('cp -r ' + zips_path + zip_name + ' .', shell=True)
            run('p7zip -d ' + zip_name, shell=True)
            run('mv ' + file_name + ' ' + data_path + file_name, shell=True)
            print('\n')
