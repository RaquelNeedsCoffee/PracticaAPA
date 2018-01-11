#!/bin/bash
echo 'installing dependencies'
sudo apt-get install python3 python3-pip
echo ''
echo 'installing requirements'
sudo pip3 install -r requirements.txt
echo ''
echo 'installation finished'
echo ''
echo ''
echo 'check data dir and files'
if [ -d "Data" ]; then
  echo "Data dir OK"
else
  echo "Data dir missing"
  echo "Make Data dir"
  mkdir Data
  echo "ERROR: download and unzip train.csv, members.csv, songs.csv and song_extra_info.csv from 'https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data' into Data directory."
fi
python3 check_files.py
