#!/bin/sh

mkdir data
cd data
wget https://storage.yandexcloud.net/hackathon-2020/trainset.zip
unzip trainset.zip
export YADC_DATASET_PATH=$pwd

$filename = 'GeneralUser_GS_1.471.zip'
wget https://storage.yandexcloud.net/hackathon-2020/GeneralUser%20GS%201.471.zip $filename
unzip $filename
export SOUND_FONT_PATH='$pwd$filename'
cd ..