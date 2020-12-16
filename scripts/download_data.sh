#!/bin/sh

mkdir data
cd data
wget https://storage.yandexcloud.net/hackathon-2020/trainset.zip
unzip trainset.zip
export YADC_DATASET_PATH=$pwd
cd ..