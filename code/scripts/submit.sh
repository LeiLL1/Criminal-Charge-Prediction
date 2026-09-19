#!/bin/bash
currentPath="$( cd "$( dirname "$0"  )" && pwd  )"
cd ..
pwdPath="$(pwd)"

MODEL_PATH=/home/icisee/CXL/chinesebertwwmext
TRAIN_DATA_PATH1=/home/icisee/CXL/Data_Clearning/CleanSimple300_1000_train0.8.csv
EVAL_DATA_PATH1=/home/icisee/CXL/Data_Clearning/CleanSimple300_1000_val0.1.csv

#MODEL_PATH=/home/lincong/llh/PythonProjects/experiment0/bert_base_uncased
#TRAIN_DATA_PATH2=/home/lincong/llh/PythonProjects/experiment0/test/16-shot-data/BBCNews/train.csv
#EVAL_DATA_PATH2=/home/lincong/llh/PythonProjects/experiment0/test/data/BBCNews/BBC_News_val.csv
#
#MODEL_PATH=/home/lincong/llh/PythonProjects/experiment0/bert_base_uncased
#TRAIN_DATA_PATH3=/home/lincong/llh/PythonProjects/experiment0/test/16-shot-data/SMS/train.csv
#EVAL_DATA_PATH3=/home/lincong/llh/PythonProjects/experiment0/test/data/SMS/SMS_val1.csv


python -m finetuned-bert \
  --seed 1234 \
  --epochs 4 \
  --batch_size 4 \
  --max_seq_length 512 \
  --learning_rate 1.5e-4 \
  --log_freq 200 \
  --eval_freq 1000 \
  --model_path $MODEL_PATH \
  --train_data_path  $TRAIN_DATA_PATH1 \
  --eval_data_path $EVAL_DATA_PATH1

#################################################

#python -m finetuned-bert \
#  --seed 123 \
#  --epochs 1600 \
#  --batch_size 8 \
#  --max_seq_length 256 \
#  --learning_rate 5e-5 \
#  --log_freq 100 \
#  --eval_freq 500 \
#  --model_path $MODEL_PATH \
#  --train_data_path  $TRAIN_DATA_PATH2 \
#  --eval_data_path $EVAL_DATA_PATH2
#
###################################################
#
#python -m finetuned-bert \
#  --seed 123 \
#  --epochs 1000 \
#  --batch_size 8 \
#  --max_seq_length 256 \
#  --learning_rate 5e-5 \
#  --log_freq 100 \
#  --eval_freq 500 \
#  --model_path $MODEL_PATH \
#  --train_data_path  $TRAIN_DATA_PATH3 \
#  --eval_data_path $EVAL_DATA_PATH3


