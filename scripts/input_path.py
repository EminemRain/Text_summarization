# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:28:12 2019

@author: pravech3
"""
from bunch import Bunch
from hyper_parameters import config
import datetime
import tensorflow as tf
import os

with open('C:/TAC/Scripts/Training/input_files/sample.txt') as f:
    text = f.read()
    
if not os.path.exists("C:/TAC/Scripts/Training/created_files/summaries"):
    os.mkdir("C:/TAC/Scripts/Training/created_files/summaries")
    
file_path = {
        'csv_path' : 'C:/TAC/Scripts/Training/input_files/2019-10-23_12_18_54.csv',
        'subword_vocab_path' : 'C:/TAC/Scripts/Training/input_files/vocab_file_summarization',
        'old_checkpoint_path' : "C:/TAC/Scripts/Training/created_files/training_summarization_model_ckpts/",
        'tensorboard_log' : "C:/TAC/Scripts/Training/created_files/tensorboard_logs/",
        'summary_write_path' : "C:/TAC/Scripts/Training/created_files/summaries/summary_for_epoch_",
        'document' : text
        
}

if config.from_scratch:
    fol_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path['new_checkpoint_path'] = "C:/TAC/Daily_cases/training_summarization_model_ckpts"+fol_name
file_path = Bunch(file_path)


if config.run_tensorboard:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = file_path.tensorboard_log + current_time + '/train'
    validation_log_dir = file_path.tensorboard_log + current_time + '/validation'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(validation_log_dir)
