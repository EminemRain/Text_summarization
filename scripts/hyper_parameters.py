# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 18:09:43 2019

@author: pravech3
"""
from bunch import Bunch
#import tensorflow_datasets as tfds
#from input_path import file_path

#tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)

hyp = {
    "num_layers" : 3,                       #number of transformer blocks
    "d_model" : 256,                        #the projected word vector dimension
    "dff" : 512,                            #feed forward network hidden parameters
    "num_heads" : 8,                        #the number of heads in the multi-headed attention unit
    "dropout_rate" : 0.0,                           
    "epsilon_ls" : 0.1,                              #label_smoothing hyper parameter
    "batch_size" : 2,
    "epochs" : 2,
    "print_chks" : 50,
    "input_vocab_size" : 8096,                  # 8094 + start and end token
    "target_vocab_size" : 8096,
    "doc_length" : 51200,
    "summ_length" : 7000,
    "beam_size" : 3,
    "test_size" : 0.20,
    "copy_gen" : True,
    "decay_lr" : False,                             #decay learning rate
    "run_tensorboard" : True,
    "from_scratch" : True,
    "write_summary_op" : True
}
#print all the model hyperparameters



config = Bunch(hyp)

