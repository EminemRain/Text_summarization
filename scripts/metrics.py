# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:41:22 2019

@author: pravech3
"""
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from hyper_parameters import config
from rouge import Rouge
from input_path import file_path

tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)
rouge_all = Rouge()


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
    
  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)
    
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

if config.decay_lr:
    lr = CustomSchedule(config.d_model)
else:
    lr = 3e-4

def label_smoothing(inputs, epsilon=config.epsilon_ls):
    V = inputs.get_shape().as_list()[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  real = label_smoothing(tf.one_hot(real, depth=tokenizer_en.vocab_size+2))
  # pred shape =  (batch_size, tar_seq_len, target_vocab_size)
  # real shape =  (batch_size, tar_seq_len, target_vocab_size)
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask
  
  return tf.reduce_mean(loss_)

def get_loss_and_accuracy():
    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    return(loss, accuracy)
    
def write_summary(tar_real, predictions, inp, epoch, write=config.write_summary_op):
  r_avg_final = []
  
  for i, sub_tar_real in enumerate(tar_real):
    predicted_id = tf.cast(tf.argmax(predictions[i], axis=-1), tf.int32)
    sum_ref = tokenizer_en.decode([i for i in sub_tar_real if i < tokenizer_en.vocab_size])
    sum_hyp = tokenizer_en.decode([i for i in predicted_id if i < tokenizer_en.vocab_size if i > 0])
    rouges = rouge_all.get_scores(sum_ref , sum_hyp)[0]
    r1_val, r2_val, rl_val = rouges['rouge-1']["f"], rouges['rouge-2']["f"], rouges['rouge-l']["f"]
    r_avg = np.mean([r1_val, r2_val, rl_val], dtype=np.float64)
    r_avg_final.append(r_avg)

  if write:
    with tf.io.gfile.GFile(file_path.summary_write_path+str(epoch.numpy()), 'a') as f:
      for i, sub_tar_real in enumerate(tar_real):
        f.write(sum_ref+'\t'+sum_hyp+'\n')
  score =  np.mean(r_avg_final, dtype=np.float64)
  return score


optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)
loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction='none')
