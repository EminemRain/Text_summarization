# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 10:26:03 2019

@author: pravech3
"""
from beam_search import beam_search
import tensorflow as tf
from transformer import Transformer, Generator, create_masks
from hyper_parameters import config
from metrics import optimizer
import tensorflow_datasets as tfds
from input_path import file_path
import time

tokenizer_en = tfds.features.text.SubwordTextEncoder.load_from_file(file_path.subword_vocab_path)

transformer = Transformer(
        num_layers=config.num_layers, 
        d_model=config.d_model, 
        num_heads=config.num_heads, 
        dff=config.dff, 
        input_vocab_size=config.input_vocab_size, 
        target_vocab_size=config.target_vocab_size, 
        rate=config.dropout_rate)

generator   = Generator()

def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(transformer=transformer,
                               optimizer=optimizer,
                               generator=generator)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)
    
    # if a checkpoint exists, restore the latest checkpoint.
    if tf.train.latest_checkpoint(checkpoint_path):
      ckpt.restore(tf.train.latest_checkpoint(checkpoint_path))
      print (ckpt_manager.latest_checkpoint, 'checkpoint restored!!')

def beam_search_eval(inp_sentences, beam_size):
  
  start = [tokenizer_en.vocab_size] * len(inp_sentences)
  end = [tokenizer_en.vocab_size+1]
  inp_sentences = [tokenizer_en.encode(i) for i in inp_sentences]
  encoder_input = tf.tile(inp_sentences, multiples=[beam_size, 1])
  batch, inp_shape = encoder_input.shape
  
  def transformer_query(output):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
          encoder_input, output)
    #print(output.shape[1])
    predictions, attention_weights, dec_output = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask
                                                   )

    if config.copy_gen:
      predictions = generator(dec_output, predictions, attention_weights, 
                              encoder_input, inp_shape, output.shape[-1], 
                              batch, False)

    # select the last sequence
    return (predictions[:,-1:,:])  # (batch_size, 1, target_vocab_size)
  return beam_search(
          transformer_query, 
          start, 
          beam_size, 
          config.summ_length, 
          config.target_vocab_size, 
          0.6, 
          stop_early=False, 
          eos_id=[end])


def infer(sample_document, beam_size, checkpoint_path):
    restore_chkpt(checkpoint_path)
    inp_sentences = [sample_document]
    beam_size_eval = beam_size
    start_time = time.time()
    translated_output_temp = beam_search_eval(inp_sentences, beam_size_eval)
    
    for true_summary, top_sentence_ids in zip(inp_sentences, translated_output_temp[0][:,0,:]):
      print('Original summary: {}'.format(true_summary))
      print('Predicted summary: {}'.format(tokenizer_en.decode([j for j in top_sentence_ids if j < tokenizer_en.vocab_size])))
    print('time to process {}'.format(time.time()-start_time))
    

if __name__ == "__main__":
    infer(file_path.document, beam_size=config.beam_size, checkpoint_path=file_path.old_checkpoint_path)