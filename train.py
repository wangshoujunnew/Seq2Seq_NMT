import tensorflow as tf

import input_helper as input_helper
from model import seq2seq_model

tf.logging.set_verbosity(tf.logging.INFO)

vocab_file = "process_data/vocab_map"
input_file = "process_data/english_final"
output_file = "process_data/german_final"

vocab = input_helper.load_vocab(vocab_file)

params = {
    'vocab_size': len(vocab),
    'batch_size': 64,
    'input_max_length': 20,
    'output_max_length': 20,
    'embed_dim': 100,
    'num_units': 256,
    'dropout': 0.2,
    'beam_width': 0
}

input_fn, feed_fn = input_helper.make_input_fn(
    params['batch_size'],
    input_file,
    output_file,
    vocab, params['input_max_length'], params['output_max_length'])

run_config = tf.estimator.RunConfig(
    model_dir="model/seq2seq",
    keep_checkpoint_max=5,
    save_checkpoints_steps=500,
    log_step_count_steps=10)

# Estimator 无需使用placeholder、session，计算结果能够立即得出
seq2seq_esti = tf.estimator.Estimator(
    model_fn=seq2seq_model, # 使用的模型
    config=run_config,
    params=params) # 超参数

seq2seq_esti.train(
    input_fn=input_fn,
    hooks=[tf.train.FeedFnHook(feed_fn)],
    steps=100)
