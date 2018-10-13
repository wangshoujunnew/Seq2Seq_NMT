import tensorflow as tf

import input_helper as p_helper
from model import seq2seq_model


vocab_file = "process_data/vocab_map"
input_file = "process_data/predict_input"
vocab = p_helper.load_vocab(vocab_file)

params = {
    'vocab_size': len(vocab),
    'batch_size': 3,
    'embed_dim': 100,
    'num_units': 256,
    'input_max_length': 20,
    'output_max_length': 20,
    'dropout': 0.0,
    'beam_width': 0
}

model = tf.estimator.Estimator(
    model_fn=seq2seq_model,
    model_dir="model/seq2seq",
    params=params)

inputs_with_tokens = p_helper.predict_input_fn(input_file, vocab)

pred_input_fn = tf.estimator.inputs.numpy_input_fn(x=inputs_with_tokens,
                                                   shuffle=False,
                                                   num_epochs=1)

predictions_obj = model.predict(input_fn=pred_input_fn)
if params['beam_width'] > 0:
    final_answer = p_helper.get_out_put_from_tokens_beam_search(predictions_obj, vocab)
else:
    final_answer = p_helper.get_out_put_from_tokens(predictions_obj, vocab)

with open(input_file) as finput:
    for each_answer in final_answer:
        question = finput.readline()
        print('Source: ', question.replace('\n', '').replace('<EOS>', ''))
        print('Target: ', str(each_answer).replace('<EOS>', '').replace('<GO>', ''))
