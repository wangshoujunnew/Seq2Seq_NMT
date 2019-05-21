import tensorflow as tf
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.contrib import layers

import input_helper as p_helper
import decoder

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2


def seq2seq_model(features, labels, mode, params):
    """
    model_fn参数
    :param features:
    :param labels: dict
    :param mode: 训练,验证,测试
    :param params: 超参数
    :return: EstimatorSpec
    """
    vocab_size = params['vocab_size']
    embed_dim = params['embed_dim']
    num_units = params['num_units']
    output_max_length = params['output_max_length']
    dropout = params['dropout']
    beam_width = params['beam_width']

    inp = features['input']
    batch_size = tf.shape(inp)[0]
    start_tokens = tf.zeros([batch_size], dtype=tf.int64)
    input_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(inp, 1)), 1)

    # 对输入序列数据的嵌入工作 , layers.embed_sequence
    input_embed = layers.embed_sequence(
        inp, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed')

    with tf.variable_scope('embed', reuse=True):
        embeddings = tf.get_variable('embeddings')

    fw_cell = tf.contrib.rnn.GRUCell(num_units=num_units)
    bw_cell = tf.contrib.rnn.GRUCell(num_units=num_units)

    if dropout > 0.0:
        print("  %s, dropout=%g " % (type(fw_cell).__name__, dropout))
        fw_cell = tf.contrib.rnn.DropoutWrapper(
            cell=fw_cell, input_keep_prob=(1.0 - dropout))
        bw_cell = tf.contrib.rnn.DropoutWrapper(
            cell=bw_cell, input_keep_prob=(1.0 - dropout))

    bd_encoder_outputs, bd_encoder_final_state = \
        tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell,
                                        inputs=input_embed, dtype=tf.float32)

    encoder_outputs = tf.concat(bd_encoder_outputs, -1)
    encoder_final_state = tf.concat(bd_encoder_final_state, -1)

    pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embeddings, start_tokens=tf.to_int32(start_tokens), end_token=END_TOKEN)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Specific for Prediction
        pred_outputs = decoder.setting_decoder(pred_helper, 'decode', num_units, encoder_outputs,
                                               encoder_final_state, input_lengths,
                                               vocab_size, batch_size, output_max_length,
                                               embeddings, start_tokens, END_TOKEN, beam_width,
                                               reuse=False)

        if beam_width > 0:
            tf.identity(pred_outputs.predicted_ids, name='predictions')
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_outputs.predicted_ids)
        else:
            tf.identity(pred_outputs.sample_id[0], name='predictions')
            return tf.estimator.EstimatorSpec(mode=mode, predictions=pred_outputs.sample_id)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Specific For Training
        output = features['output']
        train_output = tf.concat([tf.expand_dims(start_tokens, 1), output], 1)
        output_lengths = tf.reduce_sum(tf.to_int32(tf.not_equal(train_output, 1)), 1)

        output_embed = layers.embed_sequence(
            train_output, vocab_size=vocab_size, embed_dim=embed_dim, scope='embed', reuse=True)

        train_helper = tf.contrib.seq2seq.TrainingHelper(output_embed, output_lengths)

        train_outputs = decoder.setting_decoder(train_helper, 'decode', num_units, encoder_outputs,
                                                encoder_final_state, input_lengths,
                                                vocab_size, batch_size, output_max_length, embeddings,
                                                start_tokens, END_TOKEN, beam_width, reuse=None)

        pred_outputs = decoder.setting_decoder(pred_helper, 'decode', num_units, encoder_outputs,
                                               encoder_final_state, input_lengths,
                                               vocab_size, batch_size, output_max_length, embeddings,
                                               start_tokens, END_TOKEN, beam_width, reuse=True)

        tf.identity(train_outputs.sample_id[0], name='train_pred')
        weights = tf.to_float(tf.not_equal(train_output[:, :-1], 1))

        logits = tf.identity(train_outputs.rnn_output, 'logits')

        loss = tf.contrib.seq2seq.sequence_loss(
            logits, output, weights=weights)

        train_op = layers.optimize_loss(
            loss, tf.train.get_global_step(),
            optimizer=params.get('optimizer', 'Adam'),
            learning_rate=params.get('learning_rate', 0.001),
            summaries=['loss', 'learning_rate'])

        tf.identity(pred_outputs.sample_id[0], name='predictions')
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_outputs.sample_id,
            loss=loss,
            train_op=train_op
        )