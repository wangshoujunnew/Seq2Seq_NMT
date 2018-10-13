import tensorflow as tf


def setting_decoder(helper, scope, num_units, encoder_outputs, encoder_final_state, input_lengths,
                    vocab_size, batch_size, output_max_length, embeddings, start_tokens, END_TOKEN, beam_width,
                    reuse=None):
    num_units = num_units * 2

    with tf.variable_scope(scope, reuse=reuse):

        if beam_width > 0:
            encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=beam_width)
            encoder_final_state = tf.contrib.seq2seq.tile_batch(encoder_final_state, multiplier=beam_width)
            input_lengths = tf.contrib.seq2seq.tile_batch(input_lengths, multiplier=beam_width)

        # Selecting the Attention Mechanism
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units=num_units, memory=encoder_outputs,
            memory_sequence_length=input_lengths)

        # Selecting the Cell Type to use
        cell = tf.contrib.rnn.GRUCell(num_units=num_units)

        # Wrapping attention to the cell
        attn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell, attention_mechanism, attention_layer_size=num_units)
        out_cell = tf.contrib.rnn.OutputProjectionWrapper(
            attn_cell, vocab_size, reuse=reuse
        )

        if (beam_width > 0):

            encoder_state = out_cell.zero_state(dtype=tf.float32,
                                                batch_size=batch_size * beam_width).clone(
                cell_state=encoder_final_state)

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=out_cell, embedding=embeddings,
                start_tokens=tf.to_int32(start_tokens), end_token=END_TOKEN,
                initial_state=encoder_state,
                beam_width=beam_width,
                length_penalty_weight=0.0)

            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=False, maximum_iterations=output_max_length
            )
            return outputs[0]

        else:
            decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell, helper=helper,
                                                      initial_state=out_cell.zero_state(dtype=tf.float32,
                                                                                        batch_size=batch_size).clone(
                                                          cell_state=encoder_final_state))
            outputs = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder, output_time_major=False,
                impute_finished=True, maximum_iterations=output_max_length
            )
            return outputs[0]
