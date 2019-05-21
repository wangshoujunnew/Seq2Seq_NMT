import numpy as np
import tensorflow as tf

GO_TOKEN = 0
END_TOKEN = 1
UNK_TOKEN = 2


def tokenize_and_map(line, vocab):
    return [vocab.get(token, UNK_TOKEN) for token in line.split(' ')]


def make_input_fn(
        batch_size, input_filename, output_filename, vocab,
        input_max_length, output_max_length,
        input_process=tokenize_and_map, output_process=tokenize_and_map):

    def input_fn():
        source = tf.placeholder(tf.int64, shape=[None, None], name='input')
        target = tf.placeholder(tf.int64, shape=[None, None], name='output')
        tf.identity(source[0], 'input_0')
        tf.identity(target[0], 'output_0')
        return {
                   'input': source,
                   'output': target,
               }, None

    def sampler():
        while True:
            with open(input_filename,encoding='utf-8',mode='r') as finput:
                with open(output_filename,encoding='utf-8',mode='r') as foutput:
                    for in_line in finput:
                        out_line = foutput.readline()
                        yield {
                            'input': input_process(in_line, vocab)[:input_max_length - 1] + [END_TOKEN],
                            'output': output_process(out_line, vocab)[:output_max_length - 1] + [END_TOKEN]
                        }

    sample_me = sampler()

    def feed_fn():
        inputs, outputs = [], []
        input_length, output_length = 0, 0
        for i in range(batch_size):
            rec = next(sample_me)
            inputs.append(rec['input'])
            outputs.append(rec['output'])
            input_length = max(input_length, len(inputs[-1]))
            output_length = max(output_length, len(outputs[-1]))
        # Pad me right with </S> token.
        for i in range(batch_size):
            inputs[i] += [END_TOKEN] * (input_length - len(inputs[i]))
            outputs[i] += [END_TOKEN] * (output_length - len(outputs[i]))
        return {
            'input:0': inputs,
            'output:0': outputs
        }

    return input_fn, feed_fn


def predict_input_fn(input_filename, vocab, input_process=tokenize_and_map):
    max_len = 0.

    with open(input_filename,encoding='utf-8',mode='r') as finput:
        for in_line in finput:
            max_len = max(len(in_line.split(" ")), max_len)

    predict_lines = np.empty(max_len + 1, int)

    with open(input_filename,encoding='utf-8',mode='r') as finput:
        for in_line in finput:
            in_line = in_line.lower()
            new_line_tmp = np.array(input_process(in_line, vocab), dtype=int)
            new_line = np.append(new_line_tmp, np.array([UNK_TOKEN for _ in range(max_len - len(new_line_tmp))] +
                                                        [int(END_TOKEN)], dtype=int))
            predict_lines = np.vstack((predict_lines, new_line))

    pred_line_tmp = np.delete(predict_lines, 0, 0)

    pred_lines = np.array(pred_line_tmp)
    return {'input': pred_lines}


def load_vocab(filename):
    vocab = {}
    with open(filename,encoding='utf-8',mode='r') as f:
        for idx, line in enumerate(f):
            tmp_val = (line.strip(" \n")).split("=")
            vocab[tmp_val[0]] = int(tmp_val[1])
    return vocab


def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}


def get_formatter(keys, vocab):
    rev_vocab = get_rev_vocab(vocab)

    def to_str(sequence):
        tokens = [rev_vocab.get(x, "<UNK>") for x in sequence]
        return ' '.join(tokens)

    def format(values):
        res = []
        for key in keys:
            res.append("%s = %s" % (key, to_str(values[key])))
        return '\n'.join(res)

    return format


def get_out_put_from_tokens(all_sentences, vocab):
    rev_vocab = get_rev_vocab(vocab)
    all_string_sent = []
    for each_sent in all_sentences:
        string_sent = []
        for each_word in each_sent:
            string_sent.append(rev_vocab.get(each_word))
        all_string_sent.append(' '.join(string_sent))
    return all_string_sent


def get_out_put_from_tokens_beam_search(all_sentences, vocab):
    rev_vocab = get_rev_vocab(vocab)
    all_string_sent = []
    for each_sent in all_sentences:
        each_sent = each_sent[:, 0]
        string_sent = []
        for each_word in each_sent:
            string_sent.append(rev_vocab.get(each_word))
        all_string_sent.append(' '.join(string_sent))
    return all_string_sent
