import re
import string
from unicodedata import normalize

from numpy.ma import array

FILE_PATH = 'raw_data/deu.txt'


def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs


def clean_pairs(lines):
    cleaned = list()
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            line = line.split()
            line = [word.lower() for word in line]
            line = [word.translate(table) for word in line]
            line = [re_print.sub('', w) for w in line]
            line = [word for word in line if word.isalpha()]
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)


min_line_length = 2  # Minimum number of words required to be in training
max_line_length = 30  # Minimum number of words allowed to be in training
frequency_of_word = 1  # minumum number of word count usages


def read_data_from_file(filename):
    lines = open(filename).read().split('\n')
    return lines


def create_dictionary_word_usage(selected_source, selected_target):
    # Create a dictionary for the frequency of the vocabulary
    vocab = {}
    for source in selected_source:
        for word in source.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1

    for target in selected_target:
        for word in target.split():
            if word not in vocab:
                vocab[word] = 1
            else:
                vocab[word] += 1
    return vocab


def vocab_from_word_to_emb_without_rare_word(dict_word_usage, min_number_of_usage):
    vocab_words_to_int = {}

    vocab_words_to_int['<GO>'] = 0
    vocab_words_to_int['<EOS>'] = 1
    vocab_words_to_int['<UNK>'] = 2
    vocab_words_to_int['<PAD>'] = 3

    word_num = 4
    for word, count in dict_word_usage.items():
        # maximum number of characters allowed in a word
        if len(word) <= 20:
            if count >= min_number_of_usage:
                vocab_words_to_int[word] = word_num
                word_num += 1

    return vocab_words_to_int


def write_lines_to_file(filename, list_of_lines):
    with open(filename, 'w') as file_to_write:
        for i in range(len(list_of_lines)):
            file_to_write.write(list_of_lines[i] + "\n")


def write_dict_to_file(dict_to_write, file_to_write):
    with open(file_to_write, 'w') as file_to:
        for key, val in dict_to_write.items():
            file_to.write(str(key) + "=" + str(val) + "\n")


def sort_text_based_on_number_of_words(sources, targets, max_line_length):
    # Sort sources and targets by the length of sources.
    # This will reduce the amount of padding during training
    # Which should speed up training and help to reduce the loss

    sorted_sources = []
    sorted_targets = []

    for length in range(min_line_length, max_line_length):
        for i, ques in enumerate(sources):
            ques_tmp = ques.split(" ")
            if len(ques_tmp) == length:
                sorted_sources.append(sources[i])
                sorted_targets.append(targets[i])

    return sorted_sources, sorted_targets


def main_prepare_data():
    doc = load_doc(FILE_PATH)

    pairs = to_pairs(doc)

    print(len(pairs))

    data_source_file = 'process_data/english'
    data_target_file = 'process_data/german'

    total_samples = create_source_target_file_from_reddit_main_file(pairs, data_source_file, data_target_file,
                                                                    min_line_length,
                                                                    max_line_length)

    print('Total num of samples', total_samples)

    selected_sources = read_data_from_file(data_source_file)
    selected_targets = read_data_from_file(data_target_file)

    selected_sources = clean_sentence(selected_sources)
    selected_targets = clean_sentence(selected_targets)

    dict_word_usage = create_dictionary_word_usage(selected_sources, selected_targets)
    print("Total number of words started with in dictionary ", len(dict_word_usage))
    # Create a common vocab for sources and targets along with the special codes
    vocab_words_to_int = vocab_from_word_to_emb_without_rare_word(dict_word_usage, frequency_of_word)

    write_dict_to_file(vocab_words_to_int, 'process_data/vocab_map')
    print("Total number of words finally in dictionary ", len(vocab_words_to_int))

    # sort the sources and targets based on the number of words in the line
    sorted_sources, sorted_targets = sort_text_based_on_number_of_words(
        selected_sources, selected_targets, max_line_length)

    write_lines_to_file("process_data/english_final", sorted_sources)
    write_lines_to_file("process_data/german_final", sorted_targets)


def clean_sentence(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        sentence = clean_text(sentence)
        cleaned_sentences.append(sentence)
    return cleaned_sentences


def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"temme", "tell me", text)
    text = re.sub(r"gimme", "give me", text)
    text = re.sub(r"howz", "how is", text)
    text = re.sub(r"let's", "let us", text)
    text = re.sub(r" & ", " and ", text)
    text = re.sub(r"[-()\"#[\]/@;:<>{}`*_+=&~|.!/?,]", "", text)

    return text


def create_source_target_file_from_reddit_main_file(pairs, source_file, target_file, min_words, max_words):
    source_file = open(source_file, 'w', newline='\n', encoding='utf-8')
    target_file = open(target_file, 'w', newline='\n', encoding='utf-8')
    number_of_samples = 0
    for line in pairs:
        number_of_words_source = len(line[0])
        number_of_words_target = len(line[1])
        if (number_of_words_source >= min_words and number_of_words_source <= max_words
                and number_of_words_target >= min_words and number_of_words_target <= max_words):
            source_file.write(line[0])
            source_file.write('\n')
            target_file.write(line[1])
            target_file.write('\n')
            number_of_samples += 1

    source_file.close()
    target_file.close()
    return number_of_samples


main_prepare_data()
