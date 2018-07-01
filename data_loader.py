from collections import OrderedDict
import unicodedata
import re

PAD_token = 0
SOS_token = 1
EOS_token = 2

MIN_LENGTH = 3
MAX_LENGTH = 25


# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([,.!?])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


class Lang:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = OrderedDict()
        self.word2count = OrderedDict()
        self.index2word = OrderedDict({0: "PAD", 1: "SOS", 2: "EOS"})
        self.n_words = 3  # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = OrderedDict()
        self.word2count = OrderedDict()
        self.index2word = OrderedDict({0: "PAD", 1: "SOS", 2: "EOS"})
        self.n_words = 3  # Count default tokens

        for word in keep_words:
            self.index_word(word)


class LanguagePairLoader:
    def __init__(self, source_lang, target_lang):
        self.source_lang = source_lang
        self.target_lang = target_lang

    def load(self):
        input_lang, output_lang, pairs = self.prepare_data(True)

        MIN_COUNT = 3

        input_lang.trim(MIN_COUNT)
        output_lang.trim(MIN_COUNT)

        pairs = self.filter(input_lang, output_lang, pairs)

        return input_lang, output_lang, pairs

    def filter_pairs(self, pairs):
        filtered_pairs = []
        for pair in pairs:
            if len(pair[0]) >= MIN_LENGTH and len(pair[0]) <= MAX_LENGTH \
                    and len(pair[1]) >= MIN_LENGTH and len(pair[1]) <= MAX_LENGTH:
                filtered_pairs.append(pair)
        return filtered_pairs

    def prepare_data(self, reverse=False):
        input_lang, output_lang, pairs = self.read_langs(reverse)
        print("Read %d sentence pairs" % len(pairs))

        pairs = self.filter_pairs(pairs)
        print("Filtered to %d pairs" % len(pairs))

        print("Indexing words...")
        for pair in pairs:
            input_lang.index_words(pair[0])
            output_lang.index_words(pair[1])

        print('Indexed %d words in input language, %d words in output' % (input_lang.n_words, output_lang.n_words))
        return input_lang, output_lang, pairs

    def read_langs(self, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines
        #     filename = '../data/%s-%s.txt' % (lang1, lang2)
        filename = '%s-%s.txt' % (self.source_lang, self.target_lang)
        lines = open(filename).read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

        # Reverse pairs, make Lang instances
        if reverse:
            pairs = [list(reversed(p)) for p in pairs]
            input_lang = Lang(self.source_lang)
            output_lang = Lang(self.target_lang)
        else:
            input_lang = Lang(self.source_lang)
            output_lang = Lang(self.target_lang)

        return input_lang, output_lang, pairs

    def filter(self, input_lang, output_lang, pairs):
        keep_pairs = []

        for pair in pairs:
            input_sentence = pair[0]
            output_sentence = pair[1]
            keep_input = True
            keep_output = True

            for word in input_sentence.split(' '):
                if word not in input_lang.word2index:
                    keep_input = False
                    break

            for word in output_sentence.split(' '):
                if word not in output_lang.word2index:
                    keep_output = False
                    break

            # Remove if pair doesn't match input and output conditions
            if keep_input and keep_output:
                keep_pairs.append(pair)

        print(
            "Trimmed from %d pairs to %d, %.4f of total" % (len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
        return keep_pairs
