# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
PAD_text = "<PAD>"
SOS_text = "<SOS>"
UNK_text = "<UNK>"
EOS_text = "<EOS>"

# Vocabulary parameters
MIN_COUNT = 1
MIN_LENGTH = 3
MAX_LENGTH = 40

# Training hyperparameters
teacher_forcing_ratio = 1
clip = 5.0
learning_rate = 0.0001
decoder_learning_ratio = 5

# Training logging parameters
eval_bleu_every_epochs = 1
print_loss_every_iters = 10
save_every_epochs = 20

# NN hyperparameters
hidden_size = 512
embed_size = 512
batch_size = 256
n_epochs = 20
n_layers = 2
attention = "general"
dropout = 0.1

prefix = ""
# Load vocabs from training files
load_vocabs = True

# Checkpoint filenames
checkpoint_name = prefix + "checkpoint.pt"

# WMT16
source_file = prefix + "data/wmt14/train.tok.clean.bpe.32000.de"
target_file = prefix + "data/wmt14/train.tok.clean.bpe.32000.en"
source_test_file = prefix + "data/wmt14/newstest2015.tok.bpe.32000.de"
target_test_file = prefix + "data/wmt14/newstest2015.tok.bpe.32000.en"
source_eval_file = prefix + "data/wmt14/newstest2016.tok.bpe.32000.de"
target_eval_file = prefix + "data/wmt14/newstest2016.tok.bpe.32000.en"
bpe_file = prefix + "data/wmt14/bpe.32000"

'''
# Training data files
source_file = "iwslt14.tokenized.de-en/train.bpe.de"
target_file = "iwslt14.tokenized.de-en/train.bpe.en"
source_test_file = "iwslt14.tokenized.de-en/valid.bpe.de"
target_test_file = "iwslt14.tokenized.de-en/valid.bpe.en"
source_eval_file = "iwslt14.tokenized.de-en/test.bpe.de"
target_eval_file = "iwslt14.tokenized.de-en/test.bpe.en"
bpe_file = "codes.txt"
'''

reverse_languages = False
