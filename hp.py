# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# Vocabulary parameters
MIN_COUNT = 2
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
save_every_epochs = 1

# NN hyperparameters
hidden_size = 512
embed_size = 512
batch_size = 256
n_epochs = 20
n_layers = 2
attention = "general"
dropout = 0.1

# Checkpoint filenames
checkpoint_name = "checkpoint.pt"

# WMT16
source_file = "data/wmt14/train.tok.clean.bpe.32000.de"
target_file = "data/wmt14/train.tok.clean.bpe.32000.en"
source_test_file = "data/wmt14/newstest2016.tok.bpe.32000.de"
target_test_file = "data/wmt14/newstest2016.tok.bpe.32000.en"
bpe_file = "data/wmt14/bpe.32000"
'''

# Training data files
source_file = "iwslt14.tokenized.de-en/train.de"
target_file = "iwslt14.tokenized.de-en/train.en"
source_test_file = "iwslt14.tokenized.de-en/test.de"
target_test_file = "iwslt14.tokenized.de-en/test.en"
'''
reverse_languages = False
