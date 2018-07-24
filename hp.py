# Special tokens
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

# Vocabulary parameters
MIN_COUNT = 3
MIN_LENGTH = 3
MAX_LENGTH = 50

# Training hyperparameters
teacher_forcing_ratio = 1
clip = 5.0
learning_rate = 0.0001
decoder_learning_ratio = 5

# Training logging parameters
eval_bleu_every_epochs = 100
print_loss_every_iters = 100
save_every_epochs = 5

# NN hyperparameters
hidden_size = 500
embed_size = 500
batch_size = 256
n_epochs = 30
n_layers = 2
attention = "concat"
dropout = 0.1

# Checkpoint filenames
encoder_name = "encoder.pt"
decoder_name = "attn_decoder.pt"
encoder_checkpoint_name = "encoder_epoch_{}.pt"
decoder_checkpoint_name = "attn_decoder_epoch_{}.pt"

# Training data files
source_file = "iwslt14.tokenized.de-en/train.de"
target_file = "iwslt14.tokenized.de-en/train.en"
test_file = "iwslt14.tokenized.de-en/test.de"
reverse_languages = False
