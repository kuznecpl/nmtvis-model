from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable

import sys

sys.path.insert(0, './myseq2seq')
for p in sys.path:
    print(p)

from models import AttnDecoderRNN, EncoderRNN, LSTMAttnDecoderRNN, LSTMEncoderRNN

import hp
from hp import PAD_token, SOS_token, EOS_token, MIN_LENGTH, MAX_LENGTH, hidden_size, batch_size, n_epochs, embed_size
from data_loader import LanguagePairLoader, DateConverterLoader
from models import Seq2SeqModel
from train import train_iters
import pickle
import os.path

use_cuda = torch.cuda.is_available()

loader = LanguagePairLoader("de", "en")
eval_loader = LanguagePairLoader("de", "en", hp.source_test_file, hp.target_test_file)

input_lang, output_lang, pairs = None, None, None

_, _, eval_pairs = eval_loader.load()

if True:
    input_lang, output_lang, pairs = loader.load()
    pickle.dump(input_lang, open("input.dict", "wb"))
    pickle.dump(output_lang, open("output.dict", "wb"))
else:
    input_lang = pickle.load(open("input.dict", "rb"))
    output_lang = pickle.load(open("output.dict", "rb"))

encoder = LSTMEncoderRNN(input_lang.n_words, hidden_size, embed_size)
decoder = LSTMAttnDecoderRNN(encoder, hp.attention, hidden_size, output_lang.n_words)

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

print(encoder)
print(decoder)

seq2seq_model = Seq2SeqModel(encoder, decoder, input_lang, output_lang)

if os.path.isfile(hp.checkpoint_name):
    checkpoint = torch.load(hp.checkpoint_name) if use_cuda else torch.load(hp.checkpoint_name,
                                                                            map_location=lambda storage, loc: storage)
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]
    encoder_optimizer_state = checkpoint["encoder_optimizer"]
    decoder_optimizer_state = checkpoint["decoder_optimizer"]
    epoch = checkpoint["epoch"]
    train_loss = checkpoint["train_loss"]
    eval_loss = checkpoint["eval_loss"]
    bleu_scores = checkpoint["bleu_scores"]

    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)

    if epoch < hp.n_epochs:
        train_iters(seq2seq_model, pairs, eval_pairs,
                    encoder_optimizer_state=encoder_optimizer_state,
                    decoder_optimizer_state=decoder_optimizer_state, train_loss=train_loss, eval_loss=eval_loss,
                    bleu_scores=bleu_scores,
                    start_epoch=epoch + 1)
else:
    train_iters(seq2seq_model, pairs, eval_pairs, encoder_optimizer_state=None,
                decoder_optimizer_state=None, train_loss=[], eval_loss=[], bleu_scores=[], start_epoch=1)
