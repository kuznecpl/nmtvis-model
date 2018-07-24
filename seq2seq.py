from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
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
# loader = DateConverterLoader()

input_lang, output_lang, pairs = None, None, None
input_lang, output_lang, pairs = loader.load()

'''
    pickle.dump(input_lang, open("input.dict", "wb"))
    pickle.dump(output_lang, open("output.dict", "wb"))
    pickle.dump(pairs, open("pairs.data", "wb"))
else:
    input_lang = pickle.load(open("input.dict", "rb"))
    output_lang = pickle.load(open("output.dict", "rb"))
    pairs = pickle.load(open("pairs.data", "rb"))'''

print(random.choice(pairs))

encoder1 = None
attn_decoder1 = None

encoder1 = LSTMEncoderRNN(input_lang.n_words, hidden_size, embed_size)
attn_decoder1 = LSTMAttnDecoderRNN(encoder1, hp.attention, hidden_size, output_lang.n_words)

print(attn_decoder1)

if not os.path.isfile(hp.encoder_name) or not os.path.isfile(hp.decoder_name):
    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    train_iters(encoder1, attn_decoder1, input_lang, output_lang, pairs)

    torch.save(encoder1.state_dict(), hp.encoder_name)
    torch.save(attn_decoder1.state_dict(), hp.decoder_name)
else:
    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()
        encoder1.load_state_dict(torch.load(hp.encoder_name))
        attn_decoder1.load_state_dict(torch.load(hp.decoder_name))
    else:
        encoder1.load_state_dict(torch.load(hp.encoder_name, map_location=lambda storage, loc: storage))
        attn_decoder1.load_state_dict(torch.load(hp.decoder_name, map_location=lambda storage, loc: storage))

seq2seq_model = Seq2SeqModel(encoder1, attn_decoder1, input_lang, output_lang)

from train import eval_bleu

# eval_bleu(encoder1, attn_decoder1, input_lang, output_lang, 1)
