from __future__ import unicode_literals, print_function, division
import random
import torch
import torch.nn as nn
from torch.autograd import Variable

import sys

sys.path.insert(0, './myseq2seq')
for p in sys.path:
    print(p)

from models import AttnDecoderRNN, EncoderRNN

from hp import PAD_token, SOS_token, EOS_token, MIN_LENGTH, MAX_LENGTH, hidden_size, batch_size, n_epochs, embed_size
from data_loader import LanguagePairLoader, DateConverterLoader
from models import Seq2SeqModel
from train import train_iters

use_cuda = torch.cuda.is_available()

loader = LanguagePairLoader("de", "en")
# loader = DateConverterLoader()
input_lang, output_lang, pairs = loader.load()

print(random.choice(pairs))

encoder1 = None
attn_decoder1 = None

import os.path

encoder1 = EncoderRNN(input_lang.n_words, hidden_size, embed_size)
attn_decoder1 = AttnDecoderRNN("dot", hidden_size, output_lang.n_words)

if not os.path.isfile("encoder_state.pt"):
    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    train_iters(encoder1, attn_decoder1, input_lang, output_lang, pairs, n_epochs=n_epochs, batch_size=batch_size)

    torch.save(encoder1.state_dict(), "encoder_state.pt")
    torch.save(attn_decoder1.state_dict(), "attn_decoder_state.pt")
else:
    if use_cuda:
        encoder1 = encoder1.cuda()
        attn_decoder1 = attn_decoder1.cuda()
        encoder1.load_state_dict(torch.load("encoder_state.pt"))
        attn_decoder1.load_state_dict(torch.load("attn_decoder_state.pt"))
    else:
        encoder1.load_state_dict(torch.load("encoder_state.pt", map_location=lambda storage, loc: storage))
        attn_decoder1.load_state_dict(torch.load("attn_decoder_state.pt", map_location=lambda storage, loc: storage))

seq2seq_model = Seq2SeqModel(encoder1, attn_decoder1, input_lang, output_lang)

from train import eval_bleu

eval_bleu(encoder1, attn_decoder1, input_lang, output_lang, 1)
