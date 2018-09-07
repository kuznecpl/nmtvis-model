import hp
from models import Seq2SeqModel, LSTMAttnDecoderRNN, LSTMEncoderRNN
import pickle
from train import retrain_iters
from scorer import Scorer
from data_loader import LanguagePairLoader
import torch
import torch.nn as nn
import time
from torch.autograd import Variable
import random
from nltk.translate import gleu_score
from keyphrase_extractor import DomainSpecificExtractor
from train import indexes_from_sentence, pad_seq

use_cuda = torch.cuda.is_available()
step_size = 100


def load_model():
    checkpoint = torch.load(hp.checkpoint_name)

    input_lang = pickle.load(open("input.dict", "rb"))
    output_lang = pickle.load(open("output.dict", "rb"))

    encoder = LSTMEncoderRNN(input_lang.n_words, hp.hidden_size, hp.embed_size)
    decoder = LSTMAttnDecoderRNN(encoder, hp.attention, hp.hidden_size, output_lang.n_words)

    encoder = encoder.cuda()
    decoder = decoder.cuda()

    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]

    encoder.load_state_dict(encoder_state)
    decoder.load_state_dict(decoder_state)

    seq2seq_model = Seq2SeqModel(encoder, decoder, input_lang, output_lang)

    return seq2seq_model


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def next_batch(model, batch_it):
    input_seqs = []

    # Choose random pairs
    for pair in next(batch_it):
        input_seqs.append(indexes_from_sentence(model.input_lang, pair))

    # Zip into pairs, sort by length (descending), unzip
    input_seqs = sorted(input_seqs, key=lambda p: len(p), reverse=True)

    # For input and target sequences, get array of lengths and pad with 0s to max length
    input_lengths = [len(s) for s in input_seqs]
    input_padded = [pad_seq(s, max(input_lengths)) for s in input_seqs]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = Variable(torch.LongTensor(input_padded)).transpose(0, 1)

    if use_cuda:
        input_var = input_var.cuda()

    return input_var, input_lengths


def idx_to_sentence(model, idx):
    eos_id = model.output_lang.word2index[hp.EOS_text]
    eos = idx.index(eos_id) if eos_id in idx else len(idx)
    return " ".join(model.output_lang.index2word[i] for i in idx[:eos])


model = load_model()
"""
data = ["Ich bin da .", "Das ist gut ."]
batch_it = batch(data, len(data))
input_batches, input_lengths = next_batch(model, batch_it)

outputs = model.batch_translate(input_batches, input_lengths, 10, len(data))

idx = outputs.topk(1, dim=2)[1].squeeze(2).transpose(1, 0).cpu().numpy().tolist()
print([idx_to_sentence(model, i) for i in idx])
"""