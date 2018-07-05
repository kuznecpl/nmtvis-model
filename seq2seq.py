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
from beam_search import BeamSearch

from hp import PAD_token, SOS_token, EOS_token, MIN_LENGTH, MAX_LENGTH, hidden_size, batch_size, n_epochs, embed_size
from data_loader import LanguagePairLoader, DateConverterLoader

from train import train_iters

use_cuda = torch.cuda.is_available()

loader = LanguagePairLoader("de", "en")
#loader = DateConverterLoader()
input_lang, output_lang, pairs = loader.load()

print(random.choice(pairs))


# Return a list of indexes, one for each word in the sentence, plus EOS
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def beamSearch(encoder, decoder, input_seq, beam_size=3, attentionOverrideMap=None, correctionMap=None,
               max_length=MAX_LENGTH):
    input_seqs = [indexes_from_sentence(input_lang, input_seq)]
    input_lengths = [len(seq) for seq in input_seqs]
    input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

    if use_cuda:
        input_batches = input_batches.cuda()

    encoder.train(False)
    decoder.train(False)

    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    decoder_hidden = encoder_hidden[:decoder.n_layers]

    beam_search = BeamSearch(decoder, encoder_outputs, decoder_hidden, output_lang, beam_size, attentionOverrideMap,
                             correctionMap)
    result = beam_search.search()

    encoder.train(True)
    decoder.train(True)

    return result


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


class Translation:
    def __init__(self, words=None, log_probs=None, attns=None, candidates=None):
        self.words = words
        self.log_probs = log_probs
        self.attns = attns
        self.candidates = candidates

    def slice(self):
        return Translation(self.words[1:], self.log_probs[1:], self.attns[1:], self.candidates[1:])

    @classmethod
    def from_hypothesis(cls, hypothesis):
        translation = Translation()

        translation.words = [output_lang.index2word[token] for token in hypothesis.tokens]
        translation.log_probs = hypothesis.log_probs
        translation.attns = hypothesis.attns
        translation.candidates = hypothesis.candidates

        return translation


def translate(sentence, beam_size=3, attentionOverrideMap=None, correctionMap=None):
    words, attention = evaluate(encoder1, attn_decoder1, sentence)
    hyps = beamSearch(encoder1, attn_decoder1, sentence, beam_size, attentionOverrideMap, correctionMap)

    return words, attention, [Translation.from_hypothesis(h) for h in hyps]


class Seq2SeqModel:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    def evaluate(self, sentence, max_length=MAX_LENGTH):
        input_seqs = [indexes_from_sentence(input_lang, sentence)]
        input_lengths = [len(seq) for seq in input_seqs]
        input_batches = Variable(torch.LongTensor(input_seqs), volatile=True).transpose(0, 1)

        if use_cuda:
            input_batches = input_batches.cuda()

        # Set to not-training mode to disable dropout
        self.encoder.train(False)
        self.decoder.train(False)

        # Run through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_batches, input_lengths, None)

        # Create starting vectors for decoder
        decoder_input = Variable(torch.LongTensor([SOS_token]), volatile=True)  # SOS
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]  # Use last (forward) hidden state from encoder

        if use_cuda:
            decoder_input = decoder_input.cuda()

        # Store output words and attention states
        decoded_words = []
        decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

        # Run through decoder
        for di in range(max_length):

            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_attentions[di, :decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

            # Choose top word from output
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0].item()
            if ni == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[ni])

            log_output = nn.functional.log_softmax(decoder_output)
            topv, topi = log_output.data.topk(1)
            ni = topi[0][0].item()
            log_prob = topv[0][0].item()
            print("Next decoded word")
            print("{} prob: {}".format(output_lang.index2word[ni], log_prob))

            # Next input is chosen word
            decoder_input = Variable(torch.LongTensor([ni]))
            if use_cuda: decoder_input = decoder_input.cuda()

        # Set back to training mode
        self.encoder.train(True)
        self.decoder.train(True)

        return decoded_words, decoder_attentions[:di + 1, :len(encoder_outputs)]

    def translate(self, sentence, beam_size=3, attention_override_map=None, correction_map=None):
        words, attention = self.evaluate(sentence)
        hyps = beamSearch(self.encoder, self.decoder, sentence, beam_size, attention_override_map, correction_map)

        return words, attention, [Translation.from_hypothesis(h) for h in hyps]


seq2seq_model = Seq2SeqModel(encoder1, attn_decoder1)
