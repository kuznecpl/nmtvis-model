import hp

hp.print_loss_every_iters = 5

from models import Seq2SeqModel, LSTMAttnDecoderRNN, LSTMEncoderRNN
import pickle
from train import retrain_iters
from scorer import Scorer
from data_loader import LanguagePairLoader
import torch
import random
import matplotlib
from rake_nltk import Rake
import pickle
import os.path

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from keyphrase_extractor import DomainSpecificExtractor

sns.set()
plt.style.use('seaborn-darkgrid')


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


def reload_model(seq2seq_model):
    checkpoint = torch.load(hp.checkpoint_name)
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]

    seq2seq_model.encoder.load_state_dict(encoder_state)
    seq2seq_model.decoder.load_state_dict(decoder_state)


def keyphrase_score(sentence, keyphrases):
    score = 0

    for word in sentence.split(" "):
        for keyphrase, freq in keyphrases:
            score += word.lower().count(keyphrase.lower()) * freq
    return score


extractor = DomainSpecificExtractor(source_file="data/medical.tok.de", train_source_file=hp.source_file,
                                    train_vocab_file="train_vocab.pkl")

words = extractor.extract_keyphrases()

print(words)
