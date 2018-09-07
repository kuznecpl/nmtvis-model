import hp

hp.MAX_LENGTH = 100
hp.print_loss_every_iters = 1

from models import Seq2SeqModel, LSTMAttnDecoderRNN, LSTMEncoderRNN
import pickle
from train import retrain_iters
from scorer import Scorer
from data_loader import LanguagePairLoader
import torch
import random
import matplotlib


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


def save_model(model):
    checkpoint = torch.load(hp.checkpoint_name)
    checkpoint["encoder"] = model.encoder.state_dict()
    checkpoint["decoder"] = model.decoder.state_dict()
    torch.save(checkpoint, "checkpoint-finetune.pt")


def reload_model(seq2seq_model):
    checkpoint = torch.load(hp.checkpoint_name)
    encoder_state = checkpoint["encoder"]
    decoder_state = checkpoint["decoder"]

    seq2seq_model.encoder.load_state_dict(encoder_state)
    seq2seq_model.decoder.load_state_dict(decoder_state)


def compute_bleu(targets, translations):
    import nltk

    references, translations = [[target.replace("@@ ", "").split(" ")] for target in targets], [
        t.replace("@@ ", "").split(" ") for t in translations]

    bleu = nltk.translate.bleu_score.corpus_bleu(references, translations)
    return bleu


model = load_model()
loader = LanguagePairLoader("de", "en", "data/auto.bpe.de", "data/auto.bpe.en")
_, _, pairs = loader.load()

sources, targets = [p[0] for p in pairs], [p[1] for p in pairs]
translations = []
for source in sources:
    translation, _, _ = model.translate(source)
    translations.append(" ".join(translation[:-1]))

print("BLEU {}".format(compute_bleu(targets, translations)))

encoder_optimizer_state, decoder_optimizer_state = retrain_iters(model,
                                                                 pairs, [],
                                                                 batch_size=20,
                                                                 encoder_optimizer_state=None,
                                                                 decoder_optimizer_state=None,
                                                                 n_epochs=5, learning_rate=0.001,
                                                                 weight_decay=1e-3)

pre_translations = translations
translations = []
for i, source in enumerate(sources):
    translation, _, _ = model.translate(source)
    translations.append(" ".join(translation[:-1]))

    if pre_translations[i] != " ".join(translation[:-1]):
        print("#{} {}".format(i, source))
        print(">")
        print(pre_translations[i])
        print(" ".join(translation[:-1]))
        print()
print("BLEU {}".format(compute_bleu(targets, translations)))

save_model(model)
