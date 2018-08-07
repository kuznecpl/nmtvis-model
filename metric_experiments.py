import hp

hp.print_loss_every_iters = 10

from models import Seq2SeqModel, LSTMAttnDecoderRNN, LSTMEncoderRNN
import pickle
from train import retrain_iters
from scorer import Scorer
from data_loader import LanguagePairLoader
import torch
import random


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


def compute_bleu(targets, translations):
    import nltk
    bleu = nltk.translate.bleu_score.corpus_bleu([[target.split(" ")] for target in targets],
                                                 [t.split(" ") for t in translations])
    return bleu


loader = LanguagePairLoader("de", "en", hp.source_test_file, hp.target_test_file)
_, _, pairs = loader.load()
pairs = pairs[:1000]

seq2seq_model = load_model()

# Translate sources
sources, targets, translations = [p[0] for p in pairs], [p[1] for p in pairs], []
scores = []
scorer = Scorer()

for i, pair in enumerate(pairs):
    if i % 10 == 0:
        print("Translated {} of {}".format(i, len(pairs)))
    translation, attn, _ = seq2seq_model.translate(pair[0])
    translations.append(" ".join(translation[:-1]))
    score = scorer.coverage_penalty(attn)
    scores.append(score)

# Sort by metric
reverse = True
sorted_sentences = [(x, y, z) for _, x, y, z in sorted(zip(scores, sources, targets, translations), reverse=reverse)]
sources, targets, translations = zip(*sorted_sentences)

'''
# Shuffle
shuffled_sentences = list(zip(sources, targets, translations))
random.shuffle(shuffled_sentences)
sources, targets, translations = zip(*shuffled_sentences)
'''

# Compute base BLEU
base_bleu = compute_bleu(targets, translations)
print("Base BLEU: {}".format(base_bleu))

print(compute_bleu(targets, targets))

delta_bleus = []
step_size = 20

for i in range(step_size, len(sources), step_size):
    print("Correcting first {} sentences".format(i))

    corrected_translations = list(translations)
    # 'Correct' first i sentences
    corrected_translations[:i] = targets[:i]

    # Compute BLEU before training for comparison
    pretraining_bleu = compute_bleu(targets, corrected_translations)

    print("Pre-Training BLEU: {}".format(pretraining_bleu))

    # Now train, and compute BLEU again
    retrain_iters(seq2seq_model, list((a, b) for a, b in zip(sources[:i], corrected_translations[:i])), [],
                  batch_size=step_size,
                  n_epochs=5, learning_rate=0.00001)

    # Translate trained model
    for j in range(i + 1, len(sources)):
        translation, _, _ = seq2seq_model.translate(sources[j])
        corrected_translations[j] = " ".join(translation[:-1])

    reload_model(seq2seq_model)

    # Compute posttraining BLEU
    posttraining_bleu = compute_bleu(targets, corrected_translations)

    print("Post-Training BLEU: {}".format(posttraining_bleu))

    delta_bleu = posttraining_bleu - pretraining_bleu
    print("Delta BLEU {}".format(delta_bleu))
    print()

    delta_bleus.append(delta_bleu)

print(delta_bleus)
