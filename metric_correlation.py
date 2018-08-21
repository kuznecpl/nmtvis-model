import hp

hp.MAX_LENGTH = 60
from models import Seq2SeqModel, LSTMAttnDecoderRNN, LSTMEncoderRNN
import pickle
from scorer import Scorer
from data_loader import LanguagePairLoader
import torch
import nltk
from nltk.translate import gleu_score
import matplotlib
from numpy.polynomial import polynomial as P
import numpy as np
from scipy.stats.stats import pearsonr

matplotlib.use('Agg')
import matplotlib.pylab as plt
import seaborn as sns
import itertools
from keyphrase_extractor import DomainSpecificExtractor


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


def compute_gleu(target, translation):
    gleu = gleu_score.sentence_gleu([target.replace("@@ ", "").split(" ")], translation.replace("@@ ", "").split(" "))
    return gleu


loader = LanguagePairLoader("de", "en", hp.source_test_file, hp.target_test_file)
_, _, pairs = loader.load()
pairs = pairs[:1000]

seq2seq_model = load_model()

# Translate sources
sources, targets, translations = [p[0] for p in pairs], [p[1] for p in pairs], []
scorer = Scorer()

extractor = DomainSpecificExtractor(source_file="data/wmt14/newstest2016.tok.de", train_source_file=hp.source_file,
                                    train_vocab_file="train_vocab.pkl")
keyphrases = extractor.extract_keyphrases(n_results=100)
print(keyphrases)

metric_to_gleu = {}
all_gleu_scores = []

plt.style.use('seaborn-darkgrid')
palette = sns.color_palette()

for i, pair in enumerate(pairs):
    if i % 10 == 0:
        print("Translated {} of {}".format(i, len(pairs)))
    translation, attn, _ = seq2seq_model.translate(pair[0])
    translations.append(" ".join(translation[:-1]))
    scores = scorer.compute_scores(pair[0], " ".join(translation), attn, keyphrases)

    if scores["keyphrase_score"] == 0:
        continue

    for metric in scores:
        if metric == "length":
            continue

        if not metric in metric_to_gleu:
            metric_to_gleu[metric] = {}
        if not scores[metric] in metric_to_gleu[metric]:
            metric_to_gleu[metric][scores[metric]] = []
        gleu = compute_gleu(pair[1], " ".join(translation[:-1]))
        all_gleu_scores.append(gleu)
        metric_to_gleu[metric][scores[metric]].append(gleu)

f, axes = plt.subplots(2, 3)
f.set_figheight(8)
f.set_figwidth(12)
axes = np.reshape(axes, (6,))

name_map = {"ap_in": r"$\mathregular{Absentmindedness\ Penalty_{IN}}$",
            "ap_out": r"$\mathregular{Absentmindedness\ Penalty_{OUT}}$", "length": "Length",
            "coverage_penalty": "Coverage Penalty",
            "coverage_deviation_penalty": "Coverage Deviation Penalty", "confidence": "Confidence",
            "length_deviation": "Length Deviation",
            "keyphrase_score": "Keyphrase Score"}
bins_map = {"length": 60}
for i, metric in enumerate(metric_to_gleu):
    metric_scores = []
    for value in metric_to_gleu[metric]:
        metric_scores += len(metric_to_gleu[metric][value]) * [value]

    if metric == "coverage_penalty":
        axes[i].set_xlim(-0.1, 1.5)
    if metric == "length":
        axes[i].set_xlim(-1, 61)

    axes[i].set_title(name_map[metric])
    bins = bins_map[metric] if metric in bins_map else None
    dist_ax = sns.distplot(metric_scores, ax=axes[i], color=palette[i], bins=bins, hist_kws={"alpha": 0.2})
    ax2 = dist_ax.twinx()
    sns.boxplot(x=metric_scores, ax=ax2, color=palette[i])
    ax2.set(ylim=(-5, 5))

plt.tight_layout()
plt.savefig("metrics_dist.png")
plt.clf()

f.set_figheight(4)
f.set_figwidth(4)
sns.distplot(all_gleu_scores)
plt.tight_layout()
plt.savefig("gleu_dist.png")
plt.clf()

f, axes = plt.subplots(2, 3, sharey=True)
f.set_figheight(8)
f.set_figwidth(12)
axes = np.reshape(axes, (6,))

for i, metric in enumerate(metric_to_gleu):
    x, y = [], []
    x_min = float('inf')
    x_max = float('-inf')
    for score in metric_to_gleu[metric]:
        if metric == "coverage_penalty" and score > 1:
            continue
        values = metric_to_gleu[metric][score]
        x_min = min(x_min, score)
        x_max = max(x_max, score)
        x += [score] * len(values)
        y += values
        # plt.scatter([score] * len(values), values, color=palette(0), alpha=0.5)
    b, m = P.polyfit(x, y, 1)
    axes[i].set_ylim(-0.1, 1.1)
    if metric == "ap_out":
        axes[i].set_xlim(0, 2.5)
    corr, p_val = pearsonr(x, y)
    axes[i].set_title(name_map[metric] + " (r = {0:.2f})".format(corr.item()))
    sns.regplot(x, y, ax=axes[i], scatter_kws={'alpha': 0.3})
    # plt.plot(np.asarray([x_min, x_max]), b + m * np.asarray([x_min, x_max]), '-')

axes[0].set(ylabel="GLEU")
axes[3].set(ylabel="GLEU")
plt.tight_layout()
plt.savefig("corr.png".format(metric))
plt.close()
