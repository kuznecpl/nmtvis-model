import hp

hp.MAX_LENGTH = 80
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
from charac_ter import cer
from scipy.stats.stats import pearsonr

matplotlib.use('Agg')
import matplotlib.pylab as plt

plt.rcParams["axes.titlesize"] = 15
import seaborn as sns

plt.style.use('seaborn-darkgrid')
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


def compute_cter(target, translation):
    return cer(target.replace("@@ ", "").split(" "), translation.replace("@@ ", "").split(" "))


name_map = {"ap_in": r"$\mathregular{Absentmindedness\ Penalty_{in}}$",
            "ap_out": r"$\mathregular{Absentmindedness\ Penalty_{out}}$", "length": "Length",
            "coverage_penalty": "Coverage Penalty",
            "coverage_deviation_penalty": "Coverage Deviation Penalty", "confidence": "Confidence",
            "length_deviation": "Length Deviation",
            "keyphrase_score": "Keyphrase Score",
            "shortness_penalty": "Shortness Penalty"}

sort_direction = {"coverage_penalty": True,
                  "coverage_deviation_penalty": True,
                  "confidence": False,
                  "length": False,
                  "ap_in": True,
                  "ap_out": True,
                  "keyphrase_score": True,
                  }

metrics = [
    "coverage_penalty",
    "coverage_deviation_penalty",
    "confidence",
    "length",
    "ap_in",
    "ap_out"
]


class CorrelationExperiment:
    def __init__(self, model, source_file, target_file, num_sentences=1000):
        self.model = model
        self.source_file = source_file
        self.target_file = target_file
        self.scorer = Scorer()
        self.num_sentences = num_sentences
        self.metric_to_gleu = {}
        self.all_gleu_scores = []
        self.metric_to_bad = {}
        self.bad_count = {}
        # self.threshold = 0.2826 - 0.167362
        self.threshold = 0.6

    def plot_correlation(self, filename):
        f, axes = plt.subplots(2, 3, sharey=True)
        f.set_figheight(8)
        f.set_figwidth(12)
        axes = np.reshape(axes, (6,))

        for i, metric in enumerate(metrics):
            x, y = [], []
            x_min = float('inf')
            x_max = float('-inf')

            x_temp = []

            for score in self.metric_to_gleu[metric]:
                values = self.metric_to_gleu[metric][score]
                x_temp += [score] * len(values)

            bad_count = 0
            score_gleu_tuples = []
            gleus = []
            for score in self.metric_to_gleu[metric]:
                for v in self.metric_to_gleu[metric][score]:
                    if v >= self.threshold:
                        bad_count += 1
                    gleus.append(v)
                score_gleu_tuples += [(score, v) for v in self.metric_to_gleu[metric][score]]
                values = self.metric_to_gleu[metric][score]
                x_min = min(x_min, score)
                x_max = max(x_max, score)
                x += [score] * len(values)
                y += values
                # plt.scatter([score] * len(values), values, color=palette(0), alpha=0.5)

            print(bad_count)
            print("Median {}".format(np.median(gleus)))
            print("Std {}".format(np.std(gleus)))
            self.bad_count[metric] = bad_count

            score_gleu_tuples = sorted(score_gleu_tuples, key=lambda x: x[0], reverse=sort_direction[metric])

            self.metric_to_bad[metric] = score_gleu_tuples

            b, m = P.polyfit(x, y, 1)
            axes[i].set_ylim(-0.1, 1.1)
            if metric == "ap_out":
                axes[i].set_xlim(0, 2.5)
            if metric == "shortness_penalty":
                axes[i].set_xlim(0, 1)
            corr, p_val = pearsonr(x, y)

            axes[i].text(0.05, 0.95, "r = {0:.2f}".format(corr.item()), transform=axes[i].transAxes, va="top",
                         fontsize=13, weight="bold")

            axes[i].set_title(name_map[metric],
                              {'fontsize': 15, 'horizontalalignment': 'left'}, "left")
            sns.regplot(x, y, ax=axes[i], scatter_kws={'alpha': 0.2}, order=1)
            # plt.plot(np.asarray([x_min, x_max]), b + m * np.asarray([x_min, x_max]), '-')

        axes[0].set(ylabel="CharacTER")
        axes[3].set(ylabel="CharacTER")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_bad(self, filename):
        f, axes = plt.subplots(2, 3, sharey=True)
        f.set_figheight(8)
        f.set_figwidth(12)
        axes = np.reshape(axes, (6,))
        palette = sns.color_palette()

        metric_percentage = {}
        for metric in metrics:
            bad_percentage = []
            curr_bad_count = 0
            for score, gleu in self.metric_to_bad[metric]:
                if gleu >= self.threshold:
                    curr_bad_count += 1
                bad_percentage.append(curr_bad_count / self.bad_count[metric])
            metric_percentage[metric] = bad_percentage

        print(len([metric_percentage[m] for m in metric_percentage]))

        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i + 1)
            bad_percentage = metric_percentage[metric]

            percentiles = [0.25, 0.5, 0.75]
            indices = []
            for perc in percentiles:
                indices.append(next(x[0] for x in enumerate(bad_percentage) if x[1] >= perc) / len(bad_percentage))
            print(metric)
            print(indices)

            n = len(bad_percentage)
            x = [100 * i / n for i in range(1, n + 1)]
            plt.plot(x, [100 * p for p in bad_percentage], color=palette[i], linewidth=2, alpha=0.9)
            plt.plot(x, x, marker='', linestyle="--", color='black',
                     linewidth=1.5, alpha=0.9)

            for m in metrics:
                plt.plot([100 * i / n for i in range(1, len(metric_percentage[m]) + 1)],
                         [100 * p for p in metric_percentage[m]], marker='', color='grey', linewidth=1, alpha=0.3)

            if i + 1 not in [1, 4]:
                plt.tick_params(labelleft='off')

            plt.yticks([0, 25, 50, 75, 100])
            plt.xticks([0, 25, 50, 75, 100])
            # Add title
            plt.title(name_map[metric], loc='left', fontsize=15, fontweight=0)
            if i + 1 == 5:
                plt.xlabel("Percentile Threshold", fontsize=15)
            if i + 1 == 4 or i + 1 == 1:
                plt.ylabel("% Covered", fontsize=15)
        plt.tight_layout()
        plt.savefig(filename)
        print("saved bad")
        plt.close()

    def plot_distr(self, filename):
        palette = sns.color_palette()
        f, axes = plt.subplots(2, 3)
        f.set_figheight(8)
        f.set_figwidth(12)
        axes = np.reshape(axes, (6,))

        bins_map = {"length": 60}
        metrics = [
            "coverage_penalty",
            "coverage_deviation_penalty",
            "confidence",
            "length",
            "ap_in",
            "ap_out"
        ]

        for i, metric in enumerate(metrics):
            metric_scores = []
            for value in self.metric_to_gleu[metric]:
                metric_scores += len(self.metric_to_gleu[metric][value]) * [value]

            if metric == "length":
                bins_map["length"] = max(metric_scores) - min(metric_scores) + 1
                # axes[i].set_xlim(0, 61)

            axes[i].set_title(name_map[metric], {'fontsize': 15, 'horizontalalignment': 'left'}, "left")
            bins = bins_map[metric] if metric in bins_map else None
            dist_ax = sns.distplot(metric_scores, ax=axes[i], color=palette[i], bins=bins, hist_kws={"alpha": 0.2})
            ax2 = dist_ax.twinx()
            sns.boxplot(x=metric_scores, ax=ax2, color=palette[i])
            ax2.set(ylim=(-5, 5))
        plt.tight_layout()
        plt.savefig(filename)
        plt.clf()

        f.set_figheight(4)
        f.set_figwidth(4)
        sns.distplot(self.all_gleu_scores)
        plt.tight_layout()
        plt.savefig("gleu_dist.png")
        plt.clf()

    def run(self):
        loader = LanguagePairLoader("de", "en", self.source_file, self.target_file)
        _, _, pairs = loader.load()

        pairs = pairs[:self.num_sentences]
        # Translate sources
        sources, targets, translations = [p[0] for p in pairs], [p[1] for p in pairs], []

        extractor = DomainSpecificExtractor(source_file="data/khresmoi.tok.de",
                                            train_source_file=hp.source_file,
                                            train_vocab_file="train_vocab.pkl")
        keyphrases = extractor.extract_keyphrases(n_results=100)
        print(keyphrases)

        for i, pair in enumerate(pairs):
            if i % 10 == 0:
                print("Translated {} of {}".format(i, len(pairs)))
            translation, attn, _ = self.model.translate(pair[0], beam_size=1)
            translations.append(" ".join(translation[:-1]))
            scores = self.scorer.compute_scores(pair[0], " ".join(translation), attn, keyphrases)

            for metric in scores:
                if metric == "coverage_penalty" and scores[metric] > 80:
                    continue
                if metric == "keyphrase_score" and scores[metric] == 0:
                    continue

                if not metric in self.metric_to_gleu:
                    self.metric_to_gleu[metric] = {}
                if not scores[metric] in self.metric_to_gleu[metric]:
                    self.metric_to_gleu[metric][scores[metric]] = []
                gleu = compute_cter(pair[1], " ".join(translation[:-1]))
                self.all_gleu_scores.append(gleu)
                self.metric_to_gleu[metric][scores[metric]].append(gleu)


seq2seq_model = load_model()

exp1 = CorrelationExperiment(seq2seq_model,
                             # hp.source_eval_file,
                             # hp.target_eval_file,
                             "data/khresmoi.bpe.de",
                             "data/khresmoi.bpe.en",
                             num_sentences=1000)
# exp1 = CorrelationExperiment(seq2seq_model, hp.source_eval_file, hp.target_eval_file, num_sentences=1000)
exp1.run()
exp1.plot_distr("metrics_dist.png")
exp1.plot_correlation("corr_medical.png")
exp1.plot_bad("bad_progression.png")
