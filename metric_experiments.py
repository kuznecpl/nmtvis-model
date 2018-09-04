import hp

hp.batch_size = 128
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
from rake_nltk import Rake

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.style.use('seaborn-darkgrid')
from nltk.translate import gleu_score
from keyphrase_extractor import DomainSpecificExtractor

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


def compute_avg_gleu(targets, translations):
    if not targets:
        return 0

    gleu_sum = 0
    for target, translation in zip(targets, translations):
        sentence_gleu = compute_gleu(target, translation)
        gleu_sum += sentence_gleu

    return gleu_sum / len(targets)


def compute_gleu(target, translation):
    return gleu_score.sentence_gleu([target.replace("@@ ", "").split(" ")], translation.replace("@@ ", "").split(" "))


def gleu_distr(sources, targets, translations):
    gleu_scores = []
    for target, translation in zip(targets, translations):
        gleu_score = compute_gleu(target, translation)
        gleu_scores.append(gleu_score)
    return gleu_scores


sort_direction = {"coverage_penalty": True,
                  "coverage_deviation_penalty": True,
                  "confidence": False,
                  "length": True,
                  "ap_in": True,
                  "ap_out": True,
                  "keyphrase_score": True,
                  }


def find_improved_sentences(targets, base_sentences, train_sentences):
    for target, base_sentence, train_sentence in zip(targets, base_sentences, train_sentences):
        base_gleu = compute_gleu(target, base_sentence)
        train_gleu = compute_gleu(target, train_sentence)

        if train_gleu > base_gleu:
            print()
            print("Target: {}".format(target))
            print("Base: {}".format(base_sentence))
            print("<")
            print("Trained: {}".format(train_sentence))


class MetricExperiment:
    def __init__(self, model, source_file, target_file, raw_source_file, raw_target_file, num_sentences=400,
                 batch_translate=True):
        self.model = model
        self.source_file = source_file
        self.target_file = target_file
        self.loader = LanguagePairLoader("de", "en", source_file, target_file)
        self.extractor = DomainSpecificExtractor(source_file=raw_source_file, train_source_file=hp.source_file,
                                                 train_vocab_file="train_vocab.pkl")
        self.target_extractor = DomainSpecificExtractor(source_file=raw_target_file, train_source_file=hp.source_file,
                                                        train_vocab_file="train_vocab_en.pkl")
        self.scorer = Scorer()
        self.scores = {}
        self.num_sentences = num_sentences
        self.batch_translate = batch_translate

        self.metric_bleu_scores = {}
        self.metric_gleu_scores = {}
        self.metric_precisions = {}
        self.metric_recalls = {}

        # Plot each metric
        plt.style.use('seaborn-darkgrid')
        self.palette = sns.color_palette()

    def save_data(self):
        prefix = "batch_" if self.batch_translate else "beam_"
        pickle.dump(self.metric_bleu_scores, open(prefix + "metric_bleu_scores.pkl", "wb"))
        pickle.dump(self.metric_gleu_scores, open(prefix + "metric_gleu_scores.pkl", "wb"))
        pickle.dump(self.metric_precisions, open(prefix + "metric_precisions.pkl", "wb"))
        pickle.dump(self.metric_recalls, open(prefix + "metric_recalls.pkl", "wb"))
        print("Saved all scores")

    def run(self):
        _, _, pairs = self.loader.load()
        random.seed(100)
        random.shuffle(pairs)
        pairs = pairs[:self.num_sentences]

        sources, targets, translations = [p[0] for p in pairs], [p[1] for p in pairs], []

        keyphrases = self.extractor.extract_keyphrases(n_results=50)
        print(keyphrases)
        target_keyphrases = self.target_extractor.extract_keyphrases(n_results=50)

        for i, pair in enumerate(pairs):
            if i % 10 == 0:
                print("Translated {} of {}".format(i, len(pairs)))
            translation, attn, _ = self.model.translate(pair[0])
            translations.append(" ".join(translation[:-1]))

            metrics_scores = self.scorer.compute_scores(pair[0], " ".join(translation[:-1]), attn, keyphrases)
            for metric in metrics_scores:
                if metric not in self.scores:
                    self.scores[metric] = []
                self.scores[metric].append(metrics_scores[metric])

        if self.batch_translate:
            translations = [t[:-6] for t in self.model.batch_translate([pair[0] for pair in pairs])]

        x = range(0, len(pairs) // 2)

        metrics = [
            "coverage_penalty",
            "coverage_deviation_penalty",
            "confidence",
            "length",
            "ap_in",
            "ap_out",
            "random",
            "keyphrase_score"
        ]
        n_iters = 1
        for i, metric in enumerate(metrics):
            avg_bleus = [0 for _ in range(1, 100 // (step_size * 2) + 1)]
            self.metric_bleu_scores[metric] = []
            self.metric_gleu_scores[metric] = []
            self.metric_precisions[metric] = []
            self.metric_recalls[metric] = []
            for j in range(n_iters):
                self.evaluate_metric(sources, targets, translations,
                                     self.scores[metric] if metric != "random" else [],
                                     metric,
                                     target_keyphrases,
                                     need_sort=True if metric != "random" else False,
                                     reverse=sort_direction[metric] if metric != "random" else True)

                # plt.plot(x, delta_bleus, marker='', linestyle="--", color=self.palette[i], linewidth=1, alpha=0.9,
                #        label=metric)
            self.save_data()

    def shuffle_list(self, *ls):
        l = list(zip(*ls))

        random.shuffle(l)
        return zip(*l)

    def evaluate_metric(self, sources, targets, translations, scores, metric, target_keyphrases, need_sort=True,
                        reverse=False):
        print("Evaluating {}".format(metric))
        base_bleu = compute_bleu(targets, translations)
        print("Base BLEU: {}".format(base_bleu))
        # Sort by metric
        if need_sort:
            sorted_sentences = [(x, y, z) for _, x, y, z in
                                sorted(zip(scores, sources, targets, translations), reverse=reverse)]
            sources, targets, translations = zip(*sorted_sentences)
        else:
            sources, targets, translations = self.shuffle_list(sources, targets, translations)

        n = len(sources)
        encoder_optimizer_state, decoder_optimizer_state = None, None

        print("Translations")
        print(translations[:5])

        for i in range(1, n // 2):
            print()
            print("Correcting first {} sentences".format(i))

            curr_end = i
            corrected_translations = list(translations)

            # prefix_bleu = compute_bleu(targets[:curr_end], translations[:curr_end])

            # Compute BLEU before training for comparison
            pretraining_bleu = compute_bleu(targets[curr_end:], translations[curr_end:])
            pretraining_gleu = compute_avg_gleu(targets[curr_end:], corrected_translations[curr_end:])

            prerecall = unigram_recall(target_keyphrases, targets[curr_end:], corrected_translations[curr_end:])
            preprecision = unigram_precision(target_keyphrases, targets[curr_end:], corrected_translations[curr_end:])
            pre_gleu_scores = gleu_distr(sources, targets[curr_end:], translations[curr_end:])

            print("Training Data: {}\n : {}\n".format(sources[i - 1], targets[i - 1]))
            # Now train, and compute BLEU again
            encoder_optimizer_state, decoder_optimizer_state = retrain_iters(self.model,
                                                                             [[sources[i - 1],
                                                                               targets[i - 1]]], [],
                                                                             batch_size=1,
                                                                             encoder_optimizer_state=encoder_optimizer_state,
                                                                             decoder_optimizer_state=decoder_optimizer_state,
                                                                             n_epochs=1, learning_rate=0.0001,
                                                                             weight_decay=1e-3)

            if not self.batch_translate:
                # Translate trained model
                for j in range(curr_end, len(sources)):
                    translation, _, _ = seq2seq_model.translate(sources[j])
                    corrected_translations[j] = " ".join(translation[:-1])
            else:
                batch_translations = self.model.batch_translate(sources)[curr_end:]
                corrected_translations[curr_end:] = [t[:-6] for t in batch_translations]

            # find_improved_sentences(targets[curr_end:], translations[curr_end:], corrected_translations[curr_end:])

            # Compute posttraining BLEU
            posttraining_bleu = compute_bleu(targets[curr_end:], corrected_translations[curr_end:])
            posttraining_gleu = compute_avg_gleu(targets[curr_end:], corrected_translations[curr_end:])
            post_gleu_scores = gleu_distr(sources, targets[curr_end:], corrected_translations[curr_end:])

            postrecall = unigram_recall(target_keyphrases, targets[curr_end:], corrected_translations[curr_end:])
            postprecision = unigram_precision(target_keyphrases, targets[curr_end:], corrected_translations[curr_end:])
            print("Delta Recall {} -> {}".format(prerecall, postrecall))
            print("Delta Precision {} -> {}".format(preprecision, postprecision))
            print("Delta BLEU: {} -> {}".format(pretraining_bleu, posttraining_bleu))

            delta_bleu = posttraining_bleu / pretraining_bleu * 100 - 100

            self.metric_bleu_scores[metric].append((pretraining_bleu, posttraining_bleu))
            self.metric_gleu_scores[metric].append((pretraining_gleu, posttraining_gleu))
            self.metric_recalls[metric].append((prerecall, postrecall))
            self.metric_precisions[metric].append((preprecision, postprecision))

        reload_model(self.model)
        return None

    def plot(self):
        plt.xlabel('% Corrected Sentences')
        plt.ylabel('Δ BLEU')
        # Add titles
        plt.title("BLEU Change for Metrics", loc='center', fontsize=12, fontweight=0)
        # Add legend
        plt.legend(loc='lower right', ncol=1)
        plt.savefig('bleu_deltas.png')


def unigram_recall(rare_words, targets, translations):
    numer, denom = 0, 0

    targets = [target.replace("@@ ", "") for target in targets]
    translations = [translation.replace("@@ ", "") for translation in translations]

    for target, translation in zip(targets, translations):
        for rare_word, _ in rare_words:
            denom += target.count(rare_word)
            numer += min(translation.count(rare_word), target.count(rare_word))

    return numer / denom


def unigram_precision(rare_words, targets, translations):
    numer, denom = 0, 0

    targets = [target.replace("@@ ", "") for target in targets]
    translations = [translation.replace("@@ ", "") for translation in translations]

    for target, translation in zip(targets, translations):
        for rare_word, _ in rare_words:
            denom += translation.count(rare_word)
            numer += min(translation.count(rare_word), target.count(rare_word))

    return numer / denom


seq2seq_model = load_model()
exp = MetricExperiment(seq2seq_model, "data/khresmoi.bpe.de", "data/khresmoi.bpe.en", "data/khresmoi.tok.de",
                       "data/khresmoi.tok.en", num_sentences=5000, batch_translate=False)
exp.run()
exp.plot()
exp.save_data()

'''
# Shuffle
avg_bleus = [0 for _ in range(1, 100 // (step_size * 2) + 1)]
for j in range(n_iters):
    shuffled_sentences = list(zip(sources, targets, translations))
    random.shuffle(shuffled_sentences)
    sources, targets, translations = zip(*shuffled_sentences)
    delta_bleus = evaluate_metric(sources, targets, translations, None, "random", need_sort=False)
    avg_bleus = [avg + delta for (avg, delta) in zip(avg_bleus, delta_bleus)]
delta_bleus = [0] + [b / n_iters * 100 for b in delta_bleus]

plt.plot(x, delta_bleus, marker='', color=palette(7), linewidth=1, alpha=0.9, label="random")

plt.xlabel('% Corrected Sentences')
plt.ylabel('Δ BLEU')
# Add titles
plt.title("BLEU Change for Metrics", loc='center', fontsize=12, fontweight=0)
# plt.xticks([i for i in range(1, 21, 1)])
# Add legend
plt.legend(loc='upper left', ncol=1)
plt.savefig('bleu_deltas.png')
'''
