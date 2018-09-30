import hp

hp.batch_size = 128
hp.MAX_LENGTH = 100
hp.print_loss_every_iters = 100

from models import Seq2SeqModel, LSTMAttnDecoderRNN, LSTMEncoderRNN
import pickle
from train import retrain_iters
from scorer import Scorer
from data_loader import LanguagePairLoader
import torch
import random
from charac_ter import cer
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


def compute_gleu(targets, translations):
    references, translations = [[target.replace("@@ ", "").split(" ")] for target in targets], [
        t.replace("@@ ", "").split(" ") for t in translations]
    return gleu_score.corpus_gleu(references, translations)


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


def bpe_contains(sentence, word):
    return word.lower() in sentence.replace("@@ ", "").lower()


def find_improved_rareword_translation(targets, base_sentences, train_sentences, keyphrases):
    for target, base_sentence, train_sentence in zip(targets, base_sentences, train_sentences):
        for keyphrase, _ in keyphrases:
            if bpe_contains(target, keyphrase):
                if not bpe_contains(base_sentence, keyphrase) and bpe_contains(train_sentence, keyphrase):
                    print()
                    print("Target: {}".format(target.replace("@@ ", "")))
                    print("Base: {}".format(base_sentence.replace("@@ ", "")))
                    print("<")
                    print("Trained: {}".format(train_sentence.replace("@@ ", "")))


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
    def __init__(self, model, source_file, target_file, test_source_file, test_target_file,
                 raw_source_file,
                 raw_target_file, num_sentences=400,
                 batch_translate=True):
        self.model = model
        self.source_file = source_file
        self.target_file = target_file
        self.loader = LanguagePairLoader("de", "en", source_file, target_file)
        self.test_loader = LanguagePairLoader("de", "en", test_source_file, test_target_file)
        self.extractor = DomainSpecificExtractor(source_file=raw_source_file, train_source_file=hp.source_file,
                                                 train_vocab_file="train_vocab.pkl")
        self.target_extractor = DomainSpecificExtractor(source_file=raw_target_file, train_source_file=hp.source_file,
                                                        train_vocab_file="train_vocab_en.pkl")
        self.scorer = Scorer()
        self.scores = {}
        self.num_sentences = num_sentences
        self.batch_translate = batch_translate
        self.evaluate_every = 10

        self.metric_bleu_scores = {}
        self.metric_gleu_scores = {}
        self.metric_precisions = {}
        self.metric_recalls = {}

        # Plot each metric
        plt.style.use('seaborn-darkgrid')
        self.palette = sns.color_palette()

    def save_data(self):
        prefix = ("v3_batch_" if self.batch_translate else "v2_beam_") + str(self.evaluate_every) + "_"
        pickle.dump(self.metric_bleu_scores, open(prefix + "metric_bleu_scores.pkl", "wb"))
        pickle.dump(self.metric_gleu_scores, open(prefix + "metric_gleu_scores.pkl", "wb"))
        pickle.dump(self.metric_precisions, open(prefix + "metric_precisions.pkl", "wb"))
        pickle.dump(self.metric_recalls, open(prefix + "metric_recalls.pkl", "wb"))
        print("Saved all scores")

    def run(self):
        _, _, pairs = self.loader.load()
        random.seed(2018)
        random.shuffle(pairs)

        pairs = pairs[:self.num_sentences]

        sources, targets, translations = [p[0] for p in pairs], [p[1] for p in pairs], []

        keyphrases = self.extractor.extract_keyphrases(n_results=100)
        print(keyphrases)
        target_keyphrases = self.target_extractor.extract_keyphrases(n_results=100)
        print(target_keyphrases)

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

        _, _, test_pairs = self.test_loader.load()
        test_sources, test_targets, test_translations = [p[0] for p in test_pairs], [p[1] for p in test_pairs], []

        for source in test_sources:
            translation, attn, _ = self.model.translate(source)
            test_translations.append(" ".join(translation[:-1]))

        if self.batch_translate:
            translations = [t[:-6] for t in self.model.batch_translate(sources)]

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
                                     test_sources, test_targets, test_translations,
                                     need_sort=True if metric != "random" else False,
                                     reverse=sort_direction[metric] if metric != "random" else True)

                # plt.plot(x, delta_bleus, marker='', linestyle="--", color=self.palette[i], linewidth=1, alpha=0.9,
                #        label=metric)
            self.save_data()

    def shuffle_list(self, *ls):
        l = list(zip(*ls))

        random.shuffle(l)
        return zip(*l)

    def evaluate_metric(self, sources, targets, translations, scores, metric, target_keyphrases,
                        test_sources, test_targets, test_translations,
                        need_sort=True,
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

        pretraining_bleu = compute_bleu(test_targets, test_translations)
        pretraining_gleu = compute_gleu(test_targets, test_translations)

        prerecall = unigram_recall(target_keyphrases, test_targets, test_translations)
        preprecision = unigram_precision(target_keyphrases, test_targets, test_translations)

        for i in range(1, n + 1):
            print()
            print("Correcting {} of {} sentences".format(i, n))

            curr_end = i

            # Compute BLEU before training for comparison
            # pretraining_bleu = compute_bleu(targets[curr_end:], translations[curr_end:])
            # pretraining_gleu = compute_avg_gleu(targets[curr_end:], corrected_translations[curr_end:])

            # print("Training Data: {}\n : {}\n".format(sources[i - 1], targets[i - 1]))
            # Now train, and compute BLEU again
            encoder_optimizer_state, decoder_optimizer_state = retrain_iters(self.model,
                                                                             [[sources[i - 1],
                                                                               targets[i - 1]]], [],
                                                                             batch_size=1,
                                                                             encoder_optimizer_state=encoder_optimizer_state,
                                                                             decoder_optimizer_state=decoder_optimizer_state,
                                                                             n_epochs=1, learning_rate=0.0001,
                                                                             weight_decay=1e-3)

            if (i - 1) % self.evaluate_every != 0:
                continue

            corrected_translations = []
            if not self.batch_translate:
                # Translate trained model
                for j in range(0, len(test_sources)):
                    translation, _, _ = seq2seq_model.translate(test_sources[j])
                    corrected_translations.append(" ".join(translation[:-1]))
            else:
                batch_translations = self.model.batch_translate(test_sources)
                corrected_translations = [t[:-6] for t in batch_translations]

            print("Improved Rare words:")
            print(find_improved_rareword_translation(test_targets, test_translations, corrected_translations,
                                                     target_keyphrases))

            # find_improved_sentences(targets[curr_end:], translations[curr_end:], corrected_translations[curr_end:])

            # Compute posttraining BLEU
            posttraining_bleu = compute_bleu(test_targets, corrected_translations)
            posttraining_gleu = compute_gleu(test_targets, corrected_translations)

            postrecall = unigram_recall(target_keyphrases, test_targets, corrected_translations)
            postprecision = unigram_precision(target_keyphrases, test_targets, corrected_translations)
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


class AveragedMetricExperiment:
    def __init__(self, model, source_file, target_file, raw_source_file, raw_target_file, num_sentences=400):
        self.model = model
        self.source_file = source_file
        self.target_file = target_file
        self.loader = LanguagePairLoader("de", "en", source_file, target_file)
        self.extractor = DomainSpecificExtractor(source_file=raw_source_file, train_source_file=hp.source_file,
                                                 train_vocab_file="train_vocab.pkl")
        self.target_extractor = DomainSpecificExtractor(source_file=raw_target_file, train_source_file=hp.target_file,
                                                        train_vocab_file="train_vocab_en.pkl")
        self.scorer = Scorer()
        self.scores = {}
        self.num_sentences = num_sentences

        self.metric_bleu_scores = {}
        self.metric_gleu_scores = {}
        self.metric_precisions = {}
        self.metric_recalls = {}
        self.cer = {}

        # Plot each metric
        plt.style.use('seaborn-darkgrid')
        self.palette = sns.color_palette()

    def save_data(self):
        prefix = "averaged_"
        pickle.dump(self.metric_bleu_scores, open(prefix + "metric_bleu_scores.pkl", "wb"))
        pickle.dump(self.metric_gleu_scores, open(prefix + "metric_gleu_scores.pkl", "wb"))
        pickle.dump(self.metric_precisions, open(prefix + "metric_precisions.pkl", "wb"))
        pickle.dump(self.metric_recalls, open(prefix + "metric_recalls.pkl", "wb"))
        pickle.dump(self.cer, open(prefix + "metric_cer.pkl", "wb"))
        print("Saved all scores")

    def run(self):
        _, _, pairs = self.loader.load()
        random.seed(2018)
        random.shuffle(pairs)

        pairs = pairs[:self.num_sentences]

        sources, targets, translations = [p[0] for p in pairs], [p[1] for p in pairs], []

        keyphrases = self.extractor.extract_keyphrases(n_results=100)
        print(keyphrases)
        target_keyphrases = self.target_extractor.extract_keyphrases(n_results=100)
        print(target_keyphrases)

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

        metrics = [
            # "coverage_penalty",
            # "coverage_deviation_penalty",
            # "confidence",
            # "length",
            # "ap_in",
            # "ap_out",
            # "random",
            "keyphrase_score"
        ]
        n_iters = 1
        for i, metric in enumerate(metrics):
            avg_bleus = [0 for _ in range(1, 100 // (step_size * 2) + 1)]
            self.metric_bleu_scores[metric] = []
            self.metric_gleu_scores[metric] = []
            self.metric_precisions[metric] = []
            self.metric_recalls[metric] = []
            self.cer[metric] = []
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

    def evaluate_metric(self, sources, targets, translations, scores, metric, target_keyphrases,
                        need_sort=True,
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

        corrected_translations = []

        cer_improvement = []
        curr_cer = 0

        for i in range(1, n + 1):
            print()
            print("{}: Correcting {} of {} sentences".format(metric, i, n))

            curr_end = i

            # Compute BLEU before training for comparison
            pretraining_bleu = compute_bleu(targets[:curr_end], translations[:curr_end])
            pretraining_gleu = compute_gleu(targets[:curr_end], translations[:curr_end])
            prerecall = unigram_recall(target_keyphrases, targets[:curr_end], translations[:curr_end])
            preprecision = unigram_precision(target_keyphrases, targets[:curr_end], translations[:curr_end])

            precer = cer(targets[i - 1].replace("@@ ", "").split(), translations[i - 1].replace("@@ ", "").split())

            translation, _, _ = seq2seq_model.translate(sources[i - 1])
            corrected_translations.append(" ".join(translation[:-1]))

            postcer = cer(targets[i - 1].replace("@@ ", "").split(),
                          " ".join(translation[:-1]).replace("@@ ", "").split())
            curr_cer = precer - postcer
            cer_improvement.append(curr_cer)

            # Compute posttraining BLEU
            posttraining_bleu = compute_bleu(targets[:curr_end], corrected_translations)
            posttraining_gleu = compute_gleu(targets[:curr_end], corrected_translations)

            postrecall = unigram_recall(target_keyphrases, targets[:curr_end], corrected_translations)
            postprecision = unigram_precision(target_keyphrases, targets[:curr_end], corrected_translations)
            print("Delta Recall {} -> {}".format(prerecall, postrecall))
            print("Delta Precision {} -> {}".format(preprecision, postprecision))
            print("Delta BLEU: {} -> {}".format(pretraining_bleu, posttraining_bleu))
            print("Delta CER: {} -> {}".format(precer, postcer))

            self.metric_bleu_scores[metric].append((pretraining_bleu, posttraining_bleu))
            self.metric_gleu_scores[metric].append((pretraining_gleu, posttraining_gleu))
            self.metric_recalls[metric].append((prerecall, postrecall))
            self.metric_precisions[metric].append((preprecision, postprecision))

            # Now train, and compute BLEU again
            encoder_optimizer_state, decoder_optimizer_state = retrain_iters(self.model,
                                                                             [[sources[i - 1],
                                                                               targets[i - 1]]], [],
                                                                             batch_size=1,
                                                                             encoder_optimizer_state=encoder_optimizer_state,
                                                                             decoder_optimizer_state=decoder_optimizer_state,
                                                                             n_epochs=1, learning_rate=0.00005,
                                                                             weight_decay=1e-3)

        self.cer[metric] = cer_improvement
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

    return numer / denom if denom > 0 else 0


def unigram_precision(rare_words, targets, translations):
    numer, denom = 0, 0

    targets = [target.replace("@@ ", "") for target in targets]
    translations = [translation.replace("@@ ", "") for translation in translations]

    for target, translation in zip(targets, translations):
        for rare_word, _ in rare_words:
            denom += translation.count(rare_word)
            numer += min(translation.count(rare_word), target.count(rare_word))

    return numer / denom if denom > 0 else 0


seq2seq_model = load_model()
exp = MetricExperiment(seq2seq_model, "data/khresmoi.bpe.de", "data/khresmoi.bpe.en",
                       "data/khresmoi.dev.bpe.en",
                       "data/khresmoi.dev.bpe.en",
                       "data/khresmoi.tok.de",
                       "data/khresmoi.tok.en",
                       num_sentences=1000, batch_translate=False)
exp.run()
exp.plot()
exp.save_data()
