import hp

hp.MAX_LENGTH = 100
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

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
plt.style.use('seaborn-darkgrid')
from nltk.translate import gleu_score
from keyphrase_extractor import DomainSpecificExtractor

step_size = 10


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


def evaluate_metric(sources, targets, translations, scores, metric, need_sort=True, reverse=False):
    print("Evaluating {}".format(metric))
    # Sort by metric
    if need_sort:
        sorted_sentences = [(x, y, z) for _, x, y, z in
                            sorted(zip(scores, sources, targets, translations), reverse=reverse)]
        sources, targets, translations = zip(*sorted_sentences)

    # Compute base BLEU
    base_bleu = compute_bleu(targets, translations)
    print("Base BLEU: {}".format(base_bleu))

    bleu_scores = [base_bleu]

    delta_bleus = []
    delta_recalls = []

    n = len(sources)

    for i in range(1, 100 // (step_size * 2) + 1):
        print("Correcting first {}% sentences".format(i * step_size))

        curr_end = i * n // (100 // step_size)

        corrected_translations = list(translations)
        # 'Correct' first i sentences
        corrected_translations[: curr_end] = targets[: curr_end]

        # Compute BLEU before training for comparison
        pretraining_bleu = compute_bleu(targets, corrected_translations)
        bleu_scores.append(pretraining_bleu)

        prerecall = unigram_recall(target_keyphrases, targets[curr_end:], corrected_translations[curr_end:])
        preprecision = unigram_precision(target_keyphrases, targets[curr_end:], corrected_translations[curr_end:])
        print("Pre Recall {}".format(prerecall))
        print("Pre Precision {}".format(preprecision))

        pre_gleu_scores = gleu_distr(sources, targets[curr_end:], translations[curr_end:])
        # plt.figure(0)
        # sns.distplot(pre_gleu_scores, bins=20, kde=True, kde_kws={"bw": .2}, hist_kws={"align": "mid", "rwidth": 0.75})
        print("Pre-Training BLEU: {}".format(pretraining_bleu))

        # Now train, and compute BLEU again
        retrain_iters(seq2seq_model,
                      list((a, b) for a, b in zip(sources[: curr_end], corrected_translations[: curr_end])), [],
                      batch_size=n // (100 // step_size),
                      n_epochs=20, learning_rate=0.00001, weight_decay=1e-3)

        # Translate trained model
        for j in range(curr_end, len(sources)):
            translation, _, _ = seq2seq_model.translate(sources[j])
            corrected_translations[j] = " ".join(translation[:-1])

        reload_model(seq2seq_model)

        # Compute posttraining BLEU
        posttraining_bleu = compute_bleu(targets, corrected_translations)
        post_gleu_scores = gleu_distr(sources, targets[curr_end:], corrected_translations[curr_end:])

        postrecall = unigram_recall(target_keyphrases, targets[curr_end:], corrected_translations[curr_end:])
        delta_recalls.append(postrecall - prerecall)
        print("Post Recall {}".format(postrecall))
        postprecision = unigram_precision(target_keyphrases, targets[curr_end:], corrected_translations[curr_end:])
        print("Post Precision {}".format(postprecision))
        # plt.figure(0)
        # sns.distplot(post_gleu_scores, bins=20, kde=True, kde_kws={"bw": .2}, hist_kws={"align": "mid", "rwidth": 0.75})
        # plt.savefig("gleu_dist_{}_{}_{}.png".format(metric, i * step_size, "DESC" if reverse else "ASC"))
        # plt.clf()

        diff_gleu = [post - pre for (post, pre) in zip(post_gleu_scores, pre_gleu_scores)]
        diff_gleu = sorted(diff_gleu, reverse=True)
        # plt.plot(list(range(0, len(diff_gleu))), diff_gleu)
        # plt.savefig("gleu_diff_{}_{}_{}".format(metric, i * step_size, "DESC" if reverse else "ASC"))
        # plt.clf()

        print("Post-Training BLEU: {}".format(posttraining_bleu))

        delta_bleu = posttraining_bleu - pretraining_bleu
        print("Delta BLEU {}".format(delta_bleu))
        print()

        delta_bleus.append(delta_bleu)

    # plt.figure(101)
    # bleu_scores.append(1)
    # plt.plot(list(range(0, len(bleu_scores))), bleu_scores, label="{}_{}".format(metric, "DESC" if reverse else "ASC"))

    print("Delta Recalls")
    print(delta_recalls)
    return delta_bleus


def get_gensim_topics(sources):
    from summa import keywords
    from nltk.corpus import stopwords
    from nltk.stem.snowball import GermanStemmer

    stemmer = GermanStemmer()
    sources = [" ".join([stemmer.stem(w) for w in s.split()]) for s in sources]

    for sentence in sources:
        if "dokument" in sentence:
            print(sentence)

    stop_words = stopwords.words("german") + ["&quot;", "sagte", "dass"]

    text = " ".join([s.replace("@@ ", "").lower() for s in sources])
    # text = " ".join([stemmer.stem(word) for word in text.split(" ")])


    keyphrases = keywords.keywords(text, language="german", split="True", words=10, scores=True)

    return keyphrases


def get_lda_topics(sources):
    from nltk.tokenize import RegexpTokenizer
    from nltk.corpus import stopwords
    from gensim import corpora, models
    import gensim

    stop_words = stopwords.words("german") + ["&quot;", "sagte", "dass"]
    texts = [s.replace("@@ ", "").lower().split(" ") for s in sources]
    texts = [list(filter(lambda word: word.lower() not in stop_words and len(word) > 2, s)) for s in texts]

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=50)

    print(ldamodel.print_topics(num_topics=10, num_words=5))


def get_keyphrases(sources):
    sources = [s.replace("@@ ", "") for s in sources]
    r = Rake(language="german", max_length=1)

    # Extraction given the list of strings where each string is a sentence.
    r.extract_keywords_from_sentences(sources)

    # To get keyword phrases ranked highest to lowest.
    return r.get_ranked_phrases_with_scores()


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


loader = LanguagePairLoader("de", "en", "data/medical.bpe.de", "data/medical.bpe.en")
# loader = LanguagePairLoader("de", "en", hp.source_test_file, hp.target_test_file)
_, _, pairs = loader.load()

random.shuffle(pairs)
pairs = pairs[:400]

seq2seq_model = load_model()

# Translate sources
sources, targets, translations = [p[0] for p in pairs], [p[1] for p in pairs], []
scores = {}
scorer = Scorer()

print("Key phrases")
extractor = DomainSpecificExtractor(source_file="data/medical.tok.de", train_source_file=hp.source_file,
                                    train_vocab_file="train_vocab.pkl")
keyphrases = extractor.extract_keyphrases(n_results=50)

target_extractor = DomainSpecificExtractor(source_file="data/medical.tok.en", train_source_file=hp.source_file,
                                           train_vocab_file="train_vocab_en.pkl")
target_keyphrases = target_extractor.extract_keyphrases()
print(target_keyphrases)

for i, pair in enumerate(pairs):
    if i % 10 == 0:
        print("Translated {} of {}".format(i, len(pairs)))
    translation, attn, _ = seq2seq_model.translate(pair[0])
    translations.append(" ".join(translation[:-1]))

    metrics_scores = scorer.compute_scores(pair[0], " ".join(translation[:-1]), attn, keyphrases)
    for metric in metrics_scores:
        if metric not in scores:
            scores[metric] = []
        scores[metric].append(metrics_scores[metric])

# Plot each metric
plt.style.use('seaborn-darkgrid')
palette = plt.get_cmap('Set1')

x = range(0, 50 + step_size, step_size)

sort_direction = {"coverage_penalty": True,
                  "coverage_deviation_penalty": True,
                  "confidence": False,
                  "length": True,
                  "ap_in": True,
                  "ap_out": True,
                  "keyphrase_score": False,
                  }
metrics = ["keyphrase_score"]
n_iters = 1
for i, metric in enumerate(metrics):
    avg_bleus = [0 for _ in range(1, 100 // (step_size * 2) + 1)]
    for j in range(n_iters):
        delta_bleus = evaluate_metric(sources, targets, translations, scores[metric], metric,
                                      reverse=not sort_direction[metric])
        avg_bleus = [avg + delta for (avg, delta) in zip(avg_bleus, delta_bleus)]

    delta_bleus = [0] + [b / n_iters * 100 for b in delta_bleus]

    plt.plot(x, delta_bleus, marker='', color=palette(i), linewidth=1, alpha=0.9, label=metric)

# plt.figure(101)
# plt.legend(loc='upper left', ncol=1)
# plt.savefig("bleu_correction.png")
# plt.clf()

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
