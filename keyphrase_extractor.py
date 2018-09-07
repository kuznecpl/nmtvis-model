from summa import keywords
from nltk.corpus import stopwords
from nltk.stem.snowball import GermanStemmer
import pickle
from data_loader import LanguagePairLoader
from collections import Counter
import os.path


class Extractor:
    def __init__(self):
        pass

    def extract_keyphrases(self, sentences, n_results=5):
        stop_words = stopwords.words("german")

        sentences = [sentence.replace("@@ ", "") for sentence in sentences]
        sentences = [" ".join(word for word in sentence.split(" ") if word.lower() not in stop_words) for sentence in
                     sentences]
        text = " ".join(sentences)
        keyphrases = keywords.keywords(text.lower(), language="german", split=True, words=n_results, scores=True)
        print("Keyphrases: {}".format(keyphrases))
        res = []
        for keyphrase, score in keyphrases:
            count = 0
            for sentence in sentences:
                if keyphrase in sentence.lower():
                    count += 1
            res.append({"name": keyphrase, "occurrences": count, "active": False, "score": score})

        return res


class DomainSpecificExtractor:
    def __init__(self, source_file, train_source_file, train_vocab_file, frequency_threshold=10):
        self.source_file = source_file
        self.frequency_threshold = frequency_threshold

        if train_source_file:
            self.train_source_file = train_source_file
        if train_vocab_file:
            self.train_vocab_file = train_vocab_file

    def extract_keyphrases(self, n_results=20):
        train_vocab = None
        if os.path.isfile(self.train_vocab_file):
            train_vocab = pickle.load(open(self.train_vocab_file, "rb"))
        else:
            train_vocab = Counter()
            train_loader = LanguagePairLoader("de", "en", self.train_source_file, self.train_source_file)
            train_in, train_out, train_pairs = train_loader.load()
            for source, _ in train_pairs:
                for word in source.replace("@@ ", "").split(" "):
                    train_vocab[word] += 1
            pickle.dump(train_vocab, open(self.train_vocab_file, "wb"))

        loader = LanguagePairLoader("de", "en", self.source_file, self.source_file)
        in_lang, _, pairs = loader.load()

        domain_words = []
        for word in in_lang.word2count:
            if train_vocab[word] < self.frequency_threshold and in_lang.word2count[word] > 0:
                domain_words.append((word, in_lang.word2count[word]))

        domain_words = sorted(domain_words, key=lambda w: in_lang.word2count[w[0]], reverse=True)
        return domain_words[:n_results]
