import spacy
import re
import hp
from subword_nmt.apply_bpe import BPE

nlp = spacy.load("de")

UPLOAD_FOLDER = "/home/science/uploads/"


def _is_wordlike(tok):
    return tok.orth_ and tok.orth_[0].isalpha()


def sentence_division_suppresor(doc):
    """Spacy pipeline component that prohibits sentence segmentation between two tokens that start with a letter.
    Useful for taming overzealous sentence segmentation in German model, possibly others as well."""
    for i, tok in enumerate(doc[:-1]):
        if _is_wordlike(tok) and _is_wordlike(doc[i + 1]):
            doc[i + 1].is_sent_start = False
    return doc


nlp.add_pipe(sentence_division_suppresor, name='sent_fix', before='parser')


class Sentence:
    EXPERIMENT_TYPES = ["plain", "beam"]

    def __init__(self, id, source, translation, attention, beam, score):
        self.id = id
        self.source = source
        self.translation = translation
        self.attention = attention
        self.beam = beam
        self.score = score
        self.corrected = False
        self.diff = ""
        self.flagged = False
        self.experiment_metrics = None
        self.experiment_type = "BEAM"


class Document:
    def __init__(self, id, name, unk_map, filepath):
        self.id = id
        self.name = name
        self.sentences = []
        self.keyphrases = []
        self.unk_map = unk_map
        self.filepath = filepath

    def pad_punctuation(self, s):
        s = re.sub('([.,!?()])', r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        return s

    def replace(self, sentence):
        return sentence\
            .replace('"', "&quot;")\
            .replace("'", "&apos;")\
            .replace('„', "&quot;")\
            .replace('“', "&quot;")


    def load_content(self, filename):
        with open(UPLOAD_FOLDER + filename, "r") as f:
            content = f.read()
            doc = nlp(content)

            sentences = content.split("\n")
            bpe = BPE(open(hp.bpe_file))

            content = []

            for sent in sentences:
                tokens = nlp(str(sent))
                tokens = [self.replace(str(token).strip()) for token in tokens if not str(token).isspace()]
                sentence = " ".join(tokens)
                sentence = bpe.process_line(sentence)
                content.append(sentence)

        return content
