import spacy
import re

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
    def __init__(self, id, source, translation, attention, beam, score):
        self.id = id
        self.source = source
        self.translation = translation
        self.attention = attention
        self.beam = beam
        self.score = score
        self.corrected = False
        self.experiment_metrics = None


class Document:
    def __init__(self, id, name, unk_map):
        self.id = id
        self.name = name
        self.sentences = []
        self.unk_map = unk_map

    def pad_punctuation(self, s):
        s = re.sub('([.,!?()])', r' \1 ', s)
        s = re.sub('\s{2,}', ' ', s)
        return s

    def load_content(self, filename):
        with open(UPLOAD_FOLDER + filename, "r") as f:
            content = f.read()
            doc = nlp(content)

            content = []

            for sent in doc.sents:
                tokens = nlp(str(sent).lower())
                tokens = [str(token).strip().lower() for token in tokens if not str(token).isspace()]
                content.append(" ".join(tokens))

        return content
