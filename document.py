import spacy

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


class Document:
    def __init__(self, id, name, filename):
        self.id = id
        self.name = name
        self.filename = filename
        self.content = []
        self.translation_data = []

    def load_content(self):
        with open(UPLOAD_FOLDER + self.filename, "r") as f:
            content = f.read().replace("\n", "")
            doc = nlp(content)

            self.content = [str(sent) for sent in doc.sents]

        return self.content

    def add_translation_data(self, translation, attn, translations):
        self.translation_data.append((translation, attn, translations))
