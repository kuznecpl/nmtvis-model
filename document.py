UPLOAD_FOLDER = "/home/science/uploads/"


class Document:
    def __init__(self, id, name, filename):
        self.id = id
        self.name = name
        self.filename = filename
        self.content = []

    def load_content(self):
        with open(UPLOAD_FOLDER + self.filename, "r") as f:
            self.content = f.read().split("\n")

        return self.content
