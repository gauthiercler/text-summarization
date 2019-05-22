import nltk
import re


class PreProcessing:

    def __init__(self):
        self.text = ""
        self.sentences = []
        self.formatted = ""
        self.stopwords = []
        nltk.download('stopwords')

    def run(self, paragraphs):
        for p in paragraphs:
            self.text += p.text

        self.text = re.sub(r'\[[0-9]*\]', ' ', self.text)
        self.text = re.sub(r'\s+', ' ', self.text)

        self.formatted = re.sub('[^a-zA-Z]', ' ', self.text)
        self.formatted = re.sub(r'\s+', ' ', self.formatted)

        self.sentences = nltk.sent_tokenize(self.text)
        self.stopwords = nltk.corpus.stopwords.words('english')
