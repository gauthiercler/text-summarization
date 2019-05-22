from flaskr.scraper import Scraper
from flaskr.preprocessing import PreProcessing
from flaskr.processing import Processing
from flaskr.build_output import BuildOutput


class Main:
    def __init__(self):
        self.s = Scraper()
        self.pp = PreProcessing()
        self.p = Processing()
        self.b = BuildOutput()
        self.result = ""
        self.sentences = []

    def run(self, topic):
        self.s.run('https://en.wikipedia.org/wiki/' + topic)
        self.pp.run(self.s.data)
        self.p.run(self.pp.sentences, self.pp.stopwords, self.pp.formatted)
        self.sentences = self.b.get_n_relevant_sentences(10, self.p.sentence_scores)
        # self.result = self.b.summary
