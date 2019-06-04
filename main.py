from scraper import Scraper
from preprocessing import PreProcessing
from processing import Processing
from build_output import BuildOutput

from gensim.summarization import summarize
from rouge import Rouge


class Main:
    def __init__(self):
        self.s = Scraper()
        self.pp = PreProcessing()
        self.p = Processing()
        self.b = BuildOutput()
        self.result = ""
        self.sentences = []

    def run(self, topic):
        self.s.run(topic)
        ret = summarize(''.join(self.s.data), ratio=0.01)
        print(ret)
        print("=====================================================")
        print("=====================================================")
        print("=====================================================")
        print("=====================================================")
        print("=====================================================")
        print(self.s.base_summary)

        r = Rouge()
        rouge = r.get_scores(ret, self.s.base_summary)
        print(rouge)
        self.pp.run(self.s.data)
        self.p.run(self.pp.sentences, self.pp.stopwords, self.pp.formatted)
        self.sentences = self.b.get_n_relevant_sentences(10, self.p.sentence_scores)
        self.b.build_summary()
        self.result = self.b.summary

        rouge = r.get_scores(self.result, self.s.base_summary)
        print(rouge)


m = Main()
m.run("Algorithm")
