from scraper import Scraper
from preprocessing import PreProcessing
from processing import Processing
from build_output import BuildOutput

from gensim.summarization import summarize
from rouge import Rouge

import kmean


class Main:
    def __init__(self):
        self.s = Scraper()
        self.pp = PreProcessing()
        self.p = Processing()
        self.b = BuildOutput()
        self.result = ""
        self.sentences = []

    def run(self, topic):
        kmean.run(topic)

        self.s.run(topic)
        raw = ''.join(self.s.data)
        ratio = len(self.s.base_summary) / len(raw)
        # print(ratio)
        ret = summarize(raw, len(self.s.base_summary) / len(raw))

        r = Rouge()
        rouge = r.get_scores(ret, self.s.base_summary)
        print("Gensim")
        print(rouge)

        self.pp.run(self.s.data)
        self.p.run(self.pp.sentences, self.pp.stopwords, self.pp.formatted)
        nb_sentences_in_base_summary = len(self.s.base_summary.split('.'))
        self.sentences = self.b.get_n_relevant_sentences(nb_sentences_in_base_summary, self.p.sentence_scores)
        self.b.build_summary()
        self.result = self.b.summary

        print("TextRank")
        rouge = r.get_scores(self.result, self.s.base_summary)
        print(rouge)


m = Main()
m.run("Algorithm")
