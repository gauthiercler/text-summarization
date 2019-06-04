import heapq


class BuildOutput:

    def __init__(self):
        self.summary = ""
        self.sentences = []

    def get_n_relevant_sentences(self, nb, scores):
        self.sentences = heapq.nlargest(nb, scores, key=scores.get)
        return self.sentences

    def build_summary(self):
        self.summary = ' '.join(self.sentences)
