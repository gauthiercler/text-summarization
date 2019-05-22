import nltk


class Processing:

    def __init__(self):
        self.word_frequencies = {}
        self.sentence_scores = {}

    def run(self, sentences, stopwords, formatted):
        for word in nltk.word_tokenize(formatted):
            if word not in stopwords:
                if word not in self.word_frequencies.keys():
                    self.word_frequencies[word] = 1
                else:
                    self.word_frequencies[word] += 1

        maximum_frequency = max(self.word_frequencies.values())

        for word in self.word_frequencies.keys():
            self.word_frequencies[word] = (self.word_frequencies[word] / maximum_frequency)

        for sent in sentences:
            for word in nltk.word_tokenize(sent.lower()):
                if word in self.word_frequencies.keys():
                    if len(sent.split(' ')) < 30:
                        if sent not in self.sentence_scores.keys():
                            self.sentence_scores[sent] = self.word_frequencies[word]
                        else:
                            self.sentence_scores[sent] += self.word_frequencies[word]
