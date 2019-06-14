import heapq
import re

import nltk


def compute(sentences, stopwords, formatted):
    word_frequencies = {}
    sentence_scores = {}

    for word in nltk.word_tokenize(formatted):
        if word not in stopwords:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1

    maximum_frequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

    for sent in sentences:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]
    return sentence_scores


def text_rank(raw, text, ref):
    nb = len(ref.split('.'))

    formatted = re.sub('[^a-zA-Z]', ' ', raw)
    formatted = re.sub(r'\s+', ' ', formatted)

    scores = compute(text, nltk.corpus.stopwords.words('english'), formatted)
    sentences = heapq.nlargest(nb, scores, key=scores.get)

    return ' '.join(sentences)
