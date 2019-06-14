import string

import nltk
from textblob import TextBlob


def tokenize(raw):
    sent = nltk.sent_tokenize(raw)
    sent = list(map(lambda x: x.translate(string.punctuation), sent))
    sent = [word for word in sent if len(word) >= 10]
    return sent


def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words
