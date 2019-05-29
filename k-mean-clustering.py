from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import wikipediaapi
import nltk
import numpy as np

import string


def process_text(text):
    text = text.translate(string.punctuation)

    tokens = word_tokenize(text)

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def cluster(texts):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=process_text,
                                       stop_words=stopwords.words('english'),
                                       max_df=0.8,
                                       use_idf=True,
                                       min_df=0.2,
                                       lowercase=True)

    nb_clusters = 10

    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    dist = 1 - cosine_similarity(tfidf_matrix)
    print(dist)
    cosine_similarities = linear_kernel(tfidf_matrix[0:1], tfidf_matrix).flatten()
    print(cosine_similarities)

    km = KMeans(n_clusters=nb_clusters, max_iter=10000, n_init=1, init='k-means++')
    clusters = km.fit_predict(tfidf_matrix)

    for c in range(nb_clusters):
        current = np.where(clusters == c)[0]
        print(current)
        print("========== CLUSTER ========= %d" % c)
        for i in current:
            if cosine_similarities[i] > 0:
                print("%f %s" % (cosine_similarities[i], texts[i]))


def get_article(topic):
    wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    p_wiki = wiki.page(topic)
    sent = nltk.sent_tokenize(p_wiki.text)
    print(len(sent))

    return sent


st = get_article("Artificial Intelligence")
cluster(st)
