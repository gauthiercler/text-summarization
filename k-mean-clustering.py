from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
import wikipediaapi
import nltk
import numpy as np
import matplotlib.pyplot as plt
import string
from textblob import TextBlob
from collections import Counter, defaultdict

import pandas as pd

pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)


def process_text(text):
    text = text.translate(string.punctuation)

    tokens = word_tokenize(text)

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens


# def textblob_tokenizer(str_input):
#     blob = TextBlob(str_input.lower())
#     tokens = blob.words
#     words = [token.stem() for token in tokens]
#     return words


def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (x, y) in data[cluster_labels == i_centroid]]
    return distances


def cluster(texts):
    vec = TfidfVectorizer(tokenizer=process_text,
                          stop_words='english',
                          use_idf=True)
    matrix = vec.fit_transform(texts)
    df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())
    cosine_similarities = cosine_similarity(matrix[0:1], matrix).flatten()

    number_of_clusters = 10
    km = KMeans(n_clusters=number_of_clusters)
    km.fit(matrix)
    print(km.fit)

    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vec.get_feature_names()
    for i in range(number_of_clusters):
        top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
        print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))

    results = pd.DataFrame({
        'category': km.labels_,
        'similarity': cosine_similarities,
        'text': texts
    })

    results.sort_values(by=['similarity'], inplace=True, ascending=False)
    print(results)

    clusters_indices = [0] * number_of_clusters
    for index, c in enumerate(km.labels_):
        print(index)
        clusters_indices[c] += cosine_similarities[index]

    for sim in clusters_indices:
        print(sim)

    plt.pie(clusters_indices, autopct='%1.1f%%', shadow=True)
    plt.show()
    results.to_csv("out.csv")

    print(sorted(Counter(km.labels_).items()))

    return


def get_article(topic):
    wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    p_wiki = wiki.page(topic)
    sent = nltk.sent_tokenize(p_wiki.text)
    # print(len(sent))

    # print(p_wiki.summary)
    return sent, p_wiki.summary


st, ref = get_article("Artificial Intelligence")
cluster(st)
