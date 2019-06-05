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
from sklearn.metrics import pairwise_distances_argmin_min
import scipy.sparse
from rouge import Rouge
import gensim.summarization



import pandas as pd

pd.set_option("display.max_columns", 10)
pd.set_option("display.max_rows", 10)


def process_text(text):

    tokens = word_tokenize(text)

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens

def textblob_tokenizer(str_input):
    blob = TextBlob(str_input.lower())
    tokens = blob.words
    words = [token.stem() for token in tokens]
    return words


def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (x, y) in data[cluster_labels == i_centroid]]
    return distances


def delete_row_lil(mat, i):
    if not isinstance(mat, scipy.sparse.lil_matrix):
        raise ValueError("works only for LIL format -- use .tolil() first")
    mat.rows = np.delete(mat.rows, i)
    mat.data = np.delete(mat.data, i)
    mat._shape = (mat._shape[0] - 1, mat._shape[1])
    return mat

def cluster(texts):
    vec = TfidfVectorizer(tokenizer=textblob_tokenizer,
                          stop_words='english',
                          use_idf=True)
    matrix = vec.fit_transform(texts)


    df = pd.DataFrame(matrix.toarray(), columns=vec.get_feature_names())
    # cosine_similarities = cosine_similarity(matrix[0:1], matrix).flatten()

    number_of_clusters = 4
    km = KMeans(n_clusters=number_of_clusters, max_iter=10000, init='k-means++').fit(matrix)

    sentences = []

    for i in range(0, 10):
        closest, dist = pairwise_distances_argmin_min(km.cluster_centers_, matrix)
        for idx in closest:
            sentences.append(texts[idx])
            matrix = delete_row_lil(matrix.tolil(), idx)

    final = [x for x in texts if x in sentences]
    # print(final)

    # print(closest)
    # print(dist)


    # print(km.fit)

    # print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = vec.get_feature_names()
    labels = []
    for i in range(number_of_clusters):
        top_ten_words = [terms[ind] for ind in order_centroids[i, :5]]
        labels.append(' '.join(top_ten_words))
        # print("Cluster {}: {}".format(i, ' '.join(top_ten_words)))

    return ' '.join(final)


    results = pd.DataFrame({
        'category': km.labels_,
        'similarity': cosine_similarities,
        'text': texts
    })

    results.sort_values(by=['similarity'], inplace=True, ascending=False)
    print(results)

    clusters_indices = [0] * number_of_clusters
    for index, c in enumerate(km.labels_):
        clusters_indices[c] += cosine_similarities[index]

    for sim in clusters_indices:
        print(sim)

    patches, text, _ = plt.pie(clusters_indices, autopct='%1.1f%%')
    plt.legend(patches, labels, bbox_to_anchor=(0, 0), loc="upper left")
    plt.show()
    results.to_csv("out.csv")

    print(sorted(Counter(km.labels_).items()))

    print(matrix[km.labels_ == 0][0])
    # print(len(km.cluster_centers_[0]))

    # plt.scatter()
    # plt.scatter(clusters[:, 0], clusters[:, 1], s=20, c=km.labels_.tolist())
    # plt.show()

    return


def get_article(topic):
    wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    p_wiki = wiki.page(topic)
    sent = nltk.sent_tokenize(p_wiki.text)
    sent = list(map(lambda x: x.translate(string.punctuation), sent))
    # print(len(sent))

    # print(p_wiki.summary)
    return p_wiki.text, sent, p_wiki.summary

def run(topic):
    raw, st, ref = get_article(topic)
    res = cluster(st)
    r = Rouge()
    rouge = r.get_scores(res, ref)
    print("K-Mean")
    print(rouge)

# from main import Main

# m = Main()
# m.run("Artificial Intelligence")
