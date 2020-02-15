import warnings

import numpy as np
import pandas as pd
import scipy.sparse
from rouge import Rouge
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances_argmin_min

from formatting import gen_serie
from tokenizer import textblob_tokenizer, tokenize

warnings.filterwarnings("ignore")


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


def cluster(texts, ref, clusters_nb):
    vec = TfidfVectorizer(tokenizer=textblob_tokenizer,
                          stop_words='english',
                          use_idf=True)
    matrix = vec.fit_transform(texts)

    km = KMeans(n_clusters=clusters_nb, max_iter=10000, init='k-means++').fit(matrix)

    sentences = []

    nb_sentences_in_base_summary = len(tokenize(ref))
    cnt = 0
    for i in range(0, len(texts)):
        closest, dist = pairwise_distances_argmin_min(km.cluster_centers_, matrix)
        for idx in closest:
            sentences.append(texts[idx])
            cnt += 1
            if cnt == nb_sentences_in_base_summary:
                break
        else:
            for idx in closest:
                length, _ = matrix.shape
                if idx < length:
                    matrix = delete_row_lil(matrix.tolil(), idx)
            continue
        break

    final = [x for x in texts if x in sentences]
    return ' '.join(final)


def kmean(text, ref):
    df = pd.DataFrame()

    for i in range(2, 11):
        res = cluster(text, ref, i)
        r = Rouge()
        rouge = r.get_scores(' '.join(res), ref)
        df = df.append(gen_serie('K-mean-' + str(i), rouge, res), ignore_index=True)

    return df
