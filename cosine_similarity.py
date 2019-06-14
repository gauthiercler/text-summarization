from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from formatting import gen_serie
from tokenizer import textblob_tokenizer


def cosine(texts, ref):
    vec = TfidfVectorizer(tokenizer=textblob_tokenizer,
                          stop_words='english',
                          use_idf=True)
    matrix = vec.fit_transform(texts)

    cosine_similarities = cosine_similarity(matrix[0:1], matrix).flatten()

    nb_sentences_in_base_summary = len(ref.split('.'))

    cosine_similarities = list(cosine_similarities)
    cos_results = []
    for i in range(0, nb_sentences_in_base_summary):
        n = cosine_similarities.index(max(cosine_similarities))
        cos_results.append(texts[n])
        del cosine_similarities[n]

    res = ' '.join(cos_results)

    r = Rouge()
    rouge = r.get_scores(res, ref)

    return gen_serie('Cosine Similarity', rouge, res)

