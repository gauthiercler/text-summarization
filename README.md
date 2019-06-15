# Text Summarization, a Natural Language Processing project

## Introduction

This project is aiming to create summary text based on a wikipedia article.

You can find presentation here https://slides.com/googo/text-summarization/


## Motivations

Our motivations for this project were to implement and compare text summarization through different algorithms and state what could be the best ways to achieve text summarization. 

We documented ourself deeply about these subjets and what approachs we could go through.


## Pipeline

Our program is defined and is excuted in this order:

**1. Web crawler/scraper**

**2. Pre-processsing, tokenization**

**3. Algorithms computation**

**4. Results formationg**

**5. Results evaluation**

**6. Display output**


## Approachs

This section discuss about what methods we decided to use for our project and why.

#### Cosine similarity

As seen in class, cosine similarity through vector space model can be a way to find similarity in dataset. We thought that it could be applied to our project. By this, it means that if we can get the most similar sentences to the wole text, we can simply select the n most relevant sentences in the result set.


#### TextRank

Derived from Google Page Rank algorithm [1], TextRank algorithm ranks parts of text. This ranking is defined by the number of relation between sentences. We base our evaluation on word frequency such as TF-IDF. 


#### K-Mean clustering

K-Mean clustering algorithm [2] is an unsupervised classification algorithm frequently used in the world of Machine Learning and Data Science. Its main purpose is to, given a n dimension data set, be able to classify this data in categories (called clusters) according to them features. So using this algorithm, we can classify any type of data (images, text...).

Even K-Mean clustering has a data classification purpose, we decided to try to adapt its feature to text summarization. In our case, if we give as input our sentences from original text to K-Mean, it will classify our sentences by topics. Then we can pick the n most relevant sentences from each cluster/topic to form our summary.

This idea came to us after professor sugested this trail by providing some papers and exchanging with him during appointments.

We based our implementation on these two papers [3] and [4].

<img src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6258645/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6d656469612d702e736c69642e65732f75706c6f6164732f3437353230312f696d616765732f363235363735372f7061737465642d66726f6d2d636c6970626f6172642e706e67.png" style="" data-natural-width="440" data-natural-height="668">

Chart showing data repartition of topic "Artificial Intelligence" over 10 clusters. Each category shows the top words in the cluster (note that words are stemmed).

## Evaluation

In order to evalute generated summaries with reference summary, we need a relavant evaluation tool. How can we state in term of number how close is our summary to the reference one.

We decided to use ROUGE evaluation system [5]. ROUGE evaluation is a method to calculate the percentage of generate summary in reference summary and vice versa

Evaluation leans on overlap of N-grams [6] between the system and reference summaries and Longest Common Subsequence (LCS)

For a detailed overview about ROUGE package, please take a look to [7]

## Results

<img style="" data-natural-width="604" data-natural-height="339" data-lazy-loaded="" src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6256676/pasted-from-clipboard.png">

<img data-natural-width="605" data-natural-height="340" style="" data-lazy-loaded="" src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6258432/pasted-from-clipboard.png">

<img style="" data-natural-width="726" data-natural-height="440" data-lazy-loaded="" src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6256666/pasted-from-clipboard.png">

<img style="" data-natural-width="605" data-natural-height="340" data-lazy-loaded="" src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6258433/pasted-from-clipboard.png">

As we can see algorithm has been computed on different topics.


## Conclusion

Even K-mean can be in some way an approach for summarize text, our implementation didnt provide expected results. Although this basic implementation didnt do the trick, be believe that combined with other evaluation algorithms, it can be used in text summarization domain.


## Next Steps

- **Acronym replace:** In our current implementation, acronyms are evaluated as differents words than their real meaning. For example, if we process summarization on "Artificial Intelligence" topic, the presence of many AI acronym occurences in the base text will impact sentences weighting, relation and similarities.

- **Abstractive approach:** Even extractive approach could lead to pretty good results (see Gensim summarization), current implementation only select most relevant sentences from base text, unlike Wikipedia summaries that are sometimes (even quite often) generated using abstractive summarization. Abstractive summarization generate new content (sentences, words...) instead of just selecting the part of base text.

- **Topic based:** Focusing on defined topics or domains would be a way to improve drasticly algorithm performance. If we stick to only defined topics, we can define other evaluations methods based on these topics.

- **Different data source:** To evaluate algorithm performance, it would be relevant to use an other data source (such as News papers articles), or a data source which generate summaries only using extractive summarization.

## References

[1] Google Page Rank algorithm. https://en.wikipedia.org/wiki/PageRank

[2] K-means clustering, Wikipedia. https://en.wikipedia.org/wiki/K-means_clustering

[3] Automatic document summarization by sentence extraction. https://pdfs.semanticscholar.org/e7a4/8350000cec2025a212e7e3ca533b76351027.pdf

[4] Automatic extractive single document summarization, An unsupervised approach https://pdfs.semanticscholar.org/e7a4/8350000cec2025a212e7e3ca533b76351027.pdf

[5] ROUGE (Metric), Wikipedia. https://en.wikipedia.org/wiki/ROUGE_(metric)

[6] n-gram https://en.wikipedia.org/wiki/N-gram

[7] Lin, Chin-Yew. "Rouge: A package for automatic evaluation of summaries." http://anthology.aclweb.org/W/W04/W04-1013.pdf


