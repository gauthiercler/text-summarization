# NLP

## Introduction

This project is aiming to create summary text based on a wikipedia article.

You can find presentation here https://slides.com/googo/text-summarization/


## Motivations

Our motivations for this project were to implement and compare text summarization through different algorithms and state what could be the best ways to achieve text summarization. 

We documented ourself deeply about these subjets and what approachs we could go by.


## Approachs

This section discuss about what methods we decided to use for our project and why.

#### Cosine similarity

As seen in class, cosine similarity through vector space model can be a way to find similarity in dataset. We thought that it could be applied to our project. By this, it means that if we can get the most similar sentences to the wole text, we can simply select the n most relevant sentences in the result set.


#### TextRank

Derived from Google Page Rank algorithm [], TextRank algorithm ranks parts of text. This ranking is defined by the number of relation between sentences. We base our evaluation on word frequency such as TF-IDF. 


#### K-Mean clustering




## Evaluation

In order to evalute generated summaries with reference summary, we need a relavant evaluation tool. How can we state in term of number how close is our summary to the reference one.

We decided to use ROUGE evalution system.



## Results

<img style="" data-natural-width="604" data-natural-height="339" data-lazy-loaded="" src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6256676/pasted-from-clipboard.png">

<img data-natural-width="605" data-natural-height="340" style="" data-lazy-loaded="" src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6258432/pasted-from-clipboard.png">

<img style="" data-natural-width="726" data-natural-height="440" data-lazy-loaded="" src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6256666/pasted-from-clipboard.png">

<img style="" data-natural-width="605" data-natural-height="340" data-lazy-loaded="" src="https://s3.amazonaws.com/media-p.slid.es/uploads/475201/images/6258433/pasted-from-clipboard.png">

## Conclusion




## Next Steps

- Acronym replace
- Abstractive approach
- Topic based
- Different data source

## References

[1] Google Page Rank algorithm. https://en.wikipedia.org/wiki/PageRank

[2] ROUGE (Metric), Wikipedia. https://en.wikipedia.org/wiki/ROUGE_(metric)

[3] Lin, Chin-Yew. "Rouge: A package for automatic evaluation of summaries." http://anthology.aclweb.org/W/W04/W04-1013.pdf


