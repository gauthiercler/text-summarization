
# coding: utf-8

# In[ ]:

import nltk
import re

'''
To. Our Team
If you need something more, please tell me.
Delete this comment when submitting
'''

class PreProcessing:

    def __init__(self):
        self.text = ""
        self.sentences = []
        self.formatted = ""
        self.stopwords = []
        self.tokens = []
        self.taglist = ()
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

    def run(self, input):
        
        #Set Txtfile Directory
        # paragraphs = open('yourtext.txt', 'r')
        #
        # for p in paragraphs:
        #     self.text += p

        # print(input)
        for p in input:
            self.text += p

        # self.text = input
        #Remove Special_Chars
        self.text = re.sub(r'\[[0-9]*\]', ' ', self.text)
        self.text = re.sub(r'\s+', ' ', self.text)

        self.formatted = re.sub('[^a-zA-Z]', ' ', self.text)
        self.formatted = re.sub(r'\s+', ' ', self.formatted)

        self.sentences = nltk.sent_tokenize(self.text)

        #Tokenization
        self.tokens = [word for sent in nltk.sent_tokenize(self.formatted)
                      for word in nltk.word_tokenize(sent)]
        
        #Lower Capitalization
        self.tokens = [word.lower() for word in self.tokens]
        
        #Remove StopWrods
        self.stopwords = nltk.corpus.stopwords.words('english')
        self.tokens = [token for token in self.tokens if token not in self.stopwords]
        
        #Lemmatization
        lmztr = nltk.stem.WordNetLemmatizer()
        self.tokens = [lmztr.lemmatize(word) for word in self.tokens]
        
        #Stemming
        stemmer = nltk.stem.PorterStemmer()
        self.tokens = [stemmer.stem(word) for word in self.tokens]
        
        #P.O.S tagging
        self.taglist = nltk.tag.pos_tag(self.tokens)

# p = PreProcessing()
# p.run()

#print(p.tokens)
#print(p.taglist)

