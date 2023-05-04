import re
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
## initialise the inbuilt Stemmer
stemmer = PorterStemmer()
# # We can also use Lemmatizer instead of Stemmer
lemmatizer = WordNetLemmatizer()

import gensim
import gensim.downloader as api

wv = api.load('glove-twitter-100')

def document_vector(doc, keyed_vectors):
    """Remove out-of-vocabulary words. Create document vectors by averaging word vectors."""
    vocab_tokens = [word for word in doc if word in keyed_vectors.index_to_key]
    return np.mean(keyed_vectors.__getitem__(vocab_tokens), axis=0)

def listToString(s):
   
    # initialize an empty string
    str1 = " "
   
    # return string 
    return (str1.join(s))

#text processing
def preprocess(x):
    # To get the results in 4 decemal points
    SAFE_DIV = 0.0001 
    x = str(x).lower()  # Lowering all text to covert all of them to there base form
    
    # Replacing commonly use words or numbers like 1,000 to 1k and 1,000,000 to 1m and currency symbol to there respective names and many other symbols to there name
    
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'")\
                           .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not")\
                           .replace("n't", " not").replace("what's", "what is").replace("it's", "it is")\
                           .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are")\
                           .replace("he's", "he is").replace("she's", "she is").replace("'s", " is")\
                           .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ")\
                           .replace("€", " euro ").replace("'ll", " will").replace("@","at")
    
    # Renaming 1000 to 1k and 1000000 to 1m (the onces which may not be seperated with commans)
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    
    
    #Remove any special character like [= , ' ; "" ']
    
    pattern = re.compile('\W')
    
    if type(x) == type(''):
        x = re.sub(pattern, ' ', x)

    # Removing HTML tags
    x = BeautifulSoup(x)
    x = x.get_text()
    
    # tokenize into words
    x = x.split()
    
    # remove stop words                
    x = [t for t in x if t not in stopwords.words("english")]          
    
    return x

def test_common_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1 = set(map(lambda word: word.lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))

def test_word_share(q1,q2):
    w1 = set(map(lambda word: str(word).lower().strip(), q1.split(" ")))
    w2 = set(map(lambda word: str(word).lower().strip(), q2.split(" ")))    
    return 1.0 * len(w1 & w2)/(len(w1) + len(w2))

def stem_len(row):
    #tokenization of the questions
    token = str(row).split()
    # stemming the questions
    stem = [stemmer.stem(word) for word in token]
    return len(stem)

def lem_len(row):  
    #tokenization of the questions
    token = str(row).split() 
    # lemmitization
    lemm = [lemmatizer.lemmatize(word) for word in token]
    return len(lemm)

    

def query_point_creator(q1, q2):
    input_query = []

    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    q1 = q1.transform(listToString)
    q2 = q2.transform(listToString)

    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(test_word_share(q1, q2))
    input_query.append(stem_len([q1,q2]))
    input_query.append(lem_len([q1,q2]))


    # glove feature for q1
    q1_token=str(q1).split()
    q1_glove = q1_token.apply(lambda x : document_vector(x, wv)).toarray()

    # glove feature for q2
    q2_token=str(q2).split()
    q2_glove = q1_token.apply(lambda x : document_vector(x, wv)).toarray()

    return np.hstack((np.array(input_query).reshape(1, 22), q1_glove, q2_glove))