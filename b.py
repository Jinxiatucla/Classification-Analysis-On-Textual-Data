from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem import PorterStemmer
import numpy as np 
from nltk import word_tokenize

def get_data():
    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    graphics_train = [] 
    for category in categories:
        graphics_train.append(fetch_20newsgroups(subset = 'train', categories = [category], shuffle = True, random_state = 42))
    return graphics_train

def stem_text(sent, context = None):
    processed_tokens = []
    tokens = word_tokenize(sent)
    porter = PorterStemmer()
    for t in tokens:
        t = porter.stem(t)
        processed_tokens.append(t)
    return " ".join(processed_tokens)

def transform(graphics_train, stem = True):
    stop_words = text.ENGLISH_STOP_WORDS
    min_df = 5 
    vectorizer = CountVectorizer(min_df = min_df, stop_words = stop_words)
    train_counts = []
    for graphic in graphics_train:
        lst = []
        if stem:
            for data in graphic.data:
                lst.append(stem_text(data))
            train_counts.append(vectorizer.fit_transform(lst))
        else:
            train_counts.append(vectorizer.fit_transform(graphic.data))
    tfidf_transformer = TfidfTransformer()
    train_tfidfs = []
    for train_count in train_counts:
        train_tfidfs.append(tfidf_transformer.fit_transform(train_count))
    return train_tfidfs

if __name__ == "__main__":
    graphics_train = get_data()
    train_tfidfs = transform(graphics_train)
    for train_tfidf in train_tfidfs:
        print train_tfidf.shape	
#print train_tfidf.shape
