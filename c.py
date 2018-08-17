from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import numpy as np 
import re

# for each class, add all corresponding documents together and use this to calculate the TFxICF
def retrieve_data():
    graphics_train = fetch_20newsgroups(subset = 'train', shuffle = True, random_state = 42)
    categories = graphics_train.target_names
    # cluster the data from one class
    all_data = graphics_train.data
    filenames = graphics_train.filenames
    return all_data, filenames, categories

def stem_text(sent, context=None):
    processed_tokens = []
    tokens = word_tokenize(sent)
    porter = PorterStemmer()
    for t in tokens:
        t = porter.stem(t)
        processed_tokens.append(t)
    return " ".join(processed_tokens)

# use this pattern to extract the category of a file, the filename is the pattern as"xx/xxxx/category/xxx"
def cluster_data(all_data, filenames, categories):
    pattern = re.compile(".*/(.*?)/.*?$")
    clustered_data = [""] * len(categories) 
    for i in range(len(all_data)):
        data = all_data[i]
        filename = filenames[i]
        category = pattern.match(filename).group(1)	
        index = categories.index(category)
        clustered_data[index] += " " + data
    return clustered_data

# stem the data
def stem(clustered_data):
    for i, data in enumerate(clustered_data):
        clustered_data[i] = stem_text(data)

# process the data to get rid of stop words and ...
def get_refined_words():
    stop_words = text.ENGLISH_STOP_WORDS
    return stop_words

def caculate(refined_set, clustered_data):
    vectorizer = CountVectorizer(stop_words = refined_set, max_df = 0.9)
    train_counts = vectorizer.fit_transform(clustered_data)
    tfidf_transformer = TfidfTransformer()
    term_name = vectorizer.get_feature_names()
    train_tfidf = tfidf_transformer.fit_transform(train_counts)
    # get the most frequent words in sys.ibm.pc.hardware, sys.mac.hardware, misc.forsale, religion.christian, which is in the 3rd, 4th, 6th, 15th position in the categories
    positions = [3, 4, 6, 15]
    for pos in positions:
        lst = train_tfidf[pos]
        x = lst.toarray()
        l = np.argsort(x)[0][-10:]
        for k in l:
            print term_name[k],
        print ""
        print ""

if __name__ == "__main__":
    all_data, filenames, categories = retrieve_data()
    clustered_data = cluster_data(all_data, filenames, categories)
#    stem(clustered_data)
    stop_words = get_refined_words()
    caculate(stop_words, clustered_data)
     


