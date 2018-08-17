import numpy as np
import e
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

twenty_train = e.get_data()
comp_tech, recre_act = e.combine_two_class(twenty_train)
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

def get_suitable_k(train_data, train_target):
    best_k = -5
    mean = 0 
    for k in range(-3, 4):
        clf = SVC(kernel = 'linear', C = 10 ** k)
        scores = cross_val_score(clf, train_data, train_target, cv = 5)
        tmp_mean = scores.mean()
        if (tmp_mean > mean):
            mean = tmp_mean
            best_k = k
    return best_k


def caculate():
    min_df = 2 
    stop_words = text.ENGLISH_STOP_WORDS
    twenty_test = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = 42)
    text_clf = Pipeline([('vect', CountVectorizer(min_df = min_df, stop_words = stop_words)),
                         ('tfidf', TfidfTransformer()),
                         ('trunc', TruncatedSVD(n_components = 50, random_state = 42))])
    train_data = text_clf.fit_transform(comp_tech + recre_act)
    train_target = [0] * len(comp_tech) + [1] * len(recre_act)
    k = get_suitable_k(train_data, train_target) 
    svc = SVC(C = 10 ** k, random_state = 42)

    print k
    svc.fit(train_data, train_target)
    predicted = svc.predict(text_clf.transform(twenty_test.data))
    test_target = e.get_category(twenty_test)
    mean = np.mean(predicted == test_target)
    return mean

if __name__ == "__main__":
    mean = caculate()
    print mean 
