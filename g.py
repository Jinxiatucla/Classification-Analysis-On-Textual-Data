import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import metrics
import matplotlib.pyplot as plt
import e
from sklearn.naive_bayes import MultinomialNB

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'] 
 
def caculate(comp_tech, recre_act):
    min_df = 2 
    stop_words = text.ENGLISH_STOP_WORDS
    twenty_test = fetch_20newsgroups(subset = 'test', categories = categories, shuffle = True, random_state = 42)
    text_clf = Pipeline([('vect', CountVectorizer(min_df = min_df, stop_words = stop_words)),
                         ('tfidf', TfidfTransformer()),
                         ('nmf', NMF(n_components = 50, random_state = 42)),
                         ('mnb',MultinomialNB())])
    train_target = [0] * len(comp_tech) + [1] * len(recre_act)
    text_clf.fit(comp_tech + recre_act, train_target)
    predicted = text_clf.predict(twenty_test.data)
    value =  text_clf.predict_proba(twenty_test.data)
    test_target = e.get_category(twenty_test)
    mean = np.mean(predicted == test_target)
    return test_target, predicted, value, mean

def plot(test_target, value):
    fpr, tpr, _ = roc_curve(test_target, value[:,1])
    fig, ax = plt.subplots()
    roc_auc = auc(fpr,tpr)
    ax.plot(fpr, tpr, lw=2, label= 'area under curve = %0.4f' % roc_auc)
    ax.grid(color='0.7', linestyle='--', linewidth=1)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate',fontsize=15)
    ax.set_ylabel('True Positive Rate',fontsize=15)
    ax.legend(loc="lower right")
    for label in ax.get_xticklabels()+ax.get_yticklabels():
        label.set_fontsize(15)
    plt.show()

if __name__ == "__main__":
    twenty_train = e.get_data()
    comp_tech, recre_act = e.combine_two_class(twenty_train)
    test_target, predicted, value, mean = caculate(comp_tech, recre_act)
    e.plot(test_target, value)
    confusion_matrix = metrics.confusion_matrix(test_target, predicted)
    print(metrics.classification_report(test_target, predicted, target_names = ["computer", "recreation"])) 
    print confusion_matrix
    print mean









