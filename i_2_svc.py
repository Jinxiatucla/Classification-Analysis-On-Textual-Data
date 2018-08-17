import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.decomposition import NMF

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']


def get_data():
    twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    print(twenty_train.target_names)
    return twenty_train


def combine_four_class(twenty_train):
    comp_pc = []
    comp_mac = []
    misc = []
    soc = []
    target = twenty_train.target
    data = twenty_train.data
    # combine the data into two categories
    for tup in zip(target, data):
        if tup[0] == 0:
            comp_pc.append(tup[1])
        elif tup[0] == 1:
            comp_mac.append(tup[1])
        elif tup[0] == 2:
            misc.append(tup[1])
        elif tup[0] == 3:
            soc.append(tup[1])
    return comp_pc, comp_mac, misc, soc


# the target of the first class is 0, and the target of second is 1
"""
def get_category(test):
    return map(lambda x: int(x >= 4), test.target)
"""

def caculate(comp_pc, comp_mac, misc, soc ):
    min_df = 5
    C = 1000
    stop_words = text.ENGLISH_STOP_WORDS
    twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    text_clf = Pipeline([('vect', CountVectorizer(min_df=min_df, stop_words=stop_words)),
                         ('tfidf', TfidfTransformer()),
                         ('trunc', NMF(n_components=50, init='random', random_state=0)),
                         ('clf', OneVsOneClassifier(SVC(probability=True, C=C, random_state=42)))])
    train_target = [0] * len(comp_pc) + [1] * len(comp_mac) + [2]*len(misc) + [3]*len(soc)
    text_clf.fit(comp_pc + comp_mac + misc + soc, train_target)
    predicted = text_clf.predict(twenty_test.data)
    #value = text_clf.predict_proba(twenty_test.data)
    test_target = twenty_test.target
    mean = np.mean(predicted == twenty_test.target)
    return test_target, predicted, mean,


def plot(test_target, predicted):
    fpr, tpr, _ = roc_curve(test_target, predicted)
    fig, ax = plt.subplots()
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, label='area under curve = %0.4f' % roc_auc)
    ax.grid(color='0.7', linestyle='--', linewidth=1)
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=15)
    ax.set_ylabel('True Positive Rate', fontsize=15)
    ax.legend(loc="lower right")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(15)
    plt.show()


if __name__ == "__main__":
    twenty_train = get_data()
    comp_pc, comp_mac, misc, soc = combine_four_class(twenty_train)
    test_target, predicted,  mean = caculate(comp_pc, comp_mac, misc, soc)
    #plot(test_target, predicted)
    confusion_matrix = metrics.confusion_matrix(test_target, predicted)
    print(metrics.classification_report(test_target, predicted, target_names=['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']))
    print confusion_matrix
    print mean








