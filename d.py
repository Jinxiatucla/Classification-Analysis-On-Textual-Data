import b 
from sklearn.decomposition import TruncatedSVD 

def get_tdidf():
    # get the tfidf from the eight classes  
    graphics_train = b.get_data()
    train_tfidfs = b.transform(graphics_train, False)
    return train_tfidfs

def truncated_SVD(train_tfidfs):
    svd = TruncatedSVD(n_components = 50, random_state = 42)
    k_matrix = []
    for tfidf in train_tfidfs:
        k_matrix.append(svd.fit_transform(tfidf))
    return k_matrix

if __name__ == "__main__":
    train_tfidfs = get_tdidf()
    k_matrix = truncated_SVD(train_tfidfs)
