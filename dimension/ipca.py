from sklearn.datasets import fetch_mldata
import numpy as np
mnist=fetch_mldata("MNIST_original")

X=mnist["data"]
#使用np.array_split（）方法的IPCA
from sklearn.decomposition import IncrementalPCA
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X_batch)
X_mnist_reduced = inc_pca.transform(X)