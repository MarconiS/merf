"""Methods for transforming/decomposing reflectance data (e.g., using PCA)
"""
from sklearn.decomposition import PCA as _PCA

def pca(features, n_pcs=100):
    """PCA transformation function
    
    Args:
        features - the input feature data to transform
        n_pcs    - the number of components to keep after transformation
        
    Returns:
        an array of PCA-transformed features
    """
    reducer = _PCA(n_components=n_pcs, whiten=True)
    return reducer, reducer.fit_transform(features)


