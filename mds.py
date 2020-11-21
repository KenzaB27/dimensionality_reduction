import numpy as np

def double_centring(D):
    """
    Compute the gram matrix S with double centering trick using the definition: 
    subtract from each entry of D the mean of the corresponding row and the mean of
    the corresponding column, and add back the mean of all entries
    """
    n, _ = D.shape
    return -(D - D.mean(axis=0).reshape((1, n)) - D.mean(axis=1).reshape((n, 1)) + D.mean())/2

def double_centring_1(D):
    """
    Compute the gram matrix S with double centering trick using the formula 
    S = -0.5 (D - 1/n D1n1n^T - 1/n 1n1n^TD + 1/n^2 1/n 1n1n^TD1n1n^T)
    """
    S = np.zeros(D.shape)
    n, _ = D.shape
    ones = np.ones((n,1))
    S = -(D - (D @ ones @ ones.T)/n - (ones @ ones.T @ D)/n + (ones @ ones.T @ D @ ones @ ones.T)/(n**2))/2

    return S

def center_data(Y):
    """
    Center the data by substracting the mean over columns
    """
    return Y - Y.mean(axis=0)

def compute_similarity(Y):
    """
    Compute the gram matrix S = Y^T.Y (1)
    Note that the columns and rows are inversed in the data set 
    That is why we compute S = Y.Y^T instead of (1) 
    """
    return Y @ Y.T


def mds(S, k=2):
    """
    Reduce the dimensionality by applying multi-dimensional scaling. S being the gram matrix
    and k the reduced dimension = 2 by default
    """
    # Apply eigen decomposition on the gram matrix
    eig_val, eig_vect = np.linalg.eig(S)
    # Sort eigenvalues in descending order
    idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[idx]
    eig_vect = eig_vect[:, idx]
    # Compute the coordinates using positive-eigenvalues components only
    w, = np.where(eig_val > 0)
    L = np.diag(np.sqrt(eig_val[w]))[:k, ]
    V = eig_vect[:, w]
    # Get the k low dimensional representattion of the data
    X = L[:k, ] @ V.T
    return X.T.real

def mds1(S, k=2):
    """
    Reduce the dimensionality by applying multi-dimensional scaling. S being the gram matrix 
    and k the reduced dimension = 2 by default 
    """
    # Apply eigen decomposition on the gram matrix
    eig_val, eig_vect = np.linalg.eig(S)
    Lambda = np.diag(eig_val)
    X = np.sqrt(Lambda)[:k,] @ eig_vect.T
    return X.T

