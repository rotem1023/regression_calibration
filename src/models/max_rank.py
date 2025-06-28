
import numpy as np

def adjusted_q_max_rank(S, alpha):
    '''
    S: np.ndarray, shape (n, d), where n is the number of samples and d is the number of dimensions, each element is a conformality score for the dimension.
    alpha: float, the target coverage level (1 - alpha) for the conformal prediction
    '''
    n = S.shape[0]  # number of samples
    R = np.argsort(np.argsort(S, axis=0), axis=0) # column-wise rank matrix
    r_max = np.max(R, axis=1) # row-wise max-rank vec(r)_max
    q = np.ceil((1- alpha) * (n + 1)) / n # conformal target coverage level
    rank_q = np.quantile(r_max, q, axis=0, method="higher") # quantile r_max
    adj_quantile = np.sort(S, axis=0)[rank_q]
    return adj_quantile.tolist()