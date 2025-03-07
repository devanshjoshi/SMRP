import numpy as np

def compute_autocovariance(returns, max_lag=2):
    """
    Compute autocovariance matrices up to max_lag
    
    Parameters:
    -----------
    returns : numpy.ndarray
        Returns matrix with shape (T, n) where T is time periods and n is number of assets
    max_lag : int
        Maximum lag for which to compute autocovariance matrices
        
    Returns:
    --------
    autocovariance : list
        List of autocovariance matrices
    """
    T, n = returns.shape
    autocovariance = []
    
    # Calculate mean-centered returns
    centered_returns = returns - returns.mean(axis=0)
    
    for lag in range(max_lag+1):
        if lag == 0:
            # Variance-covariance matrix
            M_k = np.dot(centered_returns.T, centered_returns) / (T - 1)
        else:
            # Autocovariance matrix for lag k
            lagged_product = centered_returns[:-lag].T @ centered_returns[lag:]
            M_k = lagged_product / (T - lag - 1)
            
            # Make it symmetric (as required for SDP)
            M_k = (M_k + M_k.T) / 2
        
        autocovariance.append(M_k)
    
    return autocovariance

def calculate_predictability_score(M_0, M_1, weights):
    """
    Calculate predictability score for given weights
    
    Parameters:
    -----------
    M_0 : numpy.ndarray
        Covariance matrix
    M_1 : numpy.ndarray
        First-order autocovariance matrix
    weights : numpy.ndarray
        Portfolio weights
        
    Returns:
    --------
    score : float
        Predictability score (lower is better for mean reversion)
    """
    w = np.array(weights)
    w = w / np.sum(np.abs(w))
    M_0_reg = M_0 + 1e-4 * np.eye(M_0.shape[0])
    try:
        M_0_inv = np.linalg.inv(M_0_reg)
        M = M_1 @ M_0_inv @ M_1.T
    except:
        M_0_inv = np.linalg.pinv(M_0_reg)
        M = M_1 @ M_0_inv @ M_1.T
    
    return (w @ M @ w) / (w @ M_0 @ w)

def calculate_portmanteau_score(M_0, autocovariance_matrices, weights, max_lag=10):
    """
    Calculate portmanteau score for given weights
    
    Parameters:
    -----------
    M_0 : numpy.ndarray
        Covariance matrix
    autocovariance_matrices : list
        List of autocovariance matrices
    weights : numpy.ndarray
        Portfolio weights
    max_lag : int
        Maximum lag to consider
        
    Returns:
    --------
    score : float
        Portmanteau score (lower is better for mean reversion)
    """
    w = np.array(weights)
    
    # Ensure denominator is not zero
    denominator = w @ M_0 @ w
    if abs(denominator) < 1e-10:
        return 10.0  # Return high score for near-zero denominator
    
    score = 0
    p = min(max_lag, len(autocovariance_matrices))
    
    # Sum squared autocorrelations
    for i in range(p):
        M_i = autocovariance_matrices[i]
        numerator = w @ M_i @ w
        term = numerator / denominator
        score += term**2
    return score

def calculate_crossing_score(M_0, M_1, weights):
    """
    Calculate crossing statistic score for given weights
    
    Parameters:
    -----------
    M_0 : numpy.ndarray
        Covariance matrix
    M_1 : numpy.ndarray
        First-order autocovariance matrix
    weights : numpy.ndarray
        Portfolio weights
        
    Returns:
    --------
    score : float
        Crossing score (lower is better for mean reversion)
    """
    w = np.array(weights)
    
    # Ensure denominator is not zero
    denominator = w @ M_0 @ w
    if abs(denominator) < 1e-10:
        return 1.0  # Return high score for near-zero denominator
    
    numerator = w @ M_1 @ w
    score = numerator / denominator
    
    return score

def truncate_portfolio(weights, tickers, norm_factor, max_assets=5):
    """
    Truncate a portfolio to the top N assets based on price-normalized contributions
    
    Parameters:
    -----------
    weights : numpy.ndarray
        Full portfolio weights
    tickers : list
        List of ticker symbols corresponding to weights
    norm_factor : float
        Normalization factor for prices
    max_assets : int
        Maximum number of assets to include in truncated portfolio
        
    Returns:
    --------
    truncated_weights : numpy.ndarray
        Truncated portfolio weights
    truncated_tickers : list
        Tickers corresponding to truncated weights
    """
    
    n_assets = len(weights)
    trunc_size = min(max_assets, n_assets)
    
    # Calculate price-normalized contribution
    price_normalized_weights = np.abs(weights / norm_factor)
    
    # Create a DataFrame to store asset information
    import pandas as pd
    weight_df = pd.DataFrame({
        'asset': tickers,
        'weight': weights,
        'abs_weight': np.abs(weights),
        'price': norm_factor,
        'contribution': price_normalized_weights
    })
    
    # Sort by contribution (largest to smallest)
    weight_df = weight_df.sort_values('contribution', ascending=False)
    
    # Select top N assets
    top_assets = weight_df.head(trunc_size)['asset'].values
    
    # Create truncated weights and tickers
    truncated_weights = []
    truncated_tickers = []
    
    # Preserve only the weights and tickers for top assets, keeping original signs
    for ticker, weight in zip(tickers, weights):
        if ticker in top_assets:
            truncated_weights.append(weight)
            truncated_tickers.append(ticker)
    
    truncated_weights = np.array(truncated_weights)
    
    # Re-normalize truncated weights
    truncated_weights = truncated_weights / np.sum(np.abs(truncated_weights))
    
    return truncated_weights, truncated_tickers