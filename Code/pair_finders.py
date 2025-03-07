import numpy as np
import pandas as pd
from scipy import linalg

# Import functions from other modules
from sdp_optimizers import optimize_predictability, optimize_portmanteau, optimize_crossing
from portfolio_metrics import calculate_predictability_score, calculate_portmanteau_score, calculate_crossing_score

def find_best_pairs_sdp_exhaustive(returns, tickers, volatility_threshold=0.0001, 
                                  max_lag=10, crossing_mu=0.1, num_top_pairs=5):
    """
    Find the best pairs for each criterion by evaluating all possible pairs using SDP with regular PCA
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        Returns data with stocks as columns
    tickers : list
        List of ticker symbols
    volatility_threshold : float
        Minimum variance threshold
    max_lag : int
        Maximum lag for autocorrelation analysis
    crossing_mu : float
        Weight for higher-order terms in crossing criterion
    num_top_pairs : int
        Number of top pairs to return for each criterion
        
    Returns:
    --------
    pred_pair, port_pair, cross_pair : tuples
        Best pairs for each criterion in format (idx1, idx2, score, weights)
    """
    from portfolio_metrics import compute_autocovariance
    
    n = len(tickers)
    total_pairs = n * (n - 1) // 2
    print(f"Evaluating all {total_pairs} pairs using SDP with regular PCA...")
    
    # Initialize lists to track best pairs for each criterion
    pred_pairs = []
    port_pairs = []
    cross_pairs = []
    
    pairs_evaluated = 0
    
    # Evaluate all possible pairs
    for i in range(n):
        for j in range(i+1, n):
            pairs_evaluated += 1
            if pairs_evaluated % 50 == 0:
                print(f"  Progress: {pairs_evaluated}/{total_pairs} pairs evaluated")
                
            # Extract pair tickers and subset of the full data
            pair_tickers = [tickers[i], tickers[j]]
            pair_returns = returns.iloc[:, [i, j]]
            
            try:
                # Calculate autocovariance matrices for this pair
                autocovariance = compute_autocovariance(pair_returns.values, max_lag=max_lag)
                M_0 = autocovariance[0]
                
                # Skip pairs with singular/ill-conditioned covariance
                if np.linalg.det(M_0) < 1e-10 or np.linalg.cond(M_0) > 1e6:
                    continue
                
                if len(autocovariance) <= 1:
                    continue
                    
                M_1 = autocovariance[1]
                
                # Predictability SDP
                pred_weights, _ = optimize_predictability(
                    M_0, M_1, pair_tickers,
                    volatility_threshold=volatility_threshold,  # No threshold for pairs
                    reg_param=0,  # No regularization
                    target_nnz=2,
                    pca_method='regular'  # Use regular PCA
                )
                
                # Calculate predictability score
                pred_score = calculate_predictability_score(M_0, M_1, pred_weights)
                pred_pairs.append((i, j, pred_score, pred_weights))
                
                # Portmanteau SDP
                port_weights, _ = optimize_portmanteau(
                    M_0, autocovariance[1:], pair_tickers,
                    volatility_threshold=volatility_threshold,
                    reg_param=0,
                    max_lag=max_lag,
                    target_nnz=2,
                    pca_method='regular'
                )
                
                # Calculate portmanteau score
                port_score = calculate_portmanteau_score(M_0, autocovariance[1:], port_weights, max_lag)
                port_pairs.append((i, j, port_score, port_weights))
                
                # Crossing SDP
                cross_weights, _ = optimize_crossing(
                    M_0, autocovariance[1:], pair_tickers,
                    volatility_threshold=volatility_threshold,
                    reg_param=0,
                    mu=crossing_mu,
                    max_lag=max_lag,
                    target_nnz=2,
                    pca_method='regular'
                )
                
                # Calculate crossing score
                cross_score = calculate_crossing_score(M_0, M_1, cross_weights)
                cross_pairs.append((i, j, cross_score, cross_weights))
                
            except Exception as e:
                # Skip pairs that cause errors
                print(f"  Error processing pair {pair_tickers}: {str(e)}")
                continue
    
    print(f"Completed pair evaluation. Found {len(pred_pairs)} valid pairs.")
    
    # Sort pairs by their respective scores (ascending)
    pred_pairs.sort(key=lambda x: x[2])
    port_pairs.sort(key=lambda x: x[2])
    cross_pairs.sort(key=lambda x: x[2])
    
    # Select best pair for each criterion
    pred_pair = pred_pairs[0] if pred_pairs else None
    port_pair = port_pairs[0] if port_pairs else None
    cross_pair = cross_pairs[0] if cross_pairs else None
    
    # Report top pairs for each criterion
    print("\nTop pairs by predictability:")
    for i, (idx1, idx2, score, weights) in enumerate(pred_pairs[:num_top_pairs]):
        if i < len(pred_pairs):
            print(f"  {i+1}. {tickers[idx1]} & {tickers[idx2]}: score = {score:.6f}, weights = [{weights[0]:.4f}, {weights[1]:.4f}]")
    
    print("\nTop pairs by portmanteau:")
    for i, (idx1, idx2, score, weights) in enumerate(port_pairs[:num_top_pairs]):
        if i < len(port_pairs):
            print(f"  {i+1}. {tickers[idx1]} & {tickers[idx2]}: score = {score:.6f}, weights = [{weights[0]:.4f}, {weights[1]:.4f}]")
    
    print("\nTop pairs by crossing:")
    for i, (idx1, idx2, score, weights) in enumerate(cross_pairs[:num_top_pairs]):
        if i < len(cross_pairs):
            print(f"  {i+1}. {tickers[idx1]} & {tickers[idx2]}: score = {score:.6f}, weights = [{weights[0]:.4f}, {weights[1]:.4f}]")
    
    return pred_pair, port_pair, cross_pair

def get_optimal_pairs(returns, tickers, volatility_threshold=0.0001, 
                      max_lag=10, crossing_mu=0.1):
    """
    Get optimal pairs for each criterion by running SDP once with target_nnz=2
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        Returns data with stocks as columns
    tickers : list
        List of ticker symbols
    volatility_threshold : float
        Minimum variance threshold
    max_lag : int
        Maximum lag for autocorrelation analysis
    crossing_mu : float
        Weight for higher-order terms in crossing criterion
        
    Returns:
    --------
    pred_pair, port_pair, cross_pair : tuples
        Best pairs for each criterion in format (idx1, idx2, score, weights)
    """
    from portfolio_metrics import compute_autocovariance
    
    print("Finding optimal pairs for each criterion using SDP and sPCA...")
    
    # Calculate autocovariance matrices
    autocovariance = compute_autocovariance(returns.values, max_lag=max_lag)
    M_0 = autocovariance[0]
    M_1 = autocovariance[1]
    
    # Run SDP optimizations with target_nnz=2
    # Predictability
    print("\nOptimizing for Predictability:")
    pred_weights, _ = optimize_predictability(
        M_0, M_1, tickers,  
        volatility_threshold=volatility_threshold,
        reg_param=0.00001,  # Minimal regularization for numerical stability
        target_nnz=2,
        pca_method='sparse'
    )
    
    # Identify the two assets with non-zero weights
    pred_indices = [i for i, w in enumerate(pred_weights) if abs(w) > 1e-6]
    if len(pred_indices) != 2:
        print(f"Warning: Expected 2 assets for predictability, got {len(pred_indices)}")
        return [],[],[]
        
    # Extract just those weights and normalize
    pred_pair_weights = np.array([pred_weights[i] for i in pred_indices])
    pred_pair_weights = pred_pair_weights / np.sum(np.abs(pred_pair_weights))
    
    # Calculate predictability score
    pred_M_0 = M_0[np.ix_(pred_indices, pred_indices)]
    pred_M_1 = M_1[np.ix_(pred_indices, pred_indices)]
    pred_score = calculate_predictability_score(pred_M_0, pred_M_1, pred_pair_weights)
    
    print(f"Predictability pair: {[tickers[i] for i in pred_indices]}")
    print(f"Weights: [{pred_pair_weights[0]:.4f}, {pred_pair_weights[1]:.4f}]")
    print(f"Score: {pred_score:.6f}")
    
    # Portmanteau
    print("\nOptimizing for Portmanteau:")
    port_weights, _ = optimize_portmanteau(
        M_0, autocovariance[1:], tickers,
        volatility_threshold=volatility_threshold,
        reg_param=0.00001,
        max_lag=max_lag,
        target_nnz=2,
        pca_method='sparse'
    )
    
    # Identify the two assets with non-zero weights
    port_indices = [i for i, w in enumerate(port_weights) if abs(w) > 1e-6]
    if len(port_indices) != 2:
        print(f"Warning: Expected 2 assets for portmanteau, got {len(port_indices)}")
        return [],[],[]
    
    # Extract just those weights and normalize
    port_pair_weights = np.array([port_weights[i] for i in port_indices])
    port_pair_weights = port_pair_weights / np.sum(np.abs(port_pair_weights))
    
    # Calculate portmanteau score
    port_M_0 = M_0[np.ix_(port_indices, port_indices)]
    port_autocovariance = [M[np.ix_(port_indices, port_indices)] for M in autocovariance[1:]]
    port_score = calculate_portmanteau_score(port_M_0, port_autocovariance, port_pair_weights, max_lag)
    
    print(f"Portmanteau pair: {[tickers[i] for i in port_indices]}")
    print(f"Weights: [{port_pair_weights[0]:.4f}, {port_pair_weights[1]:.4f}]")
    print(f"Score: {port_score:.6f}")
    
    # Crossing
    print("\nOptimizing for Crossing:")
    cross_weights, _ = optimize_crossing(
        M_0, autocovariance[1:], tickers,
        volatility_threshold=volatility_threshold,
        reg_param=0.00001,
        mu=crossing_mu,
        max_lag=max_lag,
        target_nnz=2,
        pca_method='sparse'
    )
    
    # Identify the two assets with non-zero weights
    cross_indices = [i for i, w in enumerate(cross_weights) if abs(w) > 1e-6]
    if len(cross_indices) != 2:
        print(f"Warning: Expected 2 assets for crossing, got {len(cross_indices)}")
        return [],[],[]
    
    # Extract just those weights and normalize
    cross_pair_weights = np.array([cross_weights[i] for i in cross_indices])
    cross_pair_weights = cross_pair_weights / np.sum(np.abs(cross_pair_weights))
    
    # Calculate crossing score
    cross_M_0 = M_0[np.ix_(cross_indices, cross_indices)]
    cross_M_1 = M_1[np.ix_(cross_indices, cross_indices)]
    cross_score = calculate_crossing_score(cross_M_0, cross_M_1, cross_pair_weights)
    
    print(f"Crossing pair: {[tickers[i] for i in cross_indices]}")
    print(f"Weights: [{cross_pair_weights[0]:.4f}, {cross_pair_weights[1]:.4f}]")
    print(f"Score: {cross_score:.6f}")
    
    # Format the pairs in the same structure as before for compatibility
    pred_pair = (pred_indices[0], pred_indices[1], pred_score, pred_pair_weights)
    port_pair = (port_indices[0], port_indices[1], port_score, port_pair_weights)
    cross_pair = (cross_indices[0], cross_indices[1], cross_score, cross_pair_weights)
    
    return pred_pair, port_pair, cross_pair