import numpy as np
import pandas as pd
from portfolio_metrics import compute_autocovariance, calculate_predictability_score, calculate_portmanteau_score, calculate_crossing_score

def build_portfolio_min_eigenvalue(returns, tickers, best_pair, target_size=5, randomize_factor=0.5, seed=None):
    """
    Greedily build a portfolio starting from the best pair by minimizing the smallest eigenvalue
    of the covariance matrix, with randomization in the candidate pool.
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        Returns data with stocks as columns
    tickers : list
        List of ticker symbols
    best_pair : tuple
        Tuple containing (idx1, idx2, score, weights) of the best pair
    target_size : int
        Target number of assets in the portfolio
    randomize_factor : float
        Factor determining the randomness of the selection
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    portfolio_indices : list
        Indices of selected assets
    portfolio_weights : numpy.ndarray
        Weights for the selected assets
    """
    if seed is not None:
        np.random.seed(seed)
        
    # Extract best pair information
    idx1, idx2, _, init_weights = best_pair
    
    # Initialize portfolio
    portfolio_indices = [idx1, idx2]
    n = returns.shape[1]
    
    # Set a maximum number of attempts to prevent infinite loops
    max_attempts = 50
    attempts = 0
    
    # Continue adding stocks until target size is reached
    while len(portfolio_indices) < target_size and attempts < max_attempts:
        attempts += 1
        
        # Create a random subset of candidate stocks
        available_indices = [i for i in range(n) if i not in portfolio_indices]
        num_to_sample = max(1, int(len(available_indices) * randomize_factor))
        
        # Randomly sample from available stocks
        random_subset = np.random.choice(available_indices, 
                                         size=min(num_to_sample, len(available_indices)), 
                                         replace=False)
        
        candidate_scores = []
        singular_count = 0
        error_count = 0
        
        # Evaluate only the randomly selected stocks
        for i in random_subset:
            # Create candidate portfolio
            candidate_indices = portfolio_indices + [i]
            candidate_returns = returns.iloc[:, candidate_indices].values
            
            # Calculate covariance matrix
            try:
                cov_matrix = np.cov(candidate_returns, rowvar=False)
                
                # Skip if covariance is singular or nearly singular
                if np.linalg.cond(cov_matrix) > 1e6:
                    singular_count += 1
                    continue
                    
                # Calculate minimum eigenvalue
                eigenvalues = np.linalg.eigvalsh(cov_matrix)
                min_eigenvalue = min(eigenvalues)
                
                # Store candidate and score
                candidate_scores.append((i, min_eigenvalue))
                
            except Exception as e:
                error_count += 1
                print(f"  Error evaluating asset {tickers[i]}: {str(e)}")
                continue
        
        if not candidate_scores:
            # If no candidates in random subset, try again with a different subset
            continue
            
        # Sort candidates by minimum eigenvalue (ascending)
        candidate_scores.sort(key=lambda x: x[1])
        
        # Deterministically select the best candidate (smallest eigenvalue)
        best_idx = candidate_scores[0][0]
        
        # Add to portfolio
        portfolio_indices.append(best_idx)
    
    if attempts >= max_attempts:
        print(f"  Reached maximum attempts ({max_attempts}) - using current portfolio with {len(portfolio_indices)} assets")
        
    # Calculate final weights using minimum eigenvalue eigenvector
    portfolio_returns = returns.iloc[:, portfolio_indices].values
    cov_matrix = np.cov(portfolio_returns, rowvar=False)
    
    # Get eigenvector corresponding to minimum eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    portfolio_weights = eigenvectors[:, 0]  # Smallest eigenvalue eigenvector
    
    # Normalize weights
    portfolio_weights = portfolio_weights / np.sum(np.abs(portfolio_weights))
    
    return portfolio_indices, portfolio_weights

def generate_diverse_portfolios(returns, tickers, num_portfolios=10, portfolio_size=5, 
                              max_lag=10, crossing_mu=0.1, volatility_threshold=0.0001,
                              randomize_factor=0.5):
    """
    Generate diverse portfolios using SDP for optimal pairs and greedy minimum eigenvalue search
    
    Parameters:
    -----------
    returns : pandas.DataFrame
        Returns data with stocks as columns
    tickers : list
        List of ticker symbols
    num_portfolios : int
        Number of portfolios to generate
    portfolio_size : int
        Target size of each portfolio
    max_lag : int
        Maximum lag for autocorrelation analysis
    crossing_mu : float
        Weight for higher-order terms in crossing criterion
    volatility_threshold : float
        Minimum variance threshold
    randomize_factor : float
        Factor determining the randomness of the selection
        
    Returns:
    --------
    portfolios : list
        List of portfolio information dictionaries
    """
    from pair_finders import get_optimal_pairs, find_best_pairs_sdp_exhaustive
    
    portfolios = []
    
    # SPCA SDP
    pred_pair, port_pair, cross_pair = get_optimal_pairs(
        returns, tickers, 
        volatility_threshold=volatility_threshold,
        max_lag=max_lag, 
        crossing_mu=crossing_mu
    )

    if not pred_pair or not port_pair or not cross_pair:   ## PCA SDP all pairs if couldn't find sparse
        pred_pair, port_pair, cross_pair = find_best_pairs_sdp_exhaustive(
            returns, tickers, 
            volatility_threshold=volatility_threshold, 
            max_lag=max_lag, 
            crossing_mu=crossing_mu, 
            num_top_pairs=5
        )
    
    # Check if we found valid pairs
    if not all([pred_pair, port_pair, cross_pair]):
        print("Not enough valid pairs found. Exiting.")
        return []
    
    # Generate diverse portfolios
    for port_idx in range(num_portfolios):
        # Set different seeds for each portfolio
        seed_base = 42 + port_idx*100
        np.random.seed(seed_base)
        
        # Build portfolios using minimum eigenvalue greedy search
        pred_indices, pred_weights = build_portfolio_min_eigenvalue(
            returns, tickers, pred_pair, 
            target_size=portfolio_size,
            randomize_factor=randomize_factor,
            seed=seed_base
        )
        
        port_indices, port_weights = build_portfolio_min_eigenvalue(
            returns, tickers, port_pair, 
            target_size=portfolio_size,
            randomize_factor=randomize_factor,
            seed=seed_base+100
        )
        
        cross_indices, cross_weights = build_portfolio_min_eigenvalue(
            returns, tickers, cross_pair, 
            target_size=portfolio_size,
            randomize_factor=randomize_factor,
            seed=seed_base+200
        )
        
        # Combine all unique indices from the three portfolios
        all_indices = sorted(list(set(pred_indices + port_indices + cross_indices)))
        all_tickers = [tickers[i] for i in all_indices]
        
        # Map weights to combined universe
        pred_full = np.zeros(len(all_indices))
        port_full = np.zeros(len(all_indices))
        cross_full = np.zeros(len(all_indices))
        
        for i, idx in enumerate(pred_indices):
            full_idx = all_indices.index(idx)
            pred_full[full_idx] = pred_weights[i]
            
        for i, idx in enumerate(port_indices):
            full_idx = all_indices.index(idx)
            port_full[full_idx] = port_weights[i]
            
        for i, idx in enumerate(cross_indices):
            full_idx = all_indices.index(idx)
            cross_full[full_idx] = cross_weights[i]
        
        # Normalize each weight vector
        for weight_vec in [pred_full, port_full, cross_full]:
            if np.sum(np.abs(weight_vec)) > 0:
                weight_vec /= np.sum(np.abs(weight_vec))
        
        # Store criterion-specific portfolio information
        criterion_portfolios = {
            'predictability': {'indices': pred_indices, 'weights': pred_full, 'tickers': all_tickers},
            'portmanteau': {'indices': port_indices, 'weights': port_full, 'tickers': all_tickers},
            'crossing': {'indices': cross_indices, 'weights': cross_full, 'tickers': all_tickers}
        }
        
        # Store the combined portfolio information
        portfolios.append({
            'id': port_idx + 1,
            'indices': all_indices,
            'tickers': all_tickers,
            'criterion_portfolios': criterion_portfolios
        })
    
    return portfolios