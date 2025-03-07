import numpy as np
import cvxpy as cp
from sklearn.decomposition import SparsePCA

def optimize_sdp_portfolio(M_0, obj_func, constraints_func, tickers, volatility_threshold=0.0001, 
                          reg_param=0.0001, target_nnz=2, pca_method='sparse', verbose=False):
    """
    General SDP optimization function for mean-reverting portfolios
    
    Parameters:
    -----------
    M_0 : numpy array
        Covariance matrix
    obj_func : function
        Function that takes Y and returns the objective part specific to the method
    constraints_func : function
        Function that adds any additional constraints specific to the method
    tickers : list
        List of ticker symbols
    volatility_threshold : float
        Minimum variance constraint
    reg_param : float
        L1 regularization parameter
    target_nnz : int
        Target number of non-zero components in the sparse solution
    verbose : bool
        Whether to print verbose output
        
    Returns:
    --------
    weights : numpy array
        Optimized sparse weights
    Y_opt : numpy array
        Optimal Y matrix from SDP
    """
    n = M_0.shape[0]
    if verbose:
        print(f"  Matrix dimensions: {n}x{n}")
    
    # Ensure M_0 is well-conditioned
    M_0_reg = M_0 + 1e-4 * np.eye(n)
    
    # Create SDP variable
    Y = cp.Variable((n, n), symmetric=True)
    
    # Get objective function specific to the method
    obj_term = obj_func(Y)
    
    # Define objective with L1 regularization
    objective = cp.Minimize(obj_term + reg_param * cp.sum(cp.abs(Y)))
    
    # Define base constraints: Tr(A₀Y) ≥ ν, Tr(Y) = 1, Y ≽ 0
    constraints = [
        cp.trace(M_0_reg @ Y) >= volatility_threshold,
        cp.trace(Y) == 1,
        Y >> 0  # Positive semidefinite
    ]
    
    # Add method-specific constraints if any
    constraints_func(Y, constraints)
    
    # Solve the SDP problem
    problem = cp.Problem(objective, constraints)
    
    try:
        if verbose:
            print(f"  Solving SDP problem...")
        problem.solve(solver=cp.SCS, eps=1e-6, alpha=1.8, normalize=True, max_iters=10000, verbose=verbose)
        
        if verbose:
            print(f"  SDP problem status: {problem.status}, objective value: {problem.value}")
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            if verbose:
                print(f"  Warning: Optimization did not reach optimality.")
            return np.zeros(n), None
        
        Y_opt = Y.value
        
        if Y_opt is None:
            if verbose:
                print(f"  Error: No solution found")
            return np.zeros(n), None

        if pca_method=='sparse':
             # Ensure Y_opt is symmetric and PSD for SparsePCA
            Y_opt = (Y_opt + Y_opt.T) / 2
            eigvals = np.linalg.eigvalsh(Y_opt)
            min_eig = np.min(eigvals)
            if min_eig < 0:
                Y_opt = Y_opt - min_eig * np.eye(n) + 1e-10 * np.eye(n)
            
            if verbose:
                print(f"  Y_opt shape: {Y_opt.shape}, trace: {np.trace(Y_opt):.6f}")
                print(f"  Y_opt eigenvalues range: min={min_eig:.6e}, max={np.max(eigvals):.6e}")

            best_weights = extract_sparse_weights(Y_opt, target_nnz, verbose)
        else:
            best_weights = extract_regular_pca_weights(Y_opt, verbose)

        # Normalize weights to sum of absolute values = 1
        if np.sum(np.abs(best_weights)) > 0:
            best_weights = best_weights / np.sum(np.abs(best_weights))
        
        return best_weights, Y_opt
        
    except Exception as e:
        if verbose:
            print(f"  Error in optimization: {str(e)}")
        return np.zeros(n), None


def extract_sparse_weights(Y_opt, target_nnz=2, verbose=False):
    """Extract sparse weights from Y matrix using SparsePCA"""
    n = Y_opt.shape[0]
    
    # Binary search for the right alpha value
    alpha_min = 1e-8
    alpha_max = 1
    best_weights = None
    best_nnz = 0
    
    if verbose:
        print(f"  Using binary search to find SparsePCA alpha parameter...")
    
    for _ in range(20):  # Try up to 20 iterations
        alpha = np.sqrt(alpha_min * alpha_max)
        spca = SparsePCA(n_components=1, alpha=alpha, ridge_alpha=0.01, 
                       max_iter=1000, random_state=42)
        
        try:
            weights = spca.fit_transform(Y_opt)
            weights = spca.components_[0]  # Get the first component
            
            # Count non-zero elements
            nnz = np.sum(np.abs(weights) > 1e-5)
            if verbose:
                print(f"    alpha={alpha:.6e}, non-zeros: {nnz}")
            
            if nnz == target_nnz:
                best_weights = weights
                best_nnz = nnz
                break
            elif nnz < target_nnz:
                alpha_max = alpha  # Need less sparsity, lower alpha
            else:
                alpha_min = alpha  # Need more sparsity, higher alpha
                
            # Keep the best we've seen if it's close to target
            if best_weights is None or abs(nnz - target_nnz) < abs(best_nnz - target_nnz):
                best_weights = weights
                best_nnz = nnz
                
        except Exception as e:
            if verbose:
                print(f"    SparsePCA error: {str(e)}")
            alpha_min = alpha  # Try a higher alpha next time
    
    if best_weights is None:
        if verbose:
            print(f"  Failed to find suitable sparse weights")
        return np.zeros(n)
        
    # If we didn't get exactly target_nnz non-zeros, force it
    if best_nnz != target_nnz:
        if verbose:
            print(f"  Forcing exactly {target_nnz} non-zeros (closest was {best_nnz})")
        indices = np.argsort(np.abs(best_weights))[::-1]
        sparse_weights = np.zeros(n)
        sparse_weights[indices[:target_nnz]] = best_weights[indices[:target_nnz]]
        best_weights = sparse_weights
    
    return best_weights

def extract_regular_pca_weights(Y_opt, verbose=False):
    """Extract weights from Y matrix using regular PCA - using minimum eigenvalue for mean reversion"""
    if verbose:
        print(f"  Using regular PCA to extract weights")
    
    # Get the eigenvector corresponding to the smallest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(Y_opt)
    # Get index of the smallest eigenvalue
    idx = np.argmin(eigenvalues)
    weights = eigenvectors[:, idx]
    
    if verbose:
        print(f"  PCA eigenvalues range: min={np.min(eigenvalues):.6e}, max={np.max(eigenvalues):.6e}")
        print(f"  Using eigenvector corresponding to eigenvalue: {eigenvalues[idx]:.6e}")
    
    return weights

def optimize_predictability(M_0, M_1, tickers, volatility_threshold=0.0001, reg_param=0.0001, 
                           target_nnz=2, pca_method='regular'):
    """Solve the predictability optimization problem using SDP relaxation"""
    n = M_0.shape[0]
    
    # Calculate median variance of assets for reference
    median_variance = np.median(np.diag(M_0))
    
    # Calculate M = M_1 * M_0^(-1) * M_1^T for predictability
    M_0_reg = M_0 + 1e-4 * np.eye(n)  # Regularize for stability
    try:
        M_0_inv = np.linalg.inv(M_0_reg)
        M = M_1 @ M_0_inv @ M_1.T
    except np.linalg.LinAlgError:
        print(f"  Matrix inversion failed. Using pseudoinverse.")
        M_0_inv = np.linalg.pinv(M_0_reg)
        M = M_1 @ M_0_inv @ M_1.T
    
    # Ensure M is symmetric
    M = (M + M.T) / 2
    
    # Define objective function for predictability
    def obj_func(Y):
        return cp.trace(M @ Y)
    
    # No additional constraints
    def constraints_func(Y, constraints):
        pass
    
    return optimize_sdp_portfolio(M_0, obj_func, constraints_func, tickers, 
                                 volatility_threshold, reg_param, target_nnz, pca_method)

def optimize_portmanteau(M_0, M_autocovariance, tickers, volatility_threshold=0.0001, 
                        reg_param=0.0001, max_lag=10, target_nnz=2, pca_method='regular'):
    """Solve the portmanteau optimization problem using SDP relaxation"""
    n = M_0.shape[0]
    
    # Use at most max_lag autocovariance matrices
    p = min(max_lag, len(M_autocovariance))
    
    # Define objective function for portmanteau
    def obj_func(Y):
        portmanteau_terms = []
        for i in range(p):
            if i < len(M_autocovariance):
                M_i = M_autocovariance[i]
                portmanteau_terms.append(cp.square(cp.trace(M_i @ Y)))
        return cp.sum(portmanteau_terms)
    
    # No additional constraints
    def constraints_func(Y, constraints):
        pass
    
    return optimize_sdp_portfolio(M_0, obj_func, constraints_func, tickers, 
                                 volatility_threshold, reg_param, target_nnz, pca_method)

def optimize_crossing(M_0, M_autocovariance, tickers, volatility_threshold=0.0001, 
                     reg_param=0.0001, mu=0.1, max_lag=5, target_nnz=2, pca_method='regular'):
    """Solve the crossing statistics optimization problem using SDP relaxation"""
    n = M_0.shape[0]
    
    # Extract first-order autocovariance matrix
    M_1 = M_autocovariance[0]
    
    # Use at most max_lag autocovariance matrices for higher-order terms
    p = min(max_lag, len(M_autocovariance))
    
    # Define objective function for crossing statistics
    def obj_func(Y):
        # First term: minimize first-order autocorrelation
        first_term = cp.trace(M_1 @ Y)
        
        # Higher-order terms: squared autocorrelations at lags 2 to p
        higher_terms = []
        for i in range(1, p):
            if i < len(M_autocovariance):
                M_i = M_autocovariance[i]
                higher_terms.append(cp.square(cp.trace(M_i @ Y)))
        
        # Combine terms with mu weight for higher-order terms
        if higher_terms:
            return first_term + mu * cp.sum(higher_terms)
        else:
            return first_term
    
    # No additional constraints
    def constraints_func(Y, constraints):
        pass
    
    return optimize_sdp_portfolio(M_0, obj_func, constraints_func, tickers, 
                                 volatility_threshold, reg_param, target_nnz, pca_method)