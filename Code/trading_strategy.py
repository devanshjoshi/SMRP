import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from portfolio_metrics import truncate_portfolio

class StatisticalArbitrageStrategy:
    """
    Statistical Arbitrage Strategy class that implements mean-reverting portfolio
    trading with proper backtesting across reference, training, validation, and test periods.
    """
        
    def __init__(self, lookback_period=20, std_dev=1.5, adf_threshold=0.1, 
                stop_loss_pct=-0.01, take_profit_pct=0.03, 
                optimization_method='predictability', max_lag=10, 
                crossing_mu=0.1, crossing_max_lag=5, pca_method='regular', 
                target_nnz=2, **kwargs):
        """
        Initialize the strategy with the given parameters.
        
        Parameters:
        -----------
        lookback_period : int
            Window for moving average calculation
        std_dev : float
            Number of standard deviations for Bollinger Bands
        adf_threshold : float
            Significance level for ADF test
        stop_loss_pct : float
            Stop loss threshold as percentage
        take_profit_pct : float
            Take profit threshold as percentage
        optimization_method : str
            Method to use for portfolio optimization ('predictability', 'portmanteau', or 'crossing')
        """
        self.lookback_period = lookback_period
        self.std_dev = std_dev
        self.adf_threshold = adf_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.weights = None
        self.volatility_threshold = None
        self.tickers = None
        self.max_lag = max_lag ## for portmanteau
        self.crossing_mu = crossing_mu  
        self.crossing_max_lag = crossing_max_lag  ## for crossing
        self.pca_method = pca_method
        self.target_nnz = target_nnz
        
        # Set optimization method
        if optimization_method not in ['predictability', 'portmanteau', 'crossing']:
            raise ValueError("optimization_method must be one of: 'predictability', 'portmanteau', 'crossing'")
        self.optimization_method = optimization_method
    
    def split_data(self, df, start_date, end_date):
        """
        Split the data into reference, training, validation, and test periods
        """
        # Filter data to date range
        df_filtered = df.loc[start_date:end_date].copy()
        
        # Calculate total period duration in days
        total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        
        # Calculate period durations
        reference_days = int(total_days * (1/9)) 
        training_days = int(total_days * (4/9))  
        validation_days = int(total_days * (2/9))
        
        # Calculate period dates
        reference_end = pd.to_datetime(start_date) + pd.Timedelta(days=reference_days)
        training_end = reference_end + pd.Timedelta(days=training_days)
        validation_end = training_end + pd.Timedelta(days=validation_days)
        
        # Convert back to string format
        reference_end_str = reference_end.strftime('%Y-%m-%d')
        training_end_str = training_end.strftime('%Y-%m-%d')
        validation_end_str = validation_end.strftime('%Y-%m-%d')
        
        print(f"Period breakdown:")
        print(f"- Reference period: {start_date} to {reference_end_str}")
        print(f"- Training period: {reference_end_str} to {training_end_str}")
        print(f"- Validation period: {training_end_str} to {validation_end_str}")
        print(f"- Test period: {validation_end_str} to {end_date}")
        
        # Split data into periods
        reference_data = df_filtered.loc[start_date:reference_end_str]
        training_data = df_filtered.loc[reference_end_str:training_end_str]
        validation_data = df_filtered.loc[training_end_str:validation_end_str]
        test_data = df_filtered.loc[validation_end_str:end_date]
        
        return reference_data, training_data, validation_data, test_data
    
    def calculate_spread(self, prices):
        """
        Calculate portfolio spread using the stored non-zero weights and tickers
        """
        if self.weights is None or self.tickers is None:
            raise ValueError("Portfolio weights and tickers not set")
        
        # Ensure all tickers are in the prices DataFrame
        valid_tickers = [t for t in self.tickers if t in prices.columns]
        if len(valid_tickers) != len(self.tickers):
            missing = set(self.tickers) - set(valid_tickers)
            print(f"Warning: {len(missing)} tickers not found in price data: {missing}")
        
        if not valid_tickers:
            raise ValueError("No valid tickers with non-zero weights found in price data")
        
        # Get corresponding weights for valid tickers
        valid_weights = [w for t, w in zip(self.tickers, self.weights) if t in valid_tickers]
        
        # Filter price data for valid tickers
        price_data = prices[valid_tickers]
        
        # Calculate spread
        spread = price_data.dot(valid_weights)
        
        return spread, valid_tickers, valid_weights
    
    def set_volatility_threshold(self, reference_data):
        """
        Set volatility threshold based on reference period data
        """
        # Calculate returns
        returns = reference_data.pct_change().dropna()
        
        # Calculate median variance
        median_variance = np.median(np.var(returns, axis=0))
        
        # Set volatility threshold to a percentage of median variance
        self.volatility_threshold = 0.1 * median_variance
        
        print(f"Median variance: {median_variance:.6f}")
        print(f"Volatility threshold: {self.volatility_threshold:.6f}")
    
    def test_stationarity(self, spread):
        """Test if spread is stationary using ADF test"""
        result = adfuller(spread.dropna())
        p_value = result[1]
        adf_stat = result[0]
        is_stationary = p_value < self.adf_threshold
        
        return is_stationary, p_value, adf_stat
    
    def generate_signals(self, spread):
        """Generate trading signals using Bollinger Bands"""
        span = 2 * self.lookback_period - 1
        ewma = spread.ewm(span=span, min_periods=self.lookback_period).mean()
        ewm_std = spread.ewm(span=span, min_periods=self.lookback_period).std(bias=False)
        upper_band = ewma + (self.std_dev * ewm_std)
        lower_band = ewma - (self.std_dev * ewm_std)
        
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['ma'] = ewma
        signals['upper'] = upper_band
        signals['lower'] = lower_band
        
        # Initialize positions and other tracking variables
        positions = np.zeros(len(spread))
        position = 0  # Current position tracker
        entry_price = None  # Price at which position was entered
        entry_date = None  # Date position was entered
        pnl = 0  # Cumulative PnL
        active = True  # Flag for whether position is active
        last_reset = 0  # Index of last reset due to stop/take profit
        
        # Generate signals
        for i in range(self.lookback_period, len(spread)):
            current_date = spread.index[i]
            current_price = spread.iloc[i]
            
            # Only proceed if we have valid EWMA and bands
            if pd.notna(ewma.iloc[i]) and pd.notna(upper_band.iloc[i]) and pd.notna(lower_band.iloc[i]):
                current_ma = ewma.iloc[i]
                upper = upper_band.iloc[i]
                lower = lower_band.iloc[i]
                
                # Check if we need to exit based on PnL thresholds
                if position != 0 and entry_price is not None and active:
                    # Calculate PnL based on price change
                    pnl_pct = (current_price - entry_price) / abs(entry_price)
                    
                    # For short positions, negate the PnL calculation
                    if position == -1:
                        pnl_pct = -pnl_pct
                    
                    # Check stop loss
                    if pnl_pct <= self.stop_loss_pct:
                        position = 0
                        entry_price = None
                        entry_date = None
                        active = False
                        last_reset = i
                    
                    # Check take profit
                    elif pnl_pct >= self.take_profit_pct:
                        position = 0
                        entry_price = None
                        entry_date = None
                        active = False
                        last_reset = i
                
                # Allow re-entry after certain period (ignore for now)
                active = True
                
                # If no position and active, check for entry signals
                if position == 0 and active:
                    if current_price >= upper:  # Short signal
                        position = -1
                        entry_price = current_price
                        entry_date = current_date
                    elif current_price <= lower:  # Long signal
                        position = 1
                        entry_price = current_price
                        entry_date = current_date
                
                # If in a position, check mean reversion exit
                elif position == -1:  # Short position
                    if current_price <= current_ma:  # Exit at mean
                        position = 0
                        entry_price = None
                        entry_date = None
                elif position == 1:  # Long position
                    if current_price >= current_ma:  # Exit at mean
                        position = 0
                        entry_price = None
                        entry_date = None
            
            positions[i] = position
        
        # Add positions to signals DataFrame
        signals['positions'] = positions
        
        # Calculate daily PnL
        signals['spread_change'] = spread.pct_change(1).shift(-1).dropna()   
        signals['pnl'] = (signals['positions'].shift(1) * signals['spread_change']).dropna()
        
        # Calculate cumulative PnL
        signals['cumulative_pnl'] = signals['pnl'].cumsum()
        
        return signals
    
    def calculate_returns(self, signals):
        """Calculate strategy returns from signals"""
        returns = signals['pnl'].copy()
        return returns
    
    def calculate_performance_metrics(self, returns):
        """Calculate performance metrics"""
        if len(returns) == 0 or returns.isna().all():
            return {
                'total_return': 0,
                'annualized_return': 0,
                'annualized_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
        
        # Calculate returns
        returns = returns.dropna()
        if len(returns) == 0:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'annualized_volatility': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0
            }
        
        # Calculate cumulative returns
        cumulative_returns = (1 + returns).cumprod()
        
        # Total return
        total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
        
        # Annualized metrics (assuming daily data)
        days = len(returns)
        annualized_return = (1 + total_return) ** (252 / days) - 1 if (days > 0 and total_return>0) else 0
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio (using risk-free rate of 0.045)
        risk_free_rate = 0.045
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Max drawdown
        drawdown = cumulative_returns / cumulative_returns.cummax() - 1
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate=len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
        
        # Number of trades
        position_changes = returns.diff().fillna(0) != 0
        num_trades = position_changes.sum()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades
        }
    
    def plot_performance(self, signals, title):
        """Plot strategy performance"""
        plt.figure(figsize=(15, 12))
        
        # Plot 1: Spread and Bollinger Bands
        plt.subplot(3, 1, 1)
        plt.plot(signals['spread'], label='Spread', color='blue', alpha=0.7)
        plt.plot(signals['ma'], label='Moving Average', color='black')
        plt.plot(signals['upper'], label='Upper Band', color='red', linestyle='--')
        plt.plot(signals['lower'], label='Lower Band', color='green', linestyle='--')
        plt.fill_between(signals.index, signals['upper'], signals['lower'], color='gray', alpha=0.1)
        
        # Find entries and exits
        long_entries = signals.index[(signals['positions'].shift(1) == 0) & (signals['positions'] == 1)]
        short_entries = signals.index[(signals['positions'].shift(1) == 0) & (signals['positions'] == -1)]
        long_exits = signals.index[(signals['positions'].shift(1) == 1) & (signals['positions'] == 0)]
        short_exits = signals.index[(signals['positions'].shift(1) == -1) & (signals['positions'] == 0)]
        
        # Plot entries and exits
        plt.scatter(long_entries, signals.loc[long_entries, 'spread'], marker='^', color='green', s=100, label='Long Entry')
        plt.scatter(short_entries, signals.loc[short_entries, 'spread'], marker='v', color='red', s=100, label='Short Entry')
        plt.scatter(long_exits, signals.loc[long_exits, 'spread'], marker='X', color='black', s=80, label='Exit')
        plt.scatter(short_exits, signals.loc[short_exits, 'spread'], marker='X', color='black', s=80)
        
        plt.title(f'Spread and Bollinger Bands - {title}')
        plt.legend()
        
        # Plot 2: Positions
        plt.subplot(3, 1, 2)
        plt.plot(signals['positions'], label='Position', color='purple')
        plt.title(f'Positions (-1: Short, 0: Neutral, 1: Long) - {title}')
        plt.legend()
        
        # Plot 3: Cumulative PnL
        plt.subplot(3, 1, 3)
        cumulative_pnl = (1 + signals['pnl'].dropna()).cumprod()
        plt.plot(cumulative_pnl, label='Cumulative Return', color='green')
        plt.title(f'Cumulative Return - {title}')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

def run_diverse_portfolio_analysis(df, start_date, end_date, tickers, 
                                 num_portfolios=10, portfolio_size=5, 
                                 max_lag=10, crossing_mu=0.1, 
                                 randomize_factor=0.3, pgp_lambda=[1, 1, 1],
                                 lookback=20, std_dev=1.5, adf_threshold=0.05,
                                 stop_loss_pct=-0.01, take_profit_pct=0.03,
                                 top_n=3):
    """
    Run complete analysis with diverse mean-reverting portfolios
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data with Date index
    start_date, end_date : str
        Start and end dates for the analysis
    tickers : list
        List of tickers to include
    num_portfolios, portfolio_size : int
        Number of portfolios to generate and size of each
    Other parameters: various model and strategy parameters
        
    Returns:
    --------
    dict
        Analysis results
    """
    from portfolio_builders import generate_diverse_portfolios
    
    # Instantiate the strategy
    strategy = StatisticalArbitrageStrategy(
        lookback_period=lookback,
        std_dev=std_dev,
        adf_threshold=adf_threshold,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )
    
    # Split data into periods
    reference_data, training_data, validation_data, test_data = strategy.split_data(df, start_date, end_date)
    
    # Set volatility threshold based on reference period
    strategy.set_volatility_threshold(reference_data)
    
    # Calculate returns for training data
    training_returns = training_data.pct_change().dropna()
    
    # Generate diverse portfolios
    print(f"\nGenerating {num_portfolios} diverse portfolios...")
    portfolios = generate_diverse_portfolios(
        training_returns, tickers, 
        num_portfolios=num_portfolios, 
        portfolio_size=portfolio_size,
        max_lag=max_lag, 
        crossing_mu=crossing_mu,
        volatility_threshold=strategy.volatility_threshold,
        randomize_factor=randomize_factor
    )
    
    if not portfolios:
        print("No valid portfolios generated. Exiting.")
        return None
    
    truncated_portfolios = []
    idx = 1
    for port in portfolios:
        # Extract weights and tickers for each criterion
        weights_port = port['criterion_portfolios']['portmanteau']['weights']
        tickers_port = port['criterion_portfolios']['portmanteau']['tickers']
        norm_factor = training_data[tickers].mean().mean()
        
        # Apply truncation
        trunc_weights, trunc_tickers = truncate_portfolio(
            weights_port, tickers_port, norm_factor, max_assets=5
        )
        
        # Create a new portfolio entry with truncated data
        truncated_portfolio = {
            'id': idx,
            'tickers': trunc_tickers,
            'weights': trunc_weights
        }
        truncated_portfolios.append(truncated_portfolio)
        idx += 1

        # Do the same for predictability
        weights_pred = port['criterion_portfolios']['predictability']['weights']
        tickers_pred = port['criterion_portfolios']['predictability']['tickers']
        
        trunc_weights, trunc_tickers = truncate_portfolio(
            weights_pred, tickers_pred, norm_factor, max_assets=5
        )
        
        truncated_portfolio = {
            'id': idx,
            'tickers': trunc_tickers,
            'weights': trunc_weights
        }
        truncated_portfolios.append(truncated_portfolio)
        idx += 1

        # Do the same for crossing
        weights_cross = port['criterion_portfolios']['crossing']['weights']
        tickers_cross = port['criterion_portfolios']['crossing']['tickers']
        
        trunc_weights, trunc_tickers = truncate_portfolio(
            weights_cross, tickers_cross, norm_factor, max_assets=5
        )
        
        truncated_portfolio = {
            'id': idx,
            'tickers': trunc_tickers,
            'weights': trunc_weights
        }
        truncated_portfolios.append(truncated_portfolio)
        idx += 1

    # Test each portfolio
    portfolio_results = []
    
    for portfolio in truncated_portfolios:
        portfolio_id = portfolio['id']
        print(f"\nTesting portfolio {portfolio_id}...")
        
        # Set weights and tickers in strategy
        strategy.weights = portfolio['weights']
        strategy.tickers = portfolio['tickers']
        
        # Generate spread for training and validation
        training_spread, _, _ = strategy.calculate_spread(training_data)
        validation_spread, _, _ = strategy.calculate_spread(validation_data)
        
        # Check profitability
        training_signals = strategy.generate_signals(training_spread)
        training_returns_port = strategy.calculate_returns(training_signals)
        training_metrics = strategy.calculate_performance_metrics(training_returns_port)
        
        validation_signals = strategy.generate_signals(validation_spread)
        validation_returns_port = strategy.calculate_returns(validation_signals)
        validation_metrics = strategy.calculate_performance_metrics(validation_returns_port)
        
        # Check stationarity
        _, train_pvalue, train_adf = strategy.test_stationarity(training_spread)
        _, val_pvalue, val_adf = strategy.test_stationarity(validation_spread)
        
        # Portfolio passed checks?
        passed_return_check = training_metrics['total_return'] > 0 and validation_metrics['total_return'] > 0
        passed_adf_check = train_pvalue < adf_threshold and val_pvalue < adf_threshold

        print("Return check pass?", passed_return_check)
        print("ADF pass?", passed_adf_check)
        
        # Store results
        portfolio_results.append({
            'id': portfolio_id,
            'tickers': portfolio['tickers'],
            'weights': portfolio['weights'],
            'training_metrics': training_metrics,
            'validation_metrics': validation_metrics,
            'train_pvalue': train_pvalue,
            'val_pvalue': val_pvalue,
            'passed_checks': passed_return_check and passed_adf_check
        })
    
    # Select top portfolios that passed checks
    passed_portfolios = [p for p in portfolio_results if p['passed_checks']]

    if not passed_portfolios:
        print("\nNo portfolios passed both checks. Exiting.")
        return {
            'portfolios': portfolio_results,
            'selected_portfolios': [],
            'test_performance': None
        }
    
    # Sort by ADF p-value (ascending)
    passed_portfolios.sort(key=lambda x: (x['train_pvalue'] + x['val_pvalue'])/2)
    
    # Select top N portfolios
    selected_portfolios = passed_portfolios[:min(top_n, len(passed_portfolios))]
    
    print(f"\nSelected {len(selected_portfolios)} out of {len(passed_portfolios)} portfolios that passed checks:")
    for portfolio in selected_portfolios:
        print(f"  Portfolio {portfolio['id']}: ADF p-value = {portfolio['train_pvalue']:.6f} (train), {portfolio['val_pvalue']:.6f} (val)")
        print(f"    Tickers: {portfolio['tickers']}")
        print(f"    Weights: {[round(w, 4) for w in portfolio['weights']]}")
    
    # Run test period analysis for selected portfolios
    test_results = []
    
    for portfolio in selected_portfolios:
        strategy.weights = portfolio['weights']
        strategy.tickers = portfolio['tickers']
        
        test_spread, _, _ = strategy.calculate_spread(test_data)
        test_signals = strategy.generate_signals(test_spread)
        test_returns_port = strategy.calculate_returns(test_signals)
        test_metrics = strategy.calculate_performance_metrics(test_returns_port)
        strategy.plot_performance(test_signals, "Test Performance")
        
        test_results.append({
            'id': portfolio['id'],
            'signals': test_signals,
            'metrics': test_metrics
        })
    
    # Combine test results
    if test_results:
        try:
            # Extract the signals DataFrames
            signals_list = []
            for r in test_results:
                if 'signals' in r:
                    signals_list.append(r['signals'])
            
            if signals_list:
                combined_test_results = combine_test_results(signals_list, equal_weight=True)
            else:
                print("No valid signals found in test results")
                combined_test_results = None
        except Exception as e:
            print(f"Error combining test results: {str(e)}")
            combined_test_results = None
    else:
        combined_test_results = None
    
    return {
        'portfolios': portfolio_results,
        'selected_portfolios': selected_portfolios,
        'test_results': test_results,
        'combined_test_results': combined_test_results
    }

def combine_test_results(signals_list, equal_weight=True):
    """
    Combine test results from multiple portfolios
    
    Parameters:
    -----------
    signals_list : list
        List of signals DataFrames from different portfolios
    equal_weight : bool
        Whether to weight portfolios equally
        
    Returns:
    --------
    dict
        Combined test results
    """
    if not signals_list:
        return None
    
    # Get all unique dates across all portfolios
    all_dates = set()
    for signals in signals_list:
        all_dates.update(signals.index)
    all_dates = sorted(all_dates)
    
    # Create DataFrame with all dates
    combined_signals = pd.DataFrame(index=all_dates)
    
    # Calculate weights
    weights = [1/len(signals_list)] * len(signals_list) if equal_weight else None
    
    # Combine signals and PnL
    for i, signals in enumerate(signals_list):
        weight = weights[i] if weights else 1.0
        
        # Reindex to align dates
        reindexed = signals.reindex(all_dates)
        
        # Add weighted PnL to combined signals
        pnl_col = f'pnl_{i}'
        combined_signals[pnl_col] = reindexed['pnl'] * weight
    
    # Sum PnL across portfolios
    pnl_cols = [col for col in combined_signals.columns if col.startswith('pnl_')]
    combined_signals['pnl'] = combined_signals[pnl_cols].sum(axis=1)
    
    # Calculate cumulative PnL
    combined_signals['cumulative_pnl'] = combined_signals['pnl'].cumsum()
    combined_signals['cumulative_return'] = (1 + combined_signals['pnl']).cumprod() 
    
    # Calculate performance metrics for the combined portfolio
    returns = combined_signals['pnl'].dropna()
    metrics = {
        'total_return': (1 + returns).prod() - 1 if len(returns) > 0 else 0,
        'annualized_return': ((1 + returns).prod() ** (252 / len(returns)) - 1) if len(returns) > 0 and (1 + returns).prod() - 1 > 0 else 0,
        'annualized_volatility': returns.std() * np.sqrt(252),
        'sharpe_ratio': (((1 + returns).prod() ** (252 / len(returns)) - 1) - 0.045) / (returns.std() * np.sqrt(252)) if len(returns) > 0 and returns.std() > 0 else 0,
        'max_drawdown': ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min(),
        'win_rate': len(returns[returns > 0]) / len(returns[returns != 0]) if len(returns[returns != 0]) > 0 else 0
    }
    
    return {
        'signals': combined_signals,
        'metrics': metrics
    }