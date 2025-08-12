"""
Backtesting Engine for LSTM Trading Strategies
Handles NAV simulation, transaction costs, and walk-forward backtesting
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from datetime import datetime
import json


class BacktestEngine:
    """
    Main backtesting engine with transaction costs and slippage
    """
    
    def __init__(self, 
                 initial_capital: float = 1000000,
                 transaction_cost: float = 0.001,  # 0.1% per trade
                 slippage: float = 0.0005):        # 0.05% market impact
        """
        Initialize backtesting engine
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction
            slippage: Market impact as fraction
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        
        # Track trades for analysis
        self.trade_log = []
        self.daily_positions = []
    
    def simulate_nav(self,
                    prices: pd.Series,
                    signals: pd.Series,
                    capital: Optional[float] = None) -> Tuple[pd.Series, Dict]:
        """
        Simulate portfolio NAV with transaction costs
        
        Args:
            prices: Price series (close prices)
            signals: Trading signals (0=cash, 1=long, or fractional for scaling)
            capital: Initial capital (uses self.initial_capital if None)
            
        Returns:
            nav_series: Daily NAV values
            stats: Dictionary of statistics
        """
        if capital is None:
            capital = self.initial_capital
        
        # Align prices and signals
        aligned = pd.DataFrame({
            'price': prices,
            'signal': signals
        }).dropna()
        
        # Initialize tracking
        cash = capital
        units = 0
        nav = []
        trades = []
        positions = []
        costs = []
        
        # Shift signals to avoid look-ahead bias
        # Signal at time t determines position for t+1
        aligned['position'] = aligned['signal'].shift(1).fillna(0)
        
        for idx, row in aligned.iterrows():
            price = row['price']
            target_position = row['position']
            
            # Determine if we need to trade
            # For binary signals: 1 = fully invested, 0 = all cash
            if target_position > 0 and units == 0:
                # Buy signal and we're in cash
                target_units = (cash - cash * self.transaction_cost) / (price * (1 + self.slippage))
            elif target_position == 0 and units > 0:
                # Sell signal and we have position
                target_units = 0
            else:
                # No change needed
                target_units = units
            
            # Execute trades if position changes
            if target_units != units:
                trade_units = target_units - units
                
                if trade_units > 0:  # Buy
                    # Apply slippage (buy at higher price)
                    exec_price = price * (1 + self.slippage)
                    trade_value = trade_units * exec_price
                    trade_cost = trade_value * self.transaction_cost
                    
                    # Check if we have enough cash
                    if cash >= trade_value + trade_cost:
                        cash -= (trade_value + trade_cost)
                        units = target_units
                        trades.append({
                            'date': idx,
                            'action': 'buy',
                            'units': trade_units,
                            'price': exec_price,
                            'cost': trade_cost
                        })
                        costs.append(trade_cost)
                
                elif trade_units < 0:  # Sell
                    # Apply slippage (sell at lower price)
                    exec_price = price * (1 - self.slippage)
                    trade_value = abs(trade_units) * exec_price
                    trade_cost = trade_value * self.transaction_cost
                    
                    cash += (trade_value - trade_cost)
                    units = target_units
                    trades.append({
                        'date': idx,
                        'action': 'sell',
                        'units': abs(trade_units),
                        'price': exec_price,
                        'cost': trade_cost
                    })
                    costs.append(trade_cost)
            
            # Calculate NAV
            portfolio_value = cash + units * price
            nav.append(portfolio_value)
            positions.append(units)
        
        # Create NAV series
        nav_series = pd.Series(nav, index=aligned.index, name='NAV')
        
        # Calculate statistics
        total_trades = len(trades)
        total_costs = sum(costs) if costs else 0
        
        # Calculate turnover
        position_changes = pd.Series(positions).diff().abs().sum()
        avg_position = pd.Series(positions).mean()
        turnover = position_changes / (2 * avg_position) if avg_position > 0 else 0
        
        stats = {
            'total_trades': total_trades,
            'total_costs': total_costs,
            'cost_impact': total_costs / capital,
            'turnover': turnover,
            'final_nav': nav[-1] if nav else capital,
            'trades': trades
        }
        
        self.trade_log = trades
        self.daily_positions = positions
        
        return nav_series, stats
    
    def run_walk_forward_backtest(self,
                                 prices: pd.Series,
                                 walk_predictions: List[np.ndarray],
                                 walk_dates: List[np.ndarray],
                                 strategy_func) -> Dict:
        """
        Run walk-forward backtesting
        
        Args:
            prices: Full price series
            walk_predictions: List of predictions for each walk
            walk_dates: List of dates for each walk
            strategy_func: Function to convert predictions to signals
            
        Returns:
            results: Dictionary with aggregated results
        """
        all_navs = []
        all_stats = []
        
        for i, (preds, dates) in enumerate(zip(walk_predictions, walk_dates)):
            # Convert predictions to signals
            signals = strategy_func(preds)
            
            # Get prices for this walk
            walk_prices = prices[dates]
            
            # Run backtest for this walk
            nav, stats = self.simulate_nav(walk_prices, signals)
            
            all_navs.append(nav)
            all_stats.append(stats)
            
            print(f"Walk {i+1}: Final NAV = {stats['final_nav']:.0f}, "
                  f"Trades = {stats['total_trades']}")
        
        # Combine all walks
        combined_nav = pd.concat(all_navs)
        
        # Aggregate statistics
        aggregated_stats = {
            'total_trades': sum(s['total_trades'] for s in all_stats),
            'total_costs': sum(s['total_costs'] for s in all_stats),
            'avg_cost_impact': np.mean([s['cost_impact'] for s in all_stats]),
            'avg_turnover': np.mean([s['turnover'] for s in all_stats]),
            'final_nav': combined_nav.iloc[-1],
            'num_walks': len(walk_predictions)
        }
        
        return {
            'nav': combined_nav,
            'stats': aggregated_stats,
            'walk_stats': all_stats
        }


class BacktestAnalyzer:
    """
    Analyzes backtest results and generates reports
    """
    
    @staticmethod
    def analyze_trades(trades: List[Dict]) -> Dict:
        """
        Analyze trade statistics
        
        Args:
            trades: List of trade dictionaries
            
        Returns:
            analysis: Trade analysis metrics
        """
        if not trades:
            return {
                'num_trades': 0,
                'num_buys': 0,
                'num_sells': 0,
                'avg_trade_size': 0,
                'total_commission': 0
            }
        
        df_trades = pd.DataFrame(trades)
        
        buys = df_trades[df_trades['action'] == 'buy']
        sells = df_trades[df_trades['action'] == 'sell']
        
        analysis = {
            'num_trades': len(trades),
            'num_buys': len(buys),
            'num_sells': len(sells),
            'avg_trade_size': df_trades['units'].mean(),
            'total_commission': df_trades['cost'].sum(),
            'avg_buy_price': buys['price'].mean() if len(buys) > 0 else 0,
            'avg_sell_price': sells['price'].mean() if len(sells) > 0 else 0,
        }
        
        # Calculate holding periods
        if len(buys) > 0 and len(sells) > 0:
            # Simple approximation of average holding period
            buy_dates = pd.to_datetime(buys['date'])
            sell_dates = pd.to_datetime(sells['date'])
            
            if len(sell_dates) > 0:
                # Match buys with subsequent sells
                holding_periods = []
                for buy_date in buy_dates:
                    next_sell = sell_dates[sell_dates > buy_date]
                    if len(next_sell) > 0:
                        period = (next_sell.iloc[0] - buy_date).days
                        holding_periods.append(period)
                
                if holding_periods:
                    analysis['avg_holding_days'] = np.mean(holding_periods)
                else:
                    analysis['avg_holding_days'] = None
        
        return analysis
    
    @staticmethod
    def calculate_max_drawdown(nav_series: pd.Series) -> Tuple[float, int]:
        """
        Calculate maximum drawdown and duration
        
        Args:
            nav_series: NAV time series
            
        Returns:
            max_dd: Maximum drawdown as percentage
            max_duration: Maximum drawdown duration in days
        """
        # Calculate running maximum
        running_max = nav_series.expanding().max()
        
        # Calculate drawdown
        drawdown = (nav_series - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        
        # Drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        return abs(max_dd), max_duration
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict], prices: pd.Series) -> Dict:
        """
        Calculate win rate and profit factor
        
        Args:
            trades: List of trades
            prices: Price series
            
        Returns:
            metrics: Win rate metrics
        """
        if len(trades) < 2:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        # Pair up buys and sells
        profits = []
        for i in range(len(trades) - 1):
            if trades[i]['action'] == 'buy' and trades[i+1]['action'] == 'sell':
                buy_price = trades[i]['price']
                sell_price = trades[i+1]['price']
                profit = (sell_price - buy_price) / buy_price
                profits.append(profit)
        
        if not profits:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0
            }
        
        profits = np.array(profits)
        wins = profits[profits > 0]
        losses = profits[profits < 0]
        
        win_rate = len(wins) / len(profits) if len(profits) > 0 else 0
        
        total_wins = np.sum(wins) if len(wins) > 0 else 0
        total_losses = np.abs(np.sum(losses)) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': np.mean(wins) if len(wins) > 0 else 0,
            'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
            'num_wins': len(wins),
            'num_losses': len(losses)
        }


def simulate_nav_simple(prices: pd.Series, 
                        signals: pd.Series,
                        initial_capital: float = 1000000) -> pd.Series:
    """
    Simple NAV simulation without transaction costs (for baseline strategies)
    
    Args:
        prices: Price series
        signals: Trading signals
        initial_capital: Starting capital
        
    Returns:
        nav_series: NAV time series
    """
    cash = initial_capital
    units = 0
    nav = []
    
    # Shift signals to avoid look-ahead bias
    signals_shifted = signals.shift(1).fillna(0)
    
    for price, signal in zip(prices, signals_shifted):
        if signal == 1 and units == 0:  # Buy
            units = cash / price
            cash = 0
        elif signal == 0 and units > 0:  # Sell
            cash = units * price
            units = 0
        
        nav.append(cash + units * price)
    
    return pd.Series(nav, index=prices.index, name='NAV')