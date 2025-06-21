# pricing/models/market_data.py
# Market Data Manager for Live Data Integration

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import fredapi
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False


class MarketDataManager:
    """Centralized market data management"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        
    def get_live_market_overview(self):
        """Fetch live market data for overview"""
        if not YFINANCE_AVAILABLE:
            return self._get_fallback_overview()
        
        try:
            symbols = ["^TNX", "EURUSD=X", "^VIX"]
            data = yf.download(symbols, period="2d", interval="1d", progress=False)
            
            if data.empty:
                return self._get_fallback_overview()
            
            latest = data['Close'].iloc[-1]
            previous = data['Close'].iloc[-2] if len(data) > 1 else latest
            
            return {
                'usd_10y': latest.get('^TNX', 4.0),
                'usd_10y_change': (latest.get('^TNX', 4.0) - previous.get('^TNX', 4.0)) * 100,
                'eurusd': latest.get('EURUSD=X', 1.0850),
                'eurusd_change': ((latest.get('EURUSD=X', 1.0850) / previous.get('EURUSD=X', 1.0850)) - 1) * 100,
                'vix': latest.get('^VIX', 20.0),
                'vix_change': ((latest.get('^VIX', 20.0) / previous.get('^VIX', 20.0)) - 1) * 100
            }
        except Exception as e:
            return self._get_fallback_overview()
    
    def get_live_fx_rate(self, from_currency: str, to_currency: str):
        """Get live FX rate for converting from ``from_currency`` into ``to_currency``."""
        if not YFINANCE_AVAILABLE:
            return {'rate': 1.0, 'change': 0.0, 'volatility': 15.0}

        cache_key = f"fx_{from_currency}{to_currency}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']

        try:
            symbol = f"{from_currency}{to_currency}=X"
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return {'rate': 1.0, 'change': 0.0, 'volatility': 15.0}
            
            current_rate = hist['Close'].iloc[-1]
            prev_rate = hist['Close'].iloc[-2] if len(hist) > 1 else current_rate
            change_pct = ((current_rate / prev_rate) - 1) * 100
            
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            result = {
                'rate': current_rate,
                'change': change_pct,
                'volatility': volatility
            }
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            return {'rate': 1.0, 'change': 0.0, 'volatility': 15.0}
    
    def get_live_interest_rates(self, currency):
        """Get live interest rates for a currency"""
        cache_key = f"rates_{currency}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        if not YFINANCE_AVAILABLE:
            return self._get_fallback_rates(currency)
        
        try:
            if currency == "USD":
                symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
                tenors = ["3M", "5Y", "10Y", "30Y"]
            else:
                symbols = ["^TNX"]
                tenors = ["10Y"]
            
            rates = {}
            data = yf.download(symbols, period="1d", progress=False)
            
            if not data.empty and 'Close' in data.columns:
                latest_data = data['Close'].iloc[-1]
                
                for symbol, tenor in zip(symbols, tenors):
                    if symbol in latest_data.index:
                        rates[tenor] = latest_data[symbol]
                
                # Interpolate other tenors
                if "10Y" in rates:
                    base_rate = rates["10Y"]
                    rates.update({
                        "1Y": base_rate - 0.5,
                        "2Y": base_rate - 0.3,
                        "3Y": base_rate - 0.2,
                        "7Y": base_rate - 0.1
                    })
            
            if not rates:
                rates = self._get_fallback_rates(currency)
            
            self._cache_data(cache_key, rates)
            return rates
            
        except Exception as e:
            return self._get_fallback_rates(currency)
    
    def get_live_equity_data(self, symbol):
        """Get live equity data and statistics"""
        cache_key = f"equity_{symbol}"
        if self._is_cached(cache_key):
            return self.cache[cache_key]['data']
        
        if not YFINANCE_AVAILABLE:
            return self._get_fallback_equity()
        
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="30d")
            
            if hist.empty:
                return self._get_fallback_equity()
            
            current_price = hist['Close'].iloc[-1]
            prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
            change_pct = ((current_price / prev_price) - 1) * 100
            
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Try to get dividend yield
            try:
                info = ticker.info
                dividend_yield = info.get('dividendYield', 0.02) * 100 if info.get('dividendYield') else 2.0
            except:
                dividend_yield = 2.0
            
            result = {
                'price': current_price,
                'change': change_pct,
                'volatility': volatility,
                'dividend_yield': dividend_yield
            }
            
            self._cache_data(cache_key, result)
            return result
            
        except Exception as e:
            return self._get_fallback_equity()
    
    def get_treasury_curve_data(self):
        """Get live Treasury curve data"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            treasury_symbols = ["^IRX", "^FVX", "^TNX", "^TYX"]
            treasury_data = yf.download(treasury_symbols, period="1d", progress=False)
            
            if not treasury_data.empty and 'Close' in treasury_data.columns:
                return treasury_data['Close'].iloc[-1]
            
        except Exception as e:
            pass
        
        return None
    
    def get_vix_data(self, period="90d"):
        """Get VIX volatility data"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            vix_data = yf.download("^VIX", period=period, progress=False)
            return vix_data if not vix_data.empty else None
        except Exception as e:
            return None
    
    def get_fx_correlation_data(self, pairs, period="30d"):
        """Get FX correlation data"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            fx_data = yf.download(pairs, period=period, progress=False)
            if not fx_data.empty and 'Close' in fx_data.columns:
                returns = fx_data['Close'].pct_change().fillna(0)
                return returns.corr()
        except Exception as e:
            pass
        
        return None
    
    def get_major_indices_data(self):
        """Get major market indices data"""
        if not YFINANCE_AVAILABLE:
            return None
        
        try:
            indices = ["^GSPC", "^IXIC", "^DJI", "^VIX"]
            market_data = yf.download(indices, period="1d", progress=False)
            
            if not market_data.empty and 'Close' in market_data.columns:
                return market_data['Close'].iloc[-1]
        except Exception as e:
            pass
        
        return None
    
    def _get_fallback_overview(self):
        """Fallback market overview data"""
        return {
            'usd_10y': 4.25,
            'usd_10y_change': -2.5,
            'eurusd': 1.0850,
            'eurusd_change': -0.3,
            'vix': 18.5,
            'vix_change': -1.2
        }
    
    def _get_fallback_rates(self, currency):
        """Fallback interest rates"""
        fallback_rates = {
            "USD": {"1Y": 4.5, "2Y": 4.3, "3Y": 4.2, "5Y": 4.1, "7Y": 4.0, "10Y": 4.0},
            "EUR": {"1Y": 3.2, "2Y": 3.0, "3Y": 2.9, "5Y": 2.8, "7Y": 2.7, "10Y": 2.6},
            "GBP": {"1Y": 4.8, "2Y": 4.6, "3Y": 4.4, "5Y": 4.3, "7Y": 4.2, "10Y": 4.1},
            "JPY": {"1Y": 0.1, "2Y": 0.2, "3Y": 0.3, "5Y": 0.5, "7Y": 0.7, "10Y": 0.8}
        }
        return fallback_rates.get(currency, fallback_rates["USD"])
    
    def _get_fallback_equity(self):
        """Fallback equity data"""
        return {
            'price': 150.0,
            'change': 0.5,
            'volatility': 25.0,
            'dividend_yield': 2.0
        }
    
    def _is_cached(self, key):
        """Check if data is cached and still valid"""
        if key not in self.cache:
            return False
        
        cache_time = self.cache[key]['timestamp']
        current_time = datetime.now().timestamp()
        
        return (current_time - cache_time) < self.cache_timeout
    
    def _cache_data(self, key, data):
        """Cache data with timestamp"""
        self.cache[key] = {
            'data': data,
            'timestamp': datetime.now().timestamp()
        }


# Global instance
market_data_manager = MarketDataManager()
