"""
Configuration settings for the multicrypto trading system
"""
import os
from dotenv import load_dotenv
from typing import List, Dict

# Load environment variables from .env file
load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
BINANCE_TESTNET = os.getenv('BINANCE_TESTNET', 'False').lower() == 'true'

# Data Collection Configuration - TOP 20 CRYPTO SYMBOLS (Clean & Optimized)
SYMBOLS: List[str] = [
    # 🏆 TOP TIER - MAJOR CRYPTOS (High Volume & Liquidity)
    'BTCUSDT',   # Bitcoin - King of Crypto
    'ETHUSDT',   # Ethereum - Smart Contract Leader
    'SOLUSDT',   # Solana - High Performance
    'XRPUSDT',   # Ripple - Cross-border Payments
    'BNBUSDT',   # Binance Coin - Exchange Token
    
    # 🥇 TIER 1 - ESTABLISHED ALTCOINS (Proven Track Record)
    'ADAUSDT',   # Cardano - Academic Blockchain
    'DOTUSDT',   # Polkadot - Interoperability
    'ENAUSDT', # Polygon - Ethereum Scaling
    'AVAXUSDT',  # Avalanche - Fast Consensus
    'LINKUSDT',  # Chainlink - Oracle Network
    
    # 🥈 TIER 2 - HIGH MOMENTUM COINS (Active Trading)
    'ATOMUSDT',  # Cosmos - Internet of Blockchains
    'NEARUSDT',  # NEAR Protocol - Developer Friendly
    'DOGEUSDT',  # Doge - Meme King
    'TONUSDT',   # TON - Telegram
    'SUIUSDT',   # Sui - New Layer 1
    'TIAUSDT',   # Celestia - Modular Blockchain
    'HBARUSDT',  # Hedera - Enterprise DLT
    'TRXUSDT',   # Tron - Content Entertainment
    
    # 🥉 TIER 3 - EMERGING OPPORTUNITIES
    'CHZUSDT',   # Chiliz - Sports & Entertainment
    'ENJUSDT',   # Enjin Coin - Gaming NFTs
    'XLMUSDT',   # Enjin Coin - Gaming NFTs
]

# Comisiones de Binance (incluir en cálculos de rentabilidad)
maker_fee: float = 0.0004  # 0.04% - cuando agregas liquidez
taker_fee: float = 0.0005  # 0.05% - cuando tomas liquidez

# PnL Simulator Configuration
MAX_OPEN_POSITIONS: int = 5  # Máximo número de posiciones simultáneas
INITIAL_BALANCE: float = 1000.0  # Balance inicial para simulación
AUTO_CLOSE_ON_TARGET: bool = True  # Cerrar automáticamente en take profit/stop loss
POSITION_SIZE: float = 100.0  # Tamaño fijo de posición en USD

# Timeframe Configuration - OPTIMIZED FOR HIGHER TIMEFRAMES
USE_HIGHER_TIMEFRAMES: bool = os.getenv('USE_HIGHER_TIMEFRAMES', 'true').lower() == 'true'  # Feature flag para timeframes superiores
time_frame: str = '1h' if USE_HIGHER_TIMEFRAMES else '1m'  # Default to 1H for better signal quality and less noise
TIMEZONE: str = "America/Guayaquil"  # Ecuador timezone (UTC-5)

# Higher Timeframes Feature Flag
USE_HIGHER_TIMEFRAMES: bool = True  # Enable optimized higher timeframe trading

# Dynamic configuration based on timeframe - OPTIMIZED CANDLES
TIMEFRAME_CANDLES = {
    '1m': 500,    # 500 minutes = ~8 hours
    '5m': 288,    # 288 * 5min = 24 hours  
    '15m': 96,    # 96 * 15min = 24 hours
    '1h': 168,    # 168 hours = 7 days (optimal for 1H analysis)
    '2h': 168,    # 168 hours = 7 days (optimal for 1H analysis)
    '4h': 180,    # 180 * 4h = 30 days
    '1d': 90,     # 90 days = 3 months
}

def get_candles_limit() -> int:
    """Get optimized candles limit based on current timeframe"""
    return TIMEFRAME_CANDLES.get(time_frame, 500)

CANDLES_LIMIT: int = get_candles_limit()  # Optimized for current timeframe

# Optimized check intervals for different timeframes
TIMEFRAME_INTERVALS = {
    '1m': 5,      # 5 seconds for 1 minute
    '5m': 30,     # 30 seconds for 5 minutes  
    '15m': 60,    # 1 minute for 15 minutes
    '1h': 150,    # 2.5 minutes for 1 hour
    '2h': 150,    # 5 minutes for 1 hour (optimal)
    '4h': 300,    # 15 minutes for 4 hours
    '1d': 3600,   # 1 hour for 1 day
}

# Get optimized interval for current timeframe
def get_check_interval() -> int:
    """Get optimized check interval based on current timeframe"""
    return TIMEFRAME_INTERVALS.get(time_frame, 150)

# Bot Configuration - OPTIMIZED FOR HIGHER TIMEFRAMES
CHECK_INTERVAL_SECONDS: int = get_check_interval()  # Dynamic interval based on timeframe
ENABLE_INFINITE_LOOP: bool = True  # Habilitar bucle infinito

# AI Validation Configuration
USE_AI_VALIDATION: bool = True  # Feature flag para activar validación IA (Disabled due to quota)
AI_CONFIDENCE_THRESHOLD: float = 7.5  # Umbral base de confianza IA
AI_TIMEOUT_SECONDS: int = 150  # Timeout máximo para análisis IA (increased for Grok-4 fallback)

# Adaptive Threshold Configuration
USE_ADAPTIVE_THRESHOLD: bool = True  # Activar threshold adaptativo
BULL_MARKET_THRESHOLD: float = 6.5   # Más agresivo en mercados alcistas
BEAR_MARKET_THRESHOLD: float = 8.0   # Más conservador en mercados bajistas
HIGH_VOLATILITY_THRESHOLD: float = 8.5  # Extra conservador en alta volatilidad