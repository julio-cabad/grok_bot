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

# Data Collection Configuration - 20 EPIC CRYPTO SYMBOLS
SYMBOLS: List[str] = [
    # 🏆 TOP TIER - MAJOR CRYPTOS
    'BTCUSDT',   # Bitcoin - King of Crypto
    'ETHUSDT',   # Ethereum - Smart Contract Leader
    'ADAUSDT',   # Cardano - Academic Blockchain
    'SOLUSDT',   # Solana - High Performance
    'DOTUSDT',   # Polkadot - Interoperability
    
    # 🥇 TIER 1 - ESTABLISHED ALTCOINS
    'BNBUSDT',   # Binance Coin - Exchange Token
    'XRPUSDT',   # Ripple - Cross-border Payments
    'MATICUSDT', # Polygon - Ethereum Scaling
    'AVAXUSDT',  # Avalanche - Fast Consensus
    'LINKUSDT',  # Chainlink - Oracle Network
    
    # 🥈 TIER 2 - PROMISING PROJECTS
    'ATOMUSDT',  # Cosmos - Internet of Blockchains
    'ALGOUSDT',  # Algorand - Pure Proof of Stake
    'VETUSDT',   # VeChain - Supply Chain
    'FTMUSDT',   # Fantom - Fast & Secure
    'NEARUSDT',  # NEAR Protocol - Developer Friendly
    
    # 🥉 TIER 3 - EMERGING OPPORTUNITIES
    'SANDUSDT',  # The Sandbox - Metaverse Gaming
    'MANAUSDT',  # Decentraland - Virtual Reality
    'CHZUSDT',   # Chiliz - Sports & Entertainment
    'ENJUSDT',   # Enjin Coin - Gaming NFTs
    'GALAUSDT'   # Gala - Gaming Ecosystem
]

# Comisiones de Binance (incluir en cálculos de rentabilidad)
maker_fee: float = 0.0004  # 0.04% - cuando agregas liquidez
taker_fee: float = 0.0005  # 0.05% - cuando tomas liquidez

# PnL Simulator Configuration
MAX_OPEN_POSITIONS: int = 5  # Máximo número de posiciones simultáneas
INITIAL_BALANCE: float = 1000.0  # Balance inicial para simulación
AUTO_CLOSE_ON_TARGET: bool = True  # Cerrar automáticamente en take profit/stop loss
POSITION_SIZE: float = 100.0  # Tamaño fijo de posición en USD

time_frame: str = '4h'

# Número de velas a obtener por defecto
CANDLES_LIMIT: int = 500
# Bot Configuration
CHECK_INTERVAL_SECONDS: int = 30  # Intervalo entre verificaciones en segundos (300 = 5 minutos)
ENABLE_INFINITE_LOOP: bool = True  # Habilitar bucle infinito
# System Configuration
TIMEZONE: str = "America/Guayaquil"  # Ecuador timezone (UTC-5)