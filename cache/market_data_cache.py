#!/usr/bin/env python3
"""
Market Data Cache - SPARTAN EDITION
Ultra-efficient caching system for market data with intelligent TTL and compression
"""

import time
import hashlib
import logging
import pickle
import gzip
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    data: pd.DataFrame
    timestamp: float
    symbol: str
    timeframe: str
    size_bytes: int
    compressed: bool = False


class MarketDataCache:
    """
    SPARTAN MARKET DATA CACHE
    - Intelligent TTL based on timeframe
    - Compression for large datasets
    - LRU eviction policy
    - Performance metrics
    """
    
    # TTL configuration based on timeframe (in seconds)
    TIMEFRAME_TTL = {
        '1m': 30,      # 30 seconds for 1m (very fresh)
        '5m': 120,     # 2 minutes for 5m
        '15m': 300,    # 5 minutes for 15m
        '1h': 1800,    # 30 minutes for 1h (optimal)
        '4h': 7200,    # 2 hours for 4h
        '1d': 21600,   # 6 hours for 1d
    }
    
    def __init__(self, max_size_mb: int = 100, enable_compression: bool = True):
        """
        Initialize Spartan Market Data Cache
        
        Args:
            max_size_mb: Maximum cache size in MB
            enable_compression: Enable gzip compression for large datasets
        """
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert to bytes
        self.enable_compression = enable_compression
        self.logger = logging.getLogger("MarketDataCache")
        
        # Performance metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.compressions = 0
        self.total_requests = 0
        
        # Access tracking for LRU
        self.access_times: Dict[str, float] = {}
        
        self.logger.info(f"🏛️ Spartan Market Data Cache initialized - Max Size: {max_size_mb}MB")
    
    def _generate_cache_key(self, symbol: str, timeframe: str, limit: int) -> str:
        """Generate unique cache key"""
        key_data = f"{symbol}_{timeframe}_{limit}"
        return hashlib.md5(key_data.encode()).hexdigest()[:16]
    
    def _get_ttl(self, timeframe: str) -> int:
        """Get TTL based on timeframe"""
        return self.TIMEFRAME_TTL.get(timeframe, 300)  # Default 5 minutes
    
    def _compress_data(self, df: pd.DataFrame) -> bytes:
        """Compress DataFrame using gzip"""
        try:
            # Serialize DataFrame to bytes
            data_bytes = pickle.dumps(df)
            
            # Compress with gzip
            compressed_data = gzip.compress(data_bytes, compresslevel=6)
            
            compression_ratio = len(data_bytes) / len(compressed_data)
            self.logger.debug(f"Compressed data: {len(data_bytes)} -> {len(compressed_data)} bytes (ratio: {compression_ratio:.1f}x)")
            
            self.compressions += 1
            return compressed_data
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            return pickle.dumps(df)  # Fallback to uncompressed
    
    def _decompress_data(self, compressed_data: bytes) -> pd.DataFrame:
        """Decompress DataFrame from gzip"""
        try:
            # Decompress
            data_bytes = gzip.decompress(compressed_data)
            
            # Deserialize DataFrame
            df = pickle.loads(data_bytes)
            return df
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            # Try direct pickle load (uncompressed fallback)
            return pickle.loads(compressed_data)
    
    def _calculate_size(self, df: pd.DataFrame) -> int:
        """Calculate DataFrame size in bytes"""
        return df.memory_usage(deep=True).sum()
    
    def _should_compress(self, df: pd.DataFrame) -> bool:
        """Determine if DataFrame should be compressed"""
        if not self.enable_compression:
            return False
        
        # Compress if larger than 1MB
        size_bytes = self._calculate_size(df)
        return size_bytes > 1024 * 1024
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries"""
        if not self.cache:
            return
        
        # Sort by access time (oldest first)
        sorted_keys = sorted(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        # Evict oldest entries until we're under size limit
        current_size = self.get_total_size()
        
        for key in sorted_keys:
            if current_size <= self.max_size_bytes * 0.8:  # Keep 20% buffer
                break
            
            if key in self.cache:
                entry = self.cache[key]
                current_size -= entry.size_bytes
                
                del self.cache[key]
                del self.access_times[key]
                
                self.evictions += 1
                self.logger.debug(f"Evicted LRU entry: {key} ({entry.symbol} {entry.timeframe})")
    
    def get(self, symbol: str, timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        """
        Get cached market data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, 1h, etc.)
            limit: Number of candles
            
        Returns:
            Cached DataFrame or None if not found/expired
        """
        self.total_requests += 1
        
        cache_key = self._generate_cache_key(symbol, timeframe, limit)
        
        if cache_key not in self.cache:
            self.misses += 1
            self.logger.debug(f"Cache MISS: {symbol} {timeframe} (limit={limit})")
            return None
        
        entry = self.cache[cache_key]
        
        # Check if expired
        ttl = self._get_ttl(timeframe)
        if time.time() - entry.timestamp > ttl:
            # Expired - remove from cache
            del self.cache[cache_key]
            del self.access_times[cache_key]
            
            self.misses += 1
            self.logger.debug(f"Cache EXPIRED: {symbol} {timeframe} (age: {time.time() - entry.timestamp:.1f}s)")
            return None
        
        # Cache hit - update access time
        self.access_times[cache_key] = time.time()
        self.hits += 1
        
        # Decompress if needed
        if entry.compressed:
            try:
                df = self._decompress_data(entry.data)
                self.logger.debug(f"Cache HIT (compressed): {symbol} {timeframe}")
                return df
            except Exception as e:
                self.logger.error(f"Failed to decompress cached data: {e}")
                # Remove corrupted entry
                del self.cache[cache_key]
                del self.access_times[cache_key]
                return None
        else:
            self.logger.debug(f"Cache HIT: {symbol} {timeframe}")
            return entry.data.copy()  # Return copy to prevent modification
    
    def set(self, symbol: str, timeframe: str, limit: int, df: pd.DataFrame) -> None:
        """
        Cache market data
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            limit: Number of candles
            df: DataFrame to cache
        """
        if df.empty:
            self.logger.warning(f"Attempted to cache empty DataFrame for {symbol} {timeframe}")
            return
        
        cache_key = self._generate_cache_key(symbol, timeframe, limit)
        
        # Calculate size
        original_size = self._calculate_size(df)
        
        # Determine if compression is needed
        should_compress = self._should_compress(df)
        
        if should_compress:
            # Compress the data
            compressed_data = self._compress_data(df)
            data_to_store = compressed_data
            final_size = len(compressed_data)
            compressed = True
        else:
            # Store uncompressed
            data_to_store = df.copy()
            final_size = original_size
            compressed = False
        
        # Create cache entry
        entry = CacheEntry(
            data=data_to_store,
            timestamp=time.time(),
            symbol=symbol,
            timeframe=timeframe,
            size_bytes=final_size,
            compressed=compressed
        )
        
        # Check if we need to evict entries
        if self.get_total_size() + final_size > self.max_size_bytes:
            self._evict_lru()
        
        # Store in cache
        self.cache[cache_key] = entry
        self.access_times[cache_key] = time.time()
        
        compression_info = f" (compressed {original_size} -> {final_size} bytes)" if compressed else ""
        self.logger.debug(f"Cache SET: {symbol} {timeframe}{compression_info}")
    
    def get_total_size(self) -> int:
        """Get total cache size in bytes"""
        return sum(entry.size_bytes for entry in self.cache.values())
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.hits / self.total_requests) * 100
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_size_mb = self.get_total_size() / (1024 * 1024)
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'hit_rate_percent': self.get_hit_rate(),
            'cache_entries': len(self.cache),
            'total_size_mb': round(total_size_mb, 2),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'evictions': self.evictions,
            'compressions': self.compressions,
            'compression_enabled': self.enable_compression
        }
    
    def clear(self) -> None:
        """Clear all cached data"""
        entries_cleared = len(self.cache)
        self.cache.clear()
        self.access_times.clear()
        
        # Reset metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.compressions = 0
        self.total_requests = 0
        
        self.logger.info(f"🗑️ Cache cleared - {entries_cleared} entries removed")
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            ttl = self._get_ttl(entry.timeframe)
            if current_time - entry.timestamp > ttl:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            del self.cache[key]
            if key in self.access_times:
                del self.access_times[key]
        
        if expired_keys:
            self.logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    def get_cache_info(self, symbol: str = None) -> Dict[str, Any]:
        """Get detailed cache information, optionally filtered by symbol"""
        info = {
            'total_entries': len(self.cache),
            'entries_by_timeframe': {},
            'entries_by_symbol': {},
            'oldest_entry': None,
            'newest_entry': None
        }
        
        if not self.cache:
            return info
        
        # Analyze entries
        oldest_time = float('inf')
        newest_time = 0
        
        for entry in self.cache.values():
            # Filter by symbol if specified
            if symbol and entry.symbol != symbol:
                continue
            
            # Count by timeframe
            tf = entry.timeframe
            if tf not in info['entries_by_timeframe']:
                info['entries_by_timeframe'][tf] = 0
            info['entries_by_timeframe'][tf] += 1
            
            # Count by symbol
            sym = entry.symbol
            if sym not in info['entries_by_symbol']:
                info['entries_by_symbol'][sym] = 0
            info['entries_by_symbol'][sym] += 1
            
            # Track oldest/newest
            if entry.timestamp < oldest_time:
                oldest_time = entry.timestamp
                info['oldest_entry'] = {
                    'symbol': entry.symbol,
                    'timeframe': entry.timeframe,
                    'age_seconds': time.time() - entry.timestamp
                }
            
            if entry.timestamp > newest_time:
                newest_time = entry.timestamp
                info['newest_entry'] = {
                    'symbol': entry.symbol,
                    'timeframe': entry.timeframe,
                    'age_seconds': time.time() - entry.timestamp
                }
        
        return info


# Global cache instance
_market_cache = None

def get_market_cache() -> MarketDataCache:
    """Get global market data cache instance"""
    global _market_cache
    if _market_cache is None:
        _market_cache = MarketDataCache(max_size_mb=100, enable_compression=True)
    return _market_cache