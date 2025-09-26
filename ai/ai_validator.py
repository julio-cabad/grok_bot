#!/usr/bin/env python3
"""
AI Validator - Ultra Simple SMC Analysis
Validates trading signals using AI Smart Money Concepts analysis
"""

import time
import hashlib
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from decimal import Decimal
import pandas as pd

# Import existing AI client
from ai.kimi_client import GeminiClient


@dataclass
class ValidationResult:
    """Result of AI validation"""
    should_enter: bool
    confidence: float  # 0-10 score
    reasoning: str
    entry_level: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    analysis_time: float = 0.0


class SmartCache:
    """Ultra efficient memory-based cache with TTL"""
    
    def __init__(self, ttl_minutes: int = 5, max_size: int = 1000):
        self.cache = {}
        self.ttl = ttl_minutes * 60
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.logger = logging.getLogger("SmartCache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                self.logger.debug(f"Cache HIT for key: {key[:20]}...")
                return value
            else:
                del self.cache[key]
                self.logger.debug(f"Cache EXPIRED for key: {key[:20]}...")
        
        self.misses += 1
        self.logger.debug(f"Cache MISS for key: {key[:20]}...")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value with timestamp"""
        # Clean old entries if cache is full
        if len(self.cache) >= self.max_size:
            self._cleanup_expired()
        
        self.cache[key] = (value, time.time())
        self.logger.debug(f"Cache SET for key: {key[:20]}...")
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate percentage"""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'cache_size': len(self.cache),
            'max_size': self.max_size
        }
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self.cache.items()
            if current_time - timestamp >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
        
        if expired_keys:
            self.logger.info(f"Cleaned {len(expired_keys)} expired cache entries")


class AIValidator:
    """Ultra simple IA validator with smart caching"""
    
    def __init__(self, timeout_seconds: int = 30, confidence_threshold: float = 7.5):
        self.gemini_client = GeminiClient()
        self.cache = SmartCache(ttl_minutes=5)
        self.timeout = timeout_seconds
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger("AIValidator")
        
        self.logger.info(f"🤖 AIValidator initialized - Timeout: {timeout_seconds}s, Threshold: {confidence_threshold}")
    
    def validate_signal(self, df: pd.DataFrame, symbol: str, signal_type: str) -> ValidationResult:
        """
        Validate trading signal using AI SMC analysis
        
        Args:
            df: DataFrame with OHLCV data (500 candles)
            symbol: Trading symbol (e.g., 'BTCUSDT')
            signal_type: 'LONG' or 'SHORT'
            
        Returns:
            ValidationResult with should_enter, confidence, reasoning
        """
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(df, symbol, signal_type)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self.logger.info(f"🎯 Using cached AI analysis for {symbol} {signal_type}")
                return cached_result
            
            self.logger.info(f"🤖 Starting AI SMC analysis for {symbol} {signal_type}...")
            
            # Build SMC analysis prompt
            prompt = self._build_smc_prompt(df, symbol, signal_type)
            
            # Get AI analysis with timeout protection
            ai_response = self._get_ai_analysis_with_timeout(prompt)
            
            # Parse and validate response
            result = self._parse_ai_response(ai_response, symbol, signal_type)
            result.analysis_time = time.time() - start_time
            
            # Cache result
            self.cache.set(cache_key, result)
            
            self.logger.info(
                f"✅ AI Analysis complete for {symbol}: "
                f"Enter={result.should_enter}, Score={result.confidence:.1f}, "
                f"Time={result.analysis_time:.1f}s"
            )
            
            return result
            
        except Exception as e:
            # Fail gracefully - allow trade if AI fails
            analysis_time = time.time() - start_time
            self.logger.error(f"❌ AI validation failed for {symbol}: {e}")
            
            return ValidationResult(
                should_enter=True,  # Default to allowing trade
                confidence=5.0,     # Neutral confidence
                reasoning=f"AI validation failed ({str(e)[:50]}...), allowing trade based on technical analysis",
                analysis_time=analysis_time
            )
    
    def _generate_cache_key(self, df: pd.DataFrame, symbol: str, signal_type: str) -> str:
        """Generate unique cache key based on recent data"""
        # Use last 10 candles for cache key (recent market state)
        recent_data = df.tail(10)[['open', 'high', 'low', 'close', 'volume']].values.tobytes()
        data_hash = hashlib.md5(recent_data).hexdigest()[:16]
        return f"{symbol}_{signal_type}_{data_hash}"
    
    def _build_smc_prompt(self, df: pd.DataFrame, symbol: str, signal_type: str) -> str:
        """Build comprehensive SMC analysis prompt"""
        
        # Get current price and recent data
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        
        # Calculate basic statistics
        price_change_pct = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Get last 50 candles for analysis
        analysis_data = df.tail(50)
        
        prompt = f"""
SMART MONEY CONCEPTS ANALYSIS - {symbol}

SIGNAL TO VALIDATE: {signal_type}
CURRENT PRICE: ${current_price:.4f}
PRICE CHANGE: {price_change_pct:+.2f}%
VOLUME RATIO: {volume_ratio:.2f}x average

RECENT MARKET DATA (Last 50 candles):
High: ${recent_high:.4f} | Low: ${recent_low:.4f}
Current Volume: {current_volume:,.0f} | Avg Volume: {avg_volume:,.0f}

OHLCV DATA FOR ANALYSIS:
{analysis_data[['open', 'high', 'low', 'close', 'volume']].to_string()}

SMART MONEY ANALYSIS REQUIRED:

1. ORDER BLOCKS: Identify institutional supply/demand zones
2. FAIR VALUE GAPS: Detect price imbalances that need to be filled
3. LIQUIDITY SWEEPS: Check for stop hunt patterns above/below key levels
4. MARKET STRUCTURE: Analyze break of structure (BOS) or change of character (CHoCH)
5. SUPPORT/RESISTANCE: Identify key institutional levels
6. VOLUME PROFILE: Analyze volume at key price levels

CONTEXT FOR {signal_type} SIGNAL:
- Technical indicators already suggest {signal_type} opportunity
- Need SMC confirmation for institutional alignment
- Looking for confluence between technical and smart money concepts

PROVIDE ANALYSIS IN THIS EXACT FORMAT:

SCORE: [0-10 confidence score]
ENTER: [YES/NO - should we take this trade?]
REASONING: [Brief explanation of key SMC factors - max 100 words]
ENTRY: [Optimal entry price if applicable]
SL: [Suggested stop loss level]
TP: [Suggested take profit level]

ANALYSIS CRITERIA:
- Score 8-10: Strong SMC confluence, high probability setup
- Score 6-7: Moderate SMC support, acceptable risk
- Score 4-5: Neutral/mixed signals, proceed with caution
- Score 0-3: SMC against the trade, avoid entry

Focus on INSTITUTIONAL behavior patterns and SMART MONEY footprints.
"""
        
        return prompt
    
    def _get_ai_analysis_with_timeout(self, prompt: str) -> str:
        """Get AI analysis with timeout protection"""
        try:
            # Simple timeout implementation
            start_time = time.time()
            response = self.gemini_client.query(prompt)
            
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise TimeoutError(f"AI analysis took {elapsed:.1f}s, exceeded {self.timeout}s timeout")
            
            return response
            
        except Exception as e:
            raise Exception(f"AI analysis failed: {str(e)}")
    
    def _parse_ai_response(self, ai_response: str, symbol: str, signal_type: str) -> ValidationResult:
        """Parse AI response into ValidationResult"""
        try:
            lines = ai_response.strip().split('\n')
            
            print("====>",lines)
            
            # Initialize defaults
            score = 5.0
            should_enter = True
            reasoning = "AI analysis completed"
            entry_level = None
            stop_loss = None
            take_profit = None
            
            # Parse each line
            for line in lines:
                line = line.strip()
                if line.startswith('SCORE:'):
                    score_str = line.replace('SCORE:', '').strip()
                    try:
                        score = float(score_str)
                        score = max(0, min(10, score))  # Clamp to 0-10
                    except ValueError:
                        self.logger.warning(f"Could not parse score: {score_str}")
                
                elif line.startswith('ENTER:'):
                    enter_str = line.replace('ENTER:', '').strip().upper()
                    should_enter = enter_str in ['YES', 'Y', 'TRUE', '1']
                
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
                
                elif line.startswith('ENTRY:'):
                    entry_str = line.replace('ENTRY:', '').strip()
                    try:
                        entry_level = float(entry_str.replace('$', '').replace(',', ''))
                    except (ValueError, AttributeError):
                        pass
                
                elif line.startswith('SL:'):
                    sl_str = line.replace('SL:', '').strip()
                    try:
                        stop_loss = float(sl_str.replace('$', '').replace(',', ''))
                    except (ValueError, AttributeError):
                        pass
                
                elif line.startswith('TP:'):
                    tp_str = line.replace('TP:', '').strip()
                    try:
                        take_profit = float(tp_str.replace('$', '').replace(',', ''))
                    except (ValueError, AttributeError):
                        pass
            
            # Apply confidence threshold
            if score < self.confidence_threshold:
                should_enter = False
                reasoning = f"Score {score:.1f} below threshold {self.confidence_threshold}. " + reasoning
            
            return ValidationResult(
                should_enter=should_enter,
                confidence=score,
                reasoning=reasoning,
                entry_level=entry_level,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse AI response: {e}")
            self.logger.debug(f"AI Response was: {ai_response}")
            
            # Return neutral result on parsing failure
            # Apply threshold check even on parsing failure
            neutral_score = 5.0
            should_enter_on_failure = neutral_score >= self.confidence_threshold
            
            return ValidationResult(
                should_enter=should_enter_on_failure,
                confidence=neutral_score,
                reasoning=f"AI response parsing failed: {str(e)[:50]}..., " + 
                         ("allowing trade" if should_enter_on_failure else "rejecting trade due to threshold")
            )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear all cached results"""
        self.cache.cache.clear()
        self.cache.hits = 0
        self.cache.misses = 0
        self.logger.info("🗑️ AI validation cache cleared")


# Global instance for easy import
ai_validator = None

def get_ai_validator() -> AIValidator:
    """Get global AI validator instance"""
    global ai_validator
    if ai_validator is None:
        ai_validator = AIValidator()
    return ai_validator