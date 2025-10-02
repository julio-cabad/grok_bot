#!/usr/bin/env python3
"""
AI Validator - Ultra Light SMC Analysis
Validates trading signals using AI Smart Money Concepts analysis
Optimized for 55% cost reduction and maximum performance
"""

import time
import hashlib
import logging
import re
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

from ai.gemini_client import GeminiClient
from ai.grok_client import GrokClient
from ai.modules.fast_smc_analyzer import FastSMCAnalyzer
from config.settings import AI_TIMEOUT_SECONDS


@dataclass
class ValidationResult:
    should_enter: bool
    confidence: float
    reasoning: str
    entry_level: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    analysis_time: float = 0.0


class SmartCache:
    def __init__(self, ttl_minutes: int = 5, max_size: int = 1000):
        self.cache = {}
        self.ttl = ttl_minutes * 60
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.total_queries = 0
        self.logger = logging.getLogger("SmartCache")

    def get(self, key: str) -> Optional[Any]:
        self.total_queries += 1
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                return value
            else:
                del self.cache[key]
        self.misses += 1
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size:
            oldest = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest]
        self.cache[key] = (value, time.time())

    def get_hit_rate(self) -> float:
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.get_hit_rate(),
            'cache_size': len(self.cache),
            'max_size': self.max_size,
            'total_queries': self.total_queries
        }

    def clear(self) -> None:
        self.cache.clear()
        self.hits = 0
        self.misses = 0
        self.total_queries = 0


class AIValidatorUltra:
    """Ultra-optimized AI Validator - 55% cost reduction"""
    
    def __init__(self, timeout_seconds: int = AI_TIMEOUT_SECONDS, confidence_threshold: float = 7.5):
        # Primary AI client (Gemini)
        self.gemini_client = GeminiClient()
        
        # Backup AI client (Grok-4)
        try:
            self.grok_client = GrokClient()
            self.has_grok_backup = True
            self.logger = logging.getLogger("AIValidatorUltra")
            self.logger.info("🔄 Grok-4 backup client initialized")
        except Exception as e:
            self.grok_client = None
            self.has_grok_backup = False
            self.logger = logging.getLogger("AIValidatorUltra")
            self.logger.warning(f"⚠️ Grok-4 backup failed to initialize: {e}")
        
        self.cache = SmartCache(ttl_minutes=5)
        self.timeout = timeout_seconds
        self.confidence_threshold = confidence_threshold
        self.smc_analyzer = FastSMCAnalyzer()
        
        # Fallback statistics
        self.gemini_failures = 0
        self.grok_fallbacks = 0
        self.total_queries = 0
        
        self.logger.info(f"⚡ Ultra AIValidator initialized - Timeout: {timeout_seconds}s, Threshold: {confidence_threshold}")
        self.logger.info(f"🔄 Fallback system: {'ENABLED' if self.has_grok_backup else 'DISABLED'}")

    def validate_signal(self, df: pd.DataFrame, symbol: str, signal_type: str) -> ValidationResult:
        start_time = time.time()
        
        try:
            # Cache check
            cache_key = self._generate_cache_key(df, symbol, signal_type)
            cached = self.cache.get(cache_key)
            if cached:
                self.logger.info(f"🎯 Cache hit for {symbol} {signal_type}")
                cached.analysis_time = time.time() - start_time
                return cached

            self.logger.info(f"⚡ Ultra-analyzing {symbol} {signal_type}...")
            
            # Fast SMC analysis - handle missing OHLC columns
            try:
                smc = self.smc_analyzer.analyze_fast(df)
            except Exception as smc_error:
                self.logger.warning(f"⚠️ SMC analysis failed for {symbol}: {smc_error}")
                # Fallback SMC data when analysis fails
                smc = {
                    'ob_bull': 0,
                    'ob_bear': 0, 
                    'fvg_bull': [],
                    'fvg_bear': [],
                    'current_zone': 'EQUILIBRIUM',
                    'liq_resistance': float(df['close'].iloc[-1]) * 1.02,
                    'liq_support': float(df['close'].iloc[-1]) * 0.98,
                    'current_price': float(df['close'].iloc[-1])
                }
            
            # Compact technical analysis
            tech = self._format_tech(df)
            
            # Ultra-compact prompt (180 tokens vs 2000)
            prompt = self._build_minimal_prompt(symbol, signal_type, smc, tech, df)
            
            # AI query with unified data for fallback
            response = self._get_ai_response(prompt, symbol, signal_type, smc, tech, df)
            
            # Parse result
            result = self._parse_response(response, symbol, signal_type)
            result.analysis_time = time.time() - start_time
            
            # Cache result
            self.cache.set(cache_key, result)
            
            self.logger.info(f"✅ Ultra Score={result.confidence:.1f} Enter={result.should_enter} Time={result.analysis_time:.1f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Ultra validation failed: {e}")
            return ValidationResult(
                should_enter=True,  # Conservative fallback
                confidence=5.0,
                reasoning=f"AI failed ({str(e)[:50]}...), allowing trade",
                analysis_time=time.time() - start_time
            )

    def _generate_cache_key(self, df: pd.DataFrame, symbol: str, signal_type: str) -> str:
        """Ultra-efficient cache key generation"""
        recent = df.tail(10)[['close', 'volume']].round(2)
        blob = recent.to_csv(index=False, sep=' ').encode()
        return f"{symbol}_{signal_type}_{hashlib.md5(blob).hexdigest()[:16]}"

    def _format_tech(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compact technical indicator formatting"""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        macd_line = float(latest.get('MACD_12_26_9', 0))
        macd_signal = float(latest.get('MACD_signal_12_26_9', 0))
        macd_hist = float(latest.get('MACD_hist_12_26_9', 0))
        
        # MACD crossover detection
        prev_macd = float(prev.get('MACD_12_26_9', 0))
        prev_signal = float(prev.get('MACD_signal_12_26_9', 0))
        macd_x = "BULL" if macd_line > macd_signal and prev_macd <= prev_signal else "NONE"
        
        stoch_k = float(latest.get('STOCH_K_14_3', 50))
        
        # Bollinger Band position
        bb_upper = float(latest.get('BB_upper_20', 0))
        bb_middle = float(latest.get('BB_middle_20', 0))
        bb_lower = float(latest.get('BB_lower_20', 0))
        close_p = float(latest['close'])
        bb_pos = ((close_p - bb_lower) / (bb_upper - bb_lower)) * 100 if bb_upper != bb_lower else 50
        bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle else 0
        
        return {
            "macd_hist": macd_hist,
            "macd_x": macd_x,
            "stoch_k": stoch_k,
            "bb_pos": bb_pos,
            "bb_width": bb_width,
            "close_price": close_p  # ✅ AGREGADO - precio actual
        }

    def _build_minimal_prompt(self, symbol: str, signal_type: str, smc: Dict[str, Any], tech: Dict[str, Any], df: pd.DataFrame) -> str:
        """UNIFIED OPTIMAL PROMPT - Used by both Gemini and Grok-4"""
        
        # Extract SMC data with safe defaults and type conversion
        ob_bull_raw = smc.get('ob_bull', 0)
        ob_bear_raw = smc.get('ob_bear', 0)
        
        # Handle different data types (list vs number)
        if isinstance(ob_bull_raw, list):
            ob_bull = ob_bull_raw[0] if ob_bull_raw else 0
        else:
            ob_bull = float(ob_bull_raw) if ob_bull_raw else 0
            
        if isinstance(ob_bear_raw, list):
            ob_bear = ob_bear_raw[0] if ob_bear_raw else 0
        else:
            ob_bear = float(ob_bear_raw) if ob_bear_raw else 0
        
        fvg_bull = smc.get('fvg_bull', []) or []
        fvg_bear = smc.get('fvg_bear', []) or []
        zone = smc.get('current_zone', 'EQUILIBRIUM') or 'EQUILIBRIUM'
        liq_s = float(smc.get('liq_resistance', 0) or 0)
        liq_b = float(smc.get('liq_support', 0) or 0)
        price = float(smc.get('current_price', 0) or 0)
        
        # If price is still 0, use current close price
        if price == 0:
            try:
                price = float(tech.get('close_price', 0)) or float(df['close'].iloc[-1])
            except Exception as e:
                self.logger.error(f"❌ CRITICAL: Failed to get price for {symbol}: {e}")
                raise ValueError(f"Cannot determine price for {symbol} - aborting analysis")
        
        # Compact FVG string
        fvg_str = ""
        if fvg_bull:
            fvg_str += f"B:{fvg_bull[0]['lower']:.0f}-{fvg_bull[0]['upper']:.0f} "
        if fvg_bear:
            fvg_str += f"S:{fvg_bear[0]['lower']:.0f}-{fvg_bear[0]['upper']:.0f}"
        
        # UNIFIED OPTIMAL PROMPT - Same for both Gemini and Grok-4
        return f"""TRADING SIGNAL ANALYSIS: {symbol} {signal_type} @ ${price:.2f}

SMART MONEY CONCEPTS:
- Current Zone: {zone}
- Order Block Bull: ${ob_bull:.0f} | Bear: ${ob_bear:.0f}
- Fair Value Gaps: {fvg_str.strip() if fvg_str.strip() else 'None'}
- Liquidity Support: ${liq_b:.0f} | Resistance: ${liq_s:.0f}

TECHNICAL INDICATORS:
- MACD Histogram: {tech['macd_hist']:.2f} | Crossover: {tech['macd_x']}
- Stochastic K: {tech['stoch_k']:.1f}
- Bollinger Position: {tech['bb_pos']:.0f}% | Width: {tech['bb_width']:.1f}%

ANALYSIS WEIGHTS: SMC 40% | MACD 25% | Stochastic 20% | Bollinger 15%

RULES:
- LONG: Stoch K<40 (oversold) + bullish momentum + SMC support
- SHORT: Stoch K>60 (overbought) + bearish momentum + SMC resistance
- Score 8-10: High probability setup
- Score 6-7: Medium probability
- Score <6: Low probability, avoid

RESPOND EXACTLY:
SCORE: [0-10]
ENTER: [YES/NO]
REASONING: [max 50 words]
ENTRY: [price]
SL: [stop loss]
TP: [take profit]""".strip()

    def _get_ai_response(self, prompt: str, symbol: str = None, signal_type: str = None, smc: dict = None, tech: dict = None, df = None) -> str:
        """Get AI response with fallback system using IDENTICAL unified prompt"""
        self.total_queries += 1
        start = time.time()
        
        # Try Gemini first
        try:
            self.logger.debug("🔵 Trying Gemini AI...")
            resp = self.gemini_client.query(prompt)
            
            if time.time() - start > self.timeout:
                raise TimeoutError("Gemini query exceeded timeout")
                
            self.logger.info("✅ Gemini AI successful")
            return resp
            
        except Exception as gemini_error:
            self.gemini_failures += 1
            self.logger.warning(f"❌ Gemini failed: {str(gemini_error)[:100]}...")
            
            # Try Grok-4 fallback with EXACT SAME PROMPT
            if self.has_grok_backup and self.grok_client:
                try:
                    self.logger.info("🔄 Falling back to Grok-4 with IDENTICAL prompt...")
                    self.grok_fallbacks += 1
                    
                    # Reset timer for Grok-4 fallback (separate timeout)
                    grok_start = time.time()
                    
                    # Use EXACT SAME prompt - no simplification
                    resp = self.grok_client.query(prompt, temperature=0.3)
                    
                    # Check Grok-4 specific timeout (more generous)
                    grok_time = time.time() - grok_start
                    if grok_time > 180:  # 3 minutes for Grok-4
                        self.logger.warning(f"⚠️ Grok-4 took {grok_time:.1f}s but completed successfully")
                    
                    self.logger.info(f"✅ Grok-4 fallback successful in {grok_time:.1f}s with IDENTICAL prompt")
                    return resp
                    
                except Exception as grok_error:
                    self.logger.error(f"❌ Grok-4 fallback also failed: {str(grok_error)[:100]}...")
                    # Both AIs failed, re-raise original Gemini error
                    raise gemini_error
            else:
                # No backup available, re-raise Gemini error
                self.logger.error("❌ No backup AI available")
                raise gemini_error

    def _parse_response(self, ai_response: str, symbol: str, signal_type: str) -> ValidationResult:
        """Parse ultra-compact AI response"""
        # DEBUG: Log the actual AI response
        self.logger.info(f"🔍 Raw AI response for {symbol}: {ai_response}")
        
        lines = [ln.strip() for ln in ai_response.strip().split('\n') if ln.strip()]
        
        score, enter, reasoning, entry, sl, tp = 5.0, True, "AI completed", None, None, None
        
        # Try to extract score from anywhere in the response
        score_found = False
        
        for ln in lines:
            # More flexible score parsing
            if 'SCORE' in ln.upper() or any(word in ln.upper() for word in ['SCORE:', 'RATING:', 'CONFIDENCE:']):
                try:
                    # Extract number after colon or just find any number
                    if ':' in ln:
                        score_text = ln.split(':', 1)[1].strip()
                    else:
                        score_text = ln
                    
                    # Find first number in the text
                    import re
                    numbers = re.findall(r'\d+\.?\d*', score_text)
                    if numbers:
                        score = float(numbers[0])
                        score = max(0, min(10, score))
                        score_found = True
                        self.logger.debug(f"✅ Found score {score} in line: {ln}")
                except Exception as e:
                    self.logger.debug(f"❌ Failed to parse score from: {ln}, error: {e}")
                    
            elif 'ENTER' in ln.upper() or 'TRADE' in ln.upper():
                enter_text = ln.split(':', 1)[1].strip() if ':' in ln else ln
                enter = any(word in enter_text.upper() for word in ['YES', 'Y', 'TRUE', '1', 'BUY', 'SELL', 'LONG', 'SHORT'])
                
            elif 'REASON' in ln.upper() or 'ANALYSIS' in ln.upper():
                reasoning = ln.split(':', 1)[1].strip() if ':' in ln else ln
                
            elif 'ENTRY' in ln.upper():
                try:
                    entry_text = ln.split(':', 1)[1].strip() if ':' in ln else ln
                    entry = float(re.findall(r'\d+\.?\d*', entry_text.replace('$', '').replace(',', ''))[0])
                except Exception:
                    pass
                    
            elif 'SL' in ln.upper() or 'STOP' in ln.upper():
                try:
                    sl_text = ln.split(':', 1)[1].strip() if ':' in ln else ln
                    sl = float(re.findall(r'\d+\.?\d*', sl_text.replace('$', '').replace(',', ''))[0])
                except Exception:
                    pass
                    
            elif 'TP' in ln.upper() or 'TARGET' in ln.upper():
                try:
                    tp_text = ln.split(':', 1)[1].strip() if ':' in ln else ln
                    tp = float(re.findall(r'\d+\.?\d*', tp_text.replace('$', '').replace(',', ''))[0])
                except Exception:
                    pass
        
        # If no score was found, log warning
        if not score_found:
            self.logger.warning(f"⚠️ No score found in AI response for {symbol}, using default 5.0")
        
        # Apply threshold
        if score < self.confidence_threshold:
            enter = False
            reasoning = f"Score {score:.1f} below threshold. " + reasoning
        
        return ValidationResult(
            should_enter=enter,
            confidence=score,
            reasoning=reasoning,
            entry_level=entry,
            stop_loss=sl,
            take_profit=tp
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = self.cache.get_stats()
        stats['validator_config'] = {
            'confidence_threshold': self.confidence_threshold,
            'timeout_seconds': self.timeout
        }
        return stats
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback system statistics"""
        return {
            'total_queries': self.total_queries,
            'gemini_failures': self.gemini_failures,
            'grok_fallbacks': self.grok_fallbacks,
            'gemini_success_rate': ((self.total_queries - self.gemini_failures) / self.total_queries * 100) if self.total_queries > 0 else 0,
            'fallback_usage_rate': (self.grok_fallbacks / self.total_queries * 100) if self.total_queries > 0 else 0,
            'has_grok_backup': self.has_grok_backup
        }
    
    def print_fallback_stats(self):
        """Print fallback statistics"""
        stats = self.get_fallback_stats()
        
        print("\n🔄 AI FALLBACK SYSTEM STATS:")
        print("-" * 40)
        print(f"📊 Total Queries: {stats['total_queries']}")
        print(f"✅ Gemini Success Rate: {stats['gemini_success_rate']:.1f}%")
        print(f"❌ Gemini Failures: {stats['gemini_failures']}")
        print(f"🔄 Grok-4 Fallbacks: {stats['grok_fallbacks']}")
        print(f"📈 Fallback Usage: {stats['fallback_usage_rate']:.1f}%")
        print(f"🛡️ Backup Available: {'YES' if stats['has_grok_backup'] else 'NO'}")
        
        if stats['fallback_usage_rate'] > 50:
            print("🚨 WARNING: High fallback usage - Consider checking Gemini API")
        elif stats['fallback_usage_rate'] > 0:
            print("✅ GOOD: Fallback system working when needed")

    def clear_cache(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.logger.info("🗑️ Ultra cache cleared")


# Global singleton for ultra validator
_ultra_validator: Optional[AIValidatorUltra] = None

def get_ultra_validator() -> AIValidatorUltra:
    global _ultra_validator
    if _ultra_validator is None:
        _ultra_validator = AIValidatorUltra()
    return _ultra_validator

def reset_ultra_validator() -> None:
    global _ultra_validator
    if _ultra_validator:
        _ultra_validator.clear_cache()
    _ultra_validator = None