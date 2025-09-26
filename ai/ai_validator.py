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
from ai.gemini_client import GeminiClient
# Import fast SMC analyzer
from ai.modules.fast_smc_analyzer import FastSMCAnalyzer


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
    """Ultra simple IA validator with smart caching and fast SMC analysis"""
    
    def __init__(self, timeout_seconds: int = 30, confidence_threshold: float = 7.5):
        self.gemini_client = GeminiClient()
        self.cache = SmartCache(ttl_minutes=5)
        self.timeout = timeout_seconds
        self.confidence_threshold = confidence_threshold
        self.smc_analyzer = FastSMCAnalyzer()  # Initialize fast SMC analyzer
        self.logger = logging.getLogger("AIValidator")
        
        self.logger.info(f"🤖 AIValidator initialized - Timeout: {timeout_seconds}s, Threshold: {confidence_threshold}, Fast SMC: ✅")
    
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
        """Build comprehensive SMC analysis prompt with real SMC data"""
        
        # Get current price and recent data
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        avg_volume = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        
        # Calculate basic statistics
        price_change_pct = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Get REAL SMC analysis using fast analyzer
        smc_data = self.smc_analyzer.analyze_fast(df)
        
        # Format SMC analysis
        smc_analysis = self._format_smc_data(smc_data)
        
        # Get last 50 candles for analysis
        analysis_data = df.tail(50)
        
        # Get technical indicators for confluence analysis
        technical_indicators = self._format_technical_indicators(df)
        
        prompt = f"""
🏛️ SPARTAN MILLIONAIRE TRADING SYSTEM 🏛️
Elite Smart Money Concepts + Multi-Indicator Confluence Analysis

📊 SYMBOL: {symbol} | SIGNAL: {signal_type}
💰 CURRENT PRICE: ${current_price:.4f} | CHANGE: {price_change_pct:+.2f}%
📈 VOLUME: {volume_ratio:.2f}x average | HIGH: ${recent_high:.4f} | LOW: ${recent_low:.4f}

🏛️ REAL SMC ANALYSIS (INSTITUTIONAL DATA):
{smc_analysis}

🤖 TECHNICAL INDICATORS CONFLUENCE:
{technical_indicators}

📊 OHLCV DATA (Last 50 candles):
{analysis_data[['open', 'high', 'low', 'close', 'volume']].to_string()}

🏛️ INSTITUTIONAL ANALYSIS FRAMEWORK (WEIGHTED SCORING):

1. SMC STRUCTURE ANALYSIS (35% WEIGHT):
   ✅ Order Blocks: Institutional supply/demand zones
   ✅ Fair Value Gaps: Price imbalances requiring fills  
   ✅ Liquidity Sweeps: Stop hunts above/below key levels
   ✅ Market Structure: BOS/CHoCH patterns
   ✅ Support/Resistance: Key institutional levels

2. MOMENTUM ANALYSIS (25% WEIGHT):
   ✅ MACD: Histogram strength, signal crossovers, divergences
   ✅ Trend alignment with price action
   ✅ Momentum sustainability assessment

3. MEAN REVERSION ANALYSIS (20% WEIGHT):
   ✅ Stochastic: Overbought/oversold conditions
   ✅ %K/%D positioning and crossovers  
   ✅ Reversal probability assessment

4. VOLATILITY ANALYSIS (20% WEIGHT):
   ✅ Bollinger Bands: Price position relative to bands
   ✅ Squeeze/expansion patterns
   ✅ Volatility breakout potential

🎯 CONFLUENCE REQUIREMENTS FOR {signal_type}:
- SMC structure MUST align with technical direction
- Minimum 3/4 indicators showing confluence
- Risk/reward ratio minimum 1:2
- Clear institutional footprints present

📋 PROVIDE ANALYSIS IN EXACT FORMAT:

SCORE: [0-10 weighted confluence score]
ENTER: [YES/NO based on confluence threshold]
REASONING: [Key confluence factors focusing on institutional behavior - max 150 words]
ENTRY: [Optimal entry price based on confluence zones]
SL: [Stop loss at key institutional level]
TP: [Take profit at next major liquidity zone]

🏆 MILLIONAIRE SCORING CRITERIA:
- Score 9-10: PERFECT confluence, all 4 indicators aligned, institutional setup confirmed
- Score 8-8.9: EXCELLENT confluence, 3-4 indicators aligned, strong probability setup
- Score 7-7.9: GOOD confluence, 2-3 indicators aligned, acceptable risk setup
- Score 6-6.9: MODERATE confluence, mixed signals, proceed with extreme caution
- Score 0-5.9: POOR confluence, conflicting signals, AVOID TRADE

🎯 MISSION: Identify MILLIONAIRE-MAKING setups with institutional precision!
Analyze CONFLUENCE between SMC + MACD + STOCHASTIC + BOLLINGER for maximum edge.
Focus on INSTITUTIONAL behavior patterns and SMART MONEY footprints.
"""
        
        return prompt
    
    def _format_technical_indicators(self, df: pd.DataFrame) -> str:
        """Format technical indicators for AI analysis"""
        try:
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # MACD Analysis (using correct column names)
            macd_line = latest.get('MACD_12_26_9', 0)
            macd_signal = latest.get('MACD_signal_12_26_9', 0)  # Fixed: was MACDs_12_26_9
            macd_histogram = latest.get('MACD_hist_12_26_9', 0)  # Fixed: was MACDh_12_26_9
            macd_trend = "BULLISH" if macd_histogram > 0 else "BEARISH"
            macd_strength = abs(macd_histogram)
            
            # MACD Crossover detection
            prev_macd = prev.get('MACD_12_26_9', 0)
            prev_signal = prev.get('MACD_signal_12_26_9', 0)  # Fixed: was MACDs_12_26_9
            macd_crossover = ""
            if macd_line > macd_signal and prev_macd <= prev_signal:
                macd_crossover = "BULLISH CROSSOVER"
            elif macd_line < macd_signal and prev_macd >= prev_signal:
                macd_crossover = "BEARISH CROSSOVER"
            else:
                macd_crossover = "NO CROSSOVER"
            
            # Stochastic Analysis (using correct column names)
            stoch_k = latest.get('STOCH_K_14_3', 50)
            stoch_d = latest.get('STOCH_D_14_3', 50)
            stoch_condition = "OVERBOUGHT" if stoch_k > 80 else "OVERSOLD" if stoch_k < 20 else "NEUTRAL"
            stoch_crossover = "K>D (BULLISH)" if stoch_k > stoch_d else "K<D (BEARISH)"
            
            # Bollinger Bands Analysis (using correct column names)
            bb_upper = latest.get('BB_upper_20', latest.get('close', 0))
            bb_middle = latest.get('BB_middle_20', latest.get('close', 0))  
            bb_lower = latest.get('BB_lower_20', latest.get('close', 0))
            close_price = latest.get('close', 0)
            
            if close_price > bb_upper:
                bb_position = "ABOVE UPPER BAND (OVERBOUGHT)"
            elif close_price < bb_lower:
                bb_position = "BELOW LOWER BAND (OVERSOLD)"
            elif close_price > bb_middle:
                bb_position = "UPPER HALF (BULLISH BIAS)"
            else:
                bb_position = "LOWER HALF (BEARISH BIAS)"
                
            bb_width = ((bb_upper - bb_lower) / bb_middle) * 100 if bb_middle > 0 else 0
            bb_squeeze = "SQUEEZE (LOW VOLATILITY)" if bb_width < 2.0 else "EXPANSION (HIGH VOLATILITY)"
            
            # Calculate Bollinger %B (position within bands)
            bb_percent = ((close_price - bb_lower) / (bb_upper - bb_lower)) * 100 if (bb_upper - bb_lower) > 0 else 50
            
            indicators_text = f"""
📊 MOMENTUM INDICATORS (25% Weight):
   • MACD Line: {macd_line:.4f} | Signal: {macd_signal:.4f} | Histogram: {macd_histogram:.4f}
   • Trend: {macd_trend} | Strength: {macd_strength:.4f} | Status: {macd_crossover}
   
📊 MEAN REVERSION INDICATORS (20% Weight):
   • Stochastic %K: {stoch_k:.1f}% | %D: {stoch_d:.1f}%
   • Condition: {stoch_condition} | Signal: {stoch_crossover}
   
📊 VOLATILITY INDICATORS (20% Weight):
   • Bollinger Position: {bb_position}
   • BB %B: {bb_percent:.1f}% | Width: {bb_width:.2f}% | State: {bb_squeeze}
   • Upper: ${bb_upper:.4f} | Middle: ${bb_middle:.4f} | Lower: ${bb_lower:.4f}
"""
            
            return indicators_text
            
        except Exception as e:
            return f"Technical indicators formatting error: {str(e)}"
    
    def _format_smc_data(self, smc_data: Dict[str, Any]) -> str:
        """Format real SMC analysis data for AI prompt"""
        try:
            # Order Blocks
            ob_bull = smc_data.get('ob_bull')
            ob_bear = smc_data.get('ob_bear')
            ob_text = "ORDER BLOCKS (INSTITUTIONAL ZONES):\n"
            if ob_bull:
                ob_text += f"   • BULLISH OB: ${ob_bull:.4f} (Support Zone)\n"
            if ob_bear:
                ob_text += f"   • BEARISH OB: ${ob_bear:.4f} (Resistance Zone)\n"
            if not ob_bull and not ob_bear:
                ob_text += "   • No significant order blocks detected\n"
            
            # Fair Value Gaps
            fvg_bull = smc_data.get('fvg_bull', [])
            fvg_bear = smc_data.get('fvg_bear', [])
            fvg_text = "\nFAIR VALUE GAPS (PRICE IMBALANCES):\n"
            if fvg_bull:
                for fvg in fvg_bull[:2]:  # Show top 2
                    fvg_text += f"   • BULLISH FVG: ${fvg['lower']:.4f} - ${fvg['upper']:.4f} (Strength: {fvg['strength']})\n"
            if fvg_bear:
                for fvg in fvg_bear[:2]:  # Show top 2
                    fvg_text += f"   • BEARISH FVG: ${fvg['lower']:.4f} - ${fvg['upper']:.4f} (Strength: {fvg['strength']})\n"
            if not fvg_bull and not fvg_bear:
                fvg_text += "   • No unfilled FVGs detected\n"
            
            # Premium/Discount Zones
            current_zone = smc_data.get('current_zone', 'EQUILIBRIUM')
            optimal_action = smc_data.get('optimal_action', 'WAIT')
            zone_pct = smc_data.get('zone_percentage', 50.0)
            pd_text = f"\nPREMIUM/DISCOUNT ANALYSIS:\n"
            pd_text += f"   • Current Zone: {current_zone} ({zone_pct:.1f}%)\n"
            pd_text += f"   • Optimal Action: {optimal_action}\n"
            pd_text += f"   • Premium Level: ${smc_data.get('premium_level', 0):.4f}\n"
            pd_text += f"   • Discount Level: ${smc_data.get('discount_level', 0):.4f}\n"
            
            # Liquidity Levels
            liq_resistance = smc_data.get('liq_resistance', 0)
            liq_support = smc_data.get('liq_support', 0)
            liq_text = f"\nLIQUIDITY LEVELS:\n"
            liq_text += f"   • Resistance: ${liq_resistance:.4f}\n"
            liq_text += f"   • Support: ${liq_support:.4f}\n"
            
            # Break of Structure
            bos_levels = smc_data.get('bos_levels', [])
            structure_breaks = smc_data.get('structure_breaks', 0)
            bos_text = f"\nBREAK OF STRUCTURE:\n"
            bos_text += f"   • Structure Breaks: {structure_breaks}\n"
            if bos_levels:
                bos_text += f"   • Recent BOS Level: ${bos_levels[-1]:.4f}\n"
            
            # SMC Confluence
            confluence_score = smc_data.get('smc_confluence_score', 5.0)
            institutional_bias = smc_data.get('institutional_bias', 'NEUTRAL')
            confluence_text = f"\nSMC CONFLUENCE ANALYSIS:\n"
            confluence_text += f"   • Confluence Score: {confluence_score:.1f}/10\n"
            confluence_text += f"   • Institutional Bias: {institutional_bias}\n"
            
            # Market Context
            current_price = smc_data.get('current_price', 0)
            analysis_time = smc_data.get('analysis_time', 0)
            context_text = f"\nMARKET CONTEXT:\n"
            context_text += f"   • Current Price: ${current_price:.4f}\n"
            context_text += f"   • Analysis Time: {analysis_time:.4f}s\n"
            context_text += f"   • Order Blocks Found: {len(smc_data.get('order_blocks', []))}\n"
            context_text += f"   • FVGs Found: {len(smc_data.get('fair_value_gaps', []))}\n"
            
            return ob_text + fvg_text + pd_text + liq_text + bos_text + confluence_text + context_text
            
        except Exception as e:
            self.logger.error(f"SMC data formatting error: {e}")
            return "SMC analysis data unavailable"
    
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