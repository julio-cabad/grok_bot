#!/usr/bin/env python3
"""
Grok-4 AI Client - Backup para Gemini
Cliente para X.AI Grok-4 API como fallback cuando Gemini falla
"""

import os
import json
import requests
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

class GrokClient:
    """Cliente para Grok-4 API (X.AI)"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Grok-4 client"""
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load API key
        self.api_key = api_key or os.getenv('GROK_API_KEY')
        
        if not self.api_key:
            raise ValueError("GROK_API_KEY not found in environment")
        
        # Configure Grok-4
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.model = "grok-4"
        self.timeout = 180  # Increased timeout for complex prompts
        
        self.logger.info(f"🤖 Grok-4 client initialized - Model: {self.model}")
    
    def query(self, prompt: str, temperature: float = 0.7) -> str:
        """Send query to Grok-4 API"""
        
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "model": self.model,
                "stream": False,
                "temperature": temperature,
                "max_tokens": 1000
            }
            
            self.logger.debug(f"🚀 Sending query to Grok-4: {prompt[:100]}...")
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content']
                
                self.logger.info(f"✅ Grok-4 query successful - Response length: {len(content)}")
                return content
                
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                self.logger.error(f"❌ Grok-4 API error: {error_msg}")
                raise Exception(error_msg)
                
        except requests.exceptions.Timeout:
            error_msg = f"Grok-4 query timeout after {self.timeout}s"
            self.logger.error(f"⏰ {error_msg}")
            raise Exception(error_msg)
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Grok-4 request failed: {str(e)}"
            self.logger.error(f"🌐 {error_msg}")
            raise Exception(error_msg)
            
        except Exception as e:
            error_msg = f"Grok-4 query failed: {str(e)}"
            self.logger.error(f"💀 {error_msg}")
            raise
    
    def build_unified_prompt(self, symbol: str, signal_type: str, smc: dict, tech: dict, df) -> str:
        """
        UNIFIED OPTIMAL PROMPT - IDENTICAL to ai_validator_ultra.py
        Ensures 100% consistent analysis between Gemini and Grok-4
        """
        
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
        
        # UNIFIED OPTIMAL PROMPT - IDENTICAL to ai_validator_ultra.py
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

    def validate_trading_signal(self, df, symbol: str, signal_type: str, smc: dict = None, tech: dict = None) -> dict:
        """
        Validate trading signal using Grok-4 with IDENTICAL prompt as Gemini
        100% consistent analysis between both AIs
        """
        
        try:
            # If SMC and tech data not provided, extract from df (fallback)
            if not smc or not tech:
                self.logger.warning("⚠️ SMC/Tech data missing, using basic extraction")
                
                # Basic extraction (fallback)
                latest = df.iloc[-1]
                close_price = float(latest['close'])
                
                smc = {
                    'current_price': close_price,
                    'current_zone': 'EQUILIBRIUM',
                    'ob_bull': close_price * 0.98,
                    'ob_bear': close_price * 1.02,
                    'liq_support': close_price * 0.95,
                    'liq_resistance': close_price * 1.05,
                    'fvg_bull': [],
                    'fvg_bear': []
                }
                
                stoch_k = float(latest.get('STOCH_K_14_3', 50))
                macd_hist = float(latest.get('MACD_hist_12_26_9', 0))
                bb_upper = float(latest.get('BB_upper_20', close_price * 1.02))
                bb_lower = float(latest.get('BB_lower_20', close_price * 0.98))
                bb_position = ((close_price - bb_lower) / (bb_upper - bb_lower)) * 100 if bb_upper != bb_lower else 50
                bb_width = ((bb_upper - bb_lower) / close_price) * 100
                
                tech = {
                    'close_price': close_price,
                    'macd_hist': macd_hist,  # Keep as float for consistency
                    'macd_x': 'BULL' if macd_hist > 0 else 'BEAR',
                    'stoch_k': stoch_k,  # Keep as float for consistency
                    'bb_pos': bb_position,  # Keep as float for consistency
                    'bb_width': bb_width
                }
            
            # Build IDENTICAL prompt as ai_validator_ultra.py
            prompt = self.build_unified_prompt(symbol, signal_type, smc, tech, df)
            
            self.logger.info(f"🔄 Using IDENTICAL unified prompt for {symbol} {signal_type}")
            
            # Query Grok-4 with unified prompt
            response = self.query(prompt, temperature=0.3)
            
            return {
                "analysis": response,
                "model_used": self.model,
                "provider": "grok-4",
                "timestamp": "2025-09-29"
            }
            
        except Exception as e:
            self.logger.error(f"❌ Grok-4 trading validation failed: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if Grok-4 API is healthy"""
        
        try:
            response = self.query("Respond with 'HEALTHY' if you can process this.", temperature=0.1)
            
            if "HEALTHY" in response.upper():
                self.logger.info("✅ Grok-4 health check passed")
                return True
            else:
                self.logger.warning("⚠️ Grok-4 health check: unexpected response")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Grok-4 health check failed: {e}")
            return False