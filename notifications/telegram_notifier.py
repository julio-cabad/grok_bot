"""
Telegram Notifier - Spartan Trading System
Send trading notifications to Telegram
"""

import requests
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import json

class TelegramNotifier:
    """
    Telegram Notifier for Trading Events
    
    Sends notifications for:
    - Position opened
    - Position closed (TP/SL/Manual)
    - Trading alerts
    """
    
    def __init__(self, token: str, chat_id: str):
        """Initialize Telegram Notifier"""
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.logger = logging.getLogger("TelegramNotifier")
        
        # Test connection
        self._test_connection()
        
        self.logger.info("📱 Telegram Notifier initialized")
    
    def _test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                bot_info = response.json()
                bot_name = bot_info.get('result', {}).get('username', 'Unknown')
                self.logger.info(f"✅ Telegram bot connected: @{bot_name}")
                return True
            else:
                self.logger.error(f"❌ Telegram bot connection failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Telegram connection error: {str(e)}")
            return False
    
    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram"""
        try:
            url = f"{self.base_url}/sendMessage"
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.debug("📱 Telegram message sent successfully")
                return True
            else:
                self.logger.error(f"❌ Telegram send failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Telegram send error: {str(e)}")
            return False  
  
    def notify_position_opened(self, symbol: str, side: str, entry_price: float, 
                              quantity: float, stop_loss: float, take_profit: float,
                              timeframe: str = "1m", ai_score: Optional[float] = None, 
                              ai_reasoning: Optional[str] = None, smc_data: Optional[dict] = None,
                              tech_data: Optional[dict] = None, analysis_time: Optional[float] = None,
                              ai_provider: Optional[str] = None) -> bool:
        """Send detailed notification when position is opened"""
        try:
            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate position value and metrics
            position_value = entry_price * quantity
            risk = abs(entry_price - stop_loss)
            reward = abs(entry_price - take_profit)
            rr_ratio = reward / risk if risk > 0 else 0
            
            # AI Status and Score
            if ai_score is not None and ai_score >= 7.5:
                ai_status = "✅ APPROVED"
                if ai_score >= 9.0:
                    score_emoji = "💎"
                    confidence_text = "PERFECT"
                elif ai_score >= 8.5:
                    score_emoji = "🟢"
                    confidence_text = "EXCELLENT"
                elif ai_score >= 8.0:
                    score_emoji = "🟢"
                    confidence_text = "HIGH"
                else:
                    score_emoji = "🟡"
                    confidence_text = "GOOD"
            else:
                ai_status = "🔴 HIGH RISK"
                score_emoji = "🔴"
                confidence_text = "HIGH RISK"
            
            # Provider info
            provider_text = f"🤖 {ai_provider.title()}" if ai_provider else "🤖 AI"
            analysis_time_text = f"⏱️ {analysis_time:.1f}s" if analysis_time else ""
            
            # Build SMC Analysis Section
            smc_section = ""
            if smc_data:
                zone = smc_data.get('current_zone', 'EQUILIBRIUM')
                ob_bull = smc_data.get('ob_bull', 0)
                ob_bear = smc_data.get('ob_bear', 0)
                liq_support = smc_data.get('liq_support', 0)
                liq_resistance = smc_data.get('liq_resistance', 0)
                fvg_bull = smc_data.get('fvg_bull', [])
                fvg_bear = smc_data.get('fvg_bear', [])
                confluence_score = smc_data.get('confluence_score', 5.0)
                
                # FVG summary
                fvg_text = ""
                if fvg_bull:
                    fvg_text += f"{len(fvg_bull)} Bullish"
                if fvg_bear:
                    if fvg_text:
                        fvg_text += ", "
                    fvg_text += f"{len(fvg_bear)} Bearish"
                if not fvg_text:
                    fvg_text = "None detected"
                
                smc_section = f"""

📊 <b>SMART MONEY ANALYSIS:</b>
🏛️ <b>Zone:</b> {zone} | 🎯 <b>Confluence:</b> {confluence_score:.1f}/10
📈 <b>Order Blocks:</b> Bull ${ob_bull:.4f} | Bear ${ob_bear:.4f}
⚡ <b>Fair Value Gaps:</b> {fvg_text}
💧 <b>Liquidity:</b> Support ${liq_support:.4f} | Resistance ${liq_resistance:.4f}"""
            
            # Build Technical Analysis Section
            tech_section = ""
            if tech_data:
                macd_hist = tech_data.get('macd_hist', 0)
                macd_x = tech_data.get('macd_x', 'NONE')
                stoch_k = tech_data.get('stoch_k', 50)
                bb_pos = tech_data.get('bb_pos', 50)
                bb_width = tech_data.get('bb_width', 2.0)
                
                # MACD status
                macd_status = "Bullish" if macd_hist > 0 else "Bearish" if macd_hist < 0 else "Neutral"
                
                # Stochastic status  
                if stoch_k >= 80:
                    stoch_status = "Overbought"
                elif stoch_k <= 20:
                    stoch_status = "Oversold"
                else:
                    stoch_status = "Neutral"
                
                # Bollinger status
                if bb_pos >= 80:
                    bb_status = "Upper bias"
                elif bb_pos <= 20:
                    bb_status = "Lower bias"
                else:
                    bb_status = "Middle range"
                
                tech_section = f"""

📈 <b>TECHNICAL INDICATORS:</b>
📊 <b>MACD:</b> {macd_status} ({macd_hist:+.1f}) | Cross: {macd_x}
📈 <b>Stochastic:</b> K={stoch_k:.0f} | Status: {stoch_status}
🎈 <b>Bollinger:</b> {bb_pos:.0f}% ({bb_status}) | Width: {bb_width:.1f}%
⚖️ <b>Weights:</b> SMC 40% | MACD 25% | STOCH 20% | BB 15%"""
            
            # Build AI Decision Section
            ai_decision_section = ""
            if ai_reasoning:
                # Extract key points from reasoning
                reasoning_short = ai_reasoning[:200] + "..." if len(ai_reasoning) > 200 else ai_reasoning
                
                # Estimate win probability based on score
                if ai_score:
                    if ai_score >= 9.0:
                        win_prob = "85-90%"
                        duration = "2-6H"
                    elif ai_score >= 8.0:
                        win_prob = "75-85%"
                        duration = "4-8H"
                    elif ai_score >= 7.5:
                        win_prob = "65-75%"
                        duration = "6-12H"
                    else:
                        win_prob = "50-65%"
                        duration = "8-16H"
                else:
                    win_prob = "Unknown"
                    duration = "Unknown"
                
                ai_decision_section = f"""

🧠 <b>AI REASONING:</b>
💭 "{reasoning_short}"

🎯 <b>Win Probability:</b> {win_prob} | ⏱️ <b>Expected:</b> {duration}"""
            
            # Create comprehensive message
            message = f"""
🚀 <b>POSITION OPENED - {side}</b>
📊 <b>Symbol:</b> {symbol} | ⏰ {timestamp} | 🕐 {timeframe}
💰 <b>Entry:</b> ${entry_price:.4f} | 🛑 <b>SL:</b> ${stop_loss:.4f} | 🎯 <b>TP:</b> ${take_profit:.4f}
⚖️ <b>Risk/Reward:</b> 1:{rr_ratio:.2f} | 💵 <b>Size:</b> ${position_value:.2f}

🏛️ <b>SPARTAN AI CONFLUENCE:</b> {ai_status}
🎯 <b>Score:</b> {ai_score:.1f}/10 | 📊 <b>Confidence:</b> {confidence_text}
{analysis_time_text} | {provider_text}{smc_section}{tech_section}{ai_decision_section}

🏛️ <i>Spartan Trading System</i>
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send position opened notification: {str(e)}")
            return False
    
    def notify_position_closed(self, symbol: str, side: str, entry_price: float,
                              exit_price: float, quantity: float, gross_pnl: float,
                              real_pnl: float, total_commissions: float, close_reason: str,
                              entry_time: datetime, exit_time: datetime,
                              timeframe: str = "1m") -> bool:
        """Send notification when position is closed"""
        try:
            # Format timestamps
            entry_str = entry_time.strftime("%Y-%m-%d %H:%M:%S")
            exit_str = exit_time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate duration
            duration = exit_time - entry_time
            duration_minutes = duration.total_seconds() / 60
            
            # Calculate position value and percentage
            position_value = entry_price * quantity
            pnl_percentage = (real_pnl / position_value) * 100
            
            # Determine emoji based on result
            result_emoji = "💚" if real_pnl >= 0 else "💔"
            side_emoji = "🟢" if side == "LONG" else "🔴"
            
            # Format close reason
            reason_emoji = {
                "TAKE_PROFIT": "🎯",
                "STOP_LOSS": "🛑", 
                "MANUAL": "✋"
            }.get(close_reason, "❓")
            
            # Create message
            message = f"""
{result_emoji} <b>POSITION CLOSED</b>

📊 <b>Symbol:</b> {symbol}
{side_emoji} <b>Side:</b> {side}
🕐 <b>Timeframe:</b> {timeframe}

⏰ <b>Entry:</b> {entry_str}
⏰ <b>Exit:</b> {exit_str}
⏱️ <b>Duration:</b> {duration_minutes:.1f} minutes

💰 <b>Entry Price:</b> ${entry_price:.4f}
💰 <b>Exit Price:</b> ${exit_price:.4f}
📊 <b>Quantity:</b> {quantity:.6f}

💵 <b>Gross PnL:</b> ${gross_pnl:+.3f}
💰 <b>Real PnL:</b> ${real_pnl:+.3f} ({pnl_percentage:+.2f}%)
💸 <b>Commissions:</b> ${total_commissions:.3f}

{reason_emoji} <b>Close Reason:</b> {close_reason}

🏛️ <i>Spartan Trading System</i>
            """.strip()
            
            return self.send_message(message)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send position closed notification: {str(e)}")
            return False
    
    def notify_alert(self, title: str, message: str) -> bool:
        """Send general alert notification"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            alert_message = f"""
🚨 <b>{title}</b>

{message}

⏰ <b>Time:</b> {timestamp}
🏛️ <i>Spartan Trading System</i>
            """.strip()
            
            return self.send_message(alert_message)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send alert notification: {str(e)}")
            return False