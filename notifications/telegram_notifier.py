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
                              timeframe: str = "1m") -> bool:
        """Send notification when position is opened"""
        try:
            # Format timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Calculate position value
            position_value = entry_price * quantity
            
            # Create message
            message = f"""
🚀 <b>POSITION OPENED</b>

📊 <b>Symbol:</b> {symbol}
📈 <b>Side:</b> {side}
⏰ <b>Time:</b> {timestamp}
🕐 <b>Timeframe:</b> {timeframe}

💰 <b>Entry Price:</b> ${entry_price:.4f}
📊 <b>Quantity:</b> {quantity:.6f}
💵 <b>Position Value:</b> ${position_value:.2f}

🛑 <b>Stop Loss:</b> ${stop_loss:.4f}
🎯 <b>Take Profit:</b> ${take_profit:.4f}

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