#!/usr/bin/env python3
"""
Gemini AI Client - Spartan Code Edition
High-performance, scalable client for Google Gemini AI API
"""

import os
import logging
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GeminiClient:
    """
    Spartan Gemini AI Client
    Minimal, powerful, and battle-tested
    """

    def __init__(self, model: str = "gemini-2.5-pro", api_key: Optional[str] = None):
        """
        Initialize Gemini client

        Args:
            model: Gemini model to use (default: gemini-2.5-pro)
            api_key: API key (loads from env if None)
        """
        self.model = model
        self.logger = logging.getLogger("GeminiClient")

        # Load API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")

        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.client = genai.GenerativeModel(model_name=self.model)

        self.logger.info(f"🏛️ Gemini Client initialized with model {model}")

    def query(self, message: str, **kwargs) -> str:
        """
        Execute a single query to Gemini AI

        Args:
            message: User message
            **kwargs: Additional parameters

        Returns:
            AI response content
        """
        try:
            response = self.client.generate_content(message, **kwargs)
            content = response.text
            self.logger.info("✅ Query executed successfully")
            return content

        except Exception as e:
            self.logger.error(f"💀 Query failed: {str(e)}")
            raise

    def analyze_market_data(self, data_summary: str) -> Dict[str, Any]:
        """
        Analyze market data using Gemini AI

        Args:
            data_summary: Summary of market data

        Returns:
            Analysis dict with insights
        """
        prompt = f"""
        Analyze this cryptocurrency market data and provide insights:

        {data_summary}

        Provide:
        1. Trend analysis
        2. Key levels
        3. Risk assessment
        4. Trading recommendations

        Be concise and actionable.
        """

        response = self.query(prompt)

        return {
            "analysis": response,
            "model_used": self.model,
            "timestamp": "2025-09-25"  # Would use datetime in production
        }
