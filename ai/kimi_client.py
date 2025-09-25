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
            self.logger.info("Query executed successfully")
            return content

        except Exception as e:
            self.logger.error(f"💀 Query failed: {str(e)}")
            raise

    def analyze_market_data(self, data_summary: str) -> Dict[str, Any]:
        """
        Analyze market data using Gemini AI with institutional analysis

        Args:
            data_summary: Summary of market data with technical indicators

        Returns:
            Analysis dict with insights
        """
        prompt = f"""
        COMO TRADER INSTITUCIONAL EXPERTO, valida esta SEÑAL DE TRADING generada por algoritmo y determina las probabilidades de éxito de entrada.
        
        SEÑAL A VALIDAR:
        {data_summary}
        
        INSTRUCCIONES DE VALIDACIÓN:
        
        1. VALIDAR SETUP TÉCNICO: Confirma si los indicadores apoyan la señal (Trend Magic, Squeeze, RSI, etc.)
        
        2. ANÁLISIS DE RIESGO: Evalúa el Risk/Reward ratio y posición del stop loss
        
        3. PROBABILIDADES INSTITUCIONALES: Basado en Smart Money Concepts, estima win rate (>70% alta, 50-70% media, <50% baja)
        
        4. CONDICIONES DE ENTRADA: Define el mejor momento para entrar (precio específico, confirmación requerida)
        
        5. RECOMENDACIÓN FINAL: APROBAR o RECHAZAR la entrada con justificación
        
        FORMATO DE RESPUESTA:
        VALIDACIÓN SETUP: [APROBADO/RECHAZADO] - Justificación
        PROBABILIDAD ÉXITO: [ALTA/MEDIA/BAJA] - % estimado
        PRECIO ENTRADA ÓPTIMO: [precio específico]
        STOP LOSS VALIDACIÓN: [precio]
        TAKE PROFIT SUGERIDO: [precio]
        CONDICIONES CONFIRMACIÓN: [lista de eventos]
        RECOMENDACIÓN: [ENTRAR/ESPERAR/RECHAZAR] - Razón final
        
        Sé conservador: mejor perder oportunidad que tomar mala entrada.
        """
        response = self.query(prompt)
        return {
            "analysis": response,
            "model_used": self.model,
            "timestamp": "2025-09-25"
        }
