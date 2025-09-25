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
        Analyze market data using Gemini AI with institutional analysis

        Args:
            data_summary: Summary of market data with technical indicators

        Returns:
            Analysis dict with insights
        """
        prompt = f"""
        COMO TRADER INSTITUCIONAL CON 10+ AÑOS DE EXPERIENCIA, ESPECIALIZADO EN SMART MONEY CONCEPTS (SMC) Y OPERACIONES DE ALTO WIN RATE (>70%).
        Analiza los datos técnicos proporcionados de BTC/USD en timeframe de 4h (últimas 500 velas) y genera un diagnóstico claro, conservador y accionable.
        Tu prioridad es la preservación de capital: mejor perder una oportunidad que tomar una mala.
        
        📊 DATOS DISPONIBLES
        {data_summary}
        
        🔍 INSTRUCCIONES DE ANÁLISIS
        
        FAIR VALUE GAPS (FVG)
        Identifica FVGs no rellenados (alcistas y bajistas).
        Prioriza aquellos que coinciden con:
        Niveles psicológicos (ej. 110k, 115k, 120k)
        Swing points de las últimas 100 velas
        Zonas de alto volumen (> percentil 80 del último mes)
        Ignora FVGs en rango lateral sin confirmación.
        
        SWEEPS DE LIQUIDEZ
        Detecta sweeps válidos: wick rompe soporte/resistencia + cierre en sentido opuesto + volumen > 25% del promedio de 20 velas.
        Clasifícalos como:
        Bull trap: sweep arriba + cierre bajista
        Bear trap: sweep abajo + cierre alcista
        Confirma con divergencia oculta o cambio en MagicTrend.
        
        DIVERGENCIAS OCULTAS
        Solo considera si ocurren en zonas de FVG o tras un sweep.
        Alcista: precio hace nuevo mínimo, RSI/MACD hace higher low.
        Bajista: precio hace nuevo máximo, RSI/MACD hace lower high.
        Ignora divergencias en rango sin edge.
        
        MOMENTUM Y VOLATILIDAD
        Si Squeeze = BLACK y BB Width < 2.5%, etiqueta como "pre-expansión" → no operar hasta ruptura con volumen.
        Si Momentum = MAROON, asume falta de impulso direccional.
        Usa MagicTrend:
        📤 FORMATO DE SALIDA
        
        PRIMERO: Evalúa si hay SETUP VÁLIDO con al menos 3 confirmaciones.
        
        Si SÍ hay setup válido:
        [SEÑAL SMC – {{fecha}} {{hora}} UTC]
        • Tipo: Long / Short
        • Zona de entrada: {{rango}}
        • Trigger: {{evento}}
        • Stop loss: {{precio}} ({{justificación}})
        • Take profit 1: {{precio}} ({{zona}})
        • Take profit 2: {{precio}} ({{zona}})
        • R:R = {{ratio}}
        • Probabilidad: Alta/Media/Baja
        • Confirmación requerida: {{descripción}}
        • Nota: {{contexto}}
        
        SEGUNDO: SIEMPRE proporciona ANÁLISIS ESTADÍSTICO para ambas direcciones:
        
        📊 ANÁLISIS LONG:
        • Mejor precio de entrada: {{precio óptimo basado en datos}}
        • Punto estadístico: {{por qué este precio}}
        • Probabilidad de éxito: {{% basado en historical edge}}
        • Nivel de riesgo: Alto/Medio/Bajo
        • Próxima zona de liquidez: {{precio}}
        
        📊 ANÁLISIS SHORT:
        • Mejor precio de entrada: {{precio óptimo}}
        • Punto estadístico: {{por qué}}
        • Probabilidad de éxito: {{%}}
        • Nivel de riesgo: Alto/Medio/Bajo
        • Próxima zona de liquidez: {{precio}}
        
        Si NO hay setup claro, responde únicamente:
        "Esperar – No hay edge estadístico. Mercado en rango sin señales SMC." 
        
        PERO SIEMPRE incluye el ANÁLISIS ESTADÍSTICO LONG y SHORT al final.
        
        Recuerda: La paciencia es tu mayor aliado. Mejor 5 trades perfectos al año que 50 mediocres.
                
        [SEÑAL SMC – {{fecha}} {{hora}} UTC]
        • Tipo: Long / Short / Esperar  
        • Zona de entrada: {{rango de precio, ej. 111.000 – 111.600}}  
        • Trigger: {{descripción clara del evento: sweep + FVG + divergencia}}  
        • Stop loss: {{precio exacto}} (justificación: fuera de liquidez / swing)  
        • Take profit 1: {{precio}} (zona de liquidez / POC)  
        • Take profit 2: {{precio}} (zona de liquidez opuesta)  
        • R:R = {{ratio calculado}}  
        • Probabilidad: Alta (70–80%) / Media (55–70%) / Baja (<55%)  
        • Confirmación requerida: cierre de vela de 4h en zona + volumen > promedio  
        • Nota: {{contexto adicional: sesión, eventos, sesgo de MagicTrend}}
        
        Si no hay setup claro con al menos 3 confirmaciones, responde únicamente:
        "Esperar – No hay edge estadístico. El mercado está en rango sin señales de Smart Money." 
        
        Recuerda: La paciencia es tu mayor aliado. Mejor 5 trades perfectos al año que 50 mediocres.
        """
        response = self.query(prompt)

        return {
            "analysis": response,
            "model_used": self.model,
            "timestamp": "2025-09-25"
        }
