"""Gemini API client."""

import os
from typing import Optional, Dict, Any
import google.genai as genai

from nnb.utils.logging import get_logger
from nnb.utils.api_key_manager import APIKeyManager

logger = get_logger(__name__)


class GeminiClient:
    """Client for interacting with Gemini API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        # Priority: explicit api_key > environment variable > keyring
        self.api_key = api_key or APIKeyManager.get_api_key()
        
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not configured.\n"
                "Run: nnb config setup"
            )
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model
        logger.info(f"Initialized Gemini client with model: {model}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate response from Gemini."""
        try:
            logger.debug(f"Sending prompt to Gemini (length: {len(prompt)})")
            
            config = {
                "temperature": temperature,
            }
            
            if max_tokens:
                config["max_output_tokens"] = max_tokens
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            result = response.text
            logger.debug(f"Received response (length: {len(result)})")
            
            return result
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}", exc_info=True)
            raise
    
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """Generate JSON response from Gemini."""
        import json
        
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nRespond with valid JSON only, no markdown formatting."
        
        response = self.generate(json_prompt, temperature=temperature)
        
        # Strip markdown code blocks if present
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        
        try:
            return json.loads(response.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            raise ValueError(f"Invalid JSON response from Gemini: {e}")
