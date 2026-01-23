"""
Gemini API client for Forge.

Uses the new google.genai SDK for Gemini integration.
"""

import os
from typing import Optional, Any
from dataclasses import dataclass

from google import genai
from google.genai import types

from forge.core.security import get_api_key


@dataclass
class GeminiResponse:
    """Response from Gemini API."""
    
    text: str
    thinking: Optional[str] = None
    tokens_used: int = 0


class GeminiBrain:
    """
    Gemini integration for Forge.
    
    Handles all AI reasoning, configuration generation, and monitoring.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Gemini client."""
        self.api_key = api_key or get_api_key() or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "No Gemini API key found. Run 'forge init' to configure or set GEMINI_API_KEY."
            )
        
        # Initialize the new genai client
        self.client = genai.Client(api_key=self.api_key)
        
        # Model names
        self.model_name = "gemini-2.0-flash"
        self.thinking_model_name = "gemini-2.0-flash"  # Use same model, thinking via prompt
    
    def _parse_response(self, response) -> GeminiResponse:
        """Parse a Gemini response."""
        text = ""
        
        try:
            if hasattr(response, 'text'):
                text = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                for part in response.candidates[0].content.parts:
                    text += part.text
        except Exception:
            text = str(response)
        
        tokens = 0
        if hasattr(response, 'usage_metadata'):
            tokens = getattr(response.usage_metadata, 'total_token_count', 0)
        
        return GeminiResponse(
            text=text.strip(),
            tokens_used=tokens,
        )
    
    def reason_sync(self, prompt: str, use_thinking: bool = False) -> GeminiResponse:
        """Synchronous reasoning request to Gemini."""
        model = self.thinking_model_name if use_thinking else self.model_name
        
        # Add thinking instruction if needed
        if use_thinking:
            prompt = f"Think step by step before answering.\n\n{prompt}"
        
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                max_output_tokens=8192,
            ),
        )
        
        return self._parse_response(response)
    
    def analyze_dataset(self, metadata: dict[str, Any]) -> GeminiResponse:
        """Analyze a dataset based on its metadata."""
        from forge.brain.prompts import DATASET_ANALYSIS_PROMPT
        
        prompt = DATASET_ANALYSIS_PROMPT.format(
            filename=metadata.get("filename", "unknown"),
            file_size=metadata.get("file_size", 0),
            num_samples=metadata.get("num_samples", 0),
            columns=metadata.get("columns", []),
            sample_lengths=metadata.get("sample_lengths", {}),
            format=metadata.get("format", "unknown"),
        )
        
        return self.reason_sync(prompt, use_thinking=True)
    
    def generate_training_config(
        self,
        goal: str,
        hardware_profile: dict[str, Any],
        dataset_metadata: Optional[dict[str, Any]] = None,
    ) -> GeminiResponse:
        """Generate an optimized training configuration."""
        from forge.brain.prompts import CONFIG_GENERATION_PROMPT
        
        prompt = CONFIG_GENERATION_PROMPT.format(
            goal=goal,
            gpu_name=hardware_profile.get("gpu_name", "Unknown"),
            vram_gb=hardware_profile.get("vram_gb", 0),
            ram_gb=hardware_profile.get("ram_gb", 0),
            dataset_info=str(dataset_metadata) if dataset_metadata else "Not provided",
        )
        
        return self.reason_sync(prompt, use_thinking=True)
    
    def analyze_training_progress(
        self,
        metrics: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> GeminiResponse:
        """Analyze training progress and provide insights."""
        from forge.brain.prompts import TRAINING_ANALYSIS_PROMPT
        
        prompt = TRAINING_ANALYSIS_PROMPT.format(
            current_step=metrics.get("step", 0),
            current_loss=metrics.get("loss", 0),
            learning_rate=metrics.get("lr", 0),
            vram_used=metrics.get("vram_gb", 0),
            gpu_temp=metrics.get("gpu_temp", 0),
            history_summary=self._summarize_history(history),
        )
        
        return self.reason_sync(prompt)
    
    def diagnose_error(self, error_message: str, context: str = "") -> GeminiResponse:
        """Diagnose an error and suggest fixes."""
        from forge.brain.prompts import ERROR_DIAGNOSIS_PROMPT
        
        prompt = ERROR_DIAGNOSIS_PROMPT.format(
            error=error_message,
            context=context,
        )
        
        return self.reason_sync(prompt, use_thinking=True)
    
    def _summarize_history(self, history: list[dict[str, Any]]) -> str:
        """Summarize training history for the prompt."""
        if not history:
            return "No history available"
        
        recent = history[-10:]
        summary_parts = []
        
        for entry in recent:
            step = entry.get("step", "?")
            loss = entry.get("loss", "?")
            summary_parts.append(f"Step {step}: loss={loss}")
        
        return "\n".join(summary_parts)
    
    def chat(self, message: str) -> GeminiResponse:
        """Simple chat with Gemini."""
        return self.reason_sync(message)


def create_brain(api_key: Optional[str] = None) -> GeminiBrain:
    """Factory function to create a GeminiBrain instance."""
    return GeminiBrain(api_key=api_key)
