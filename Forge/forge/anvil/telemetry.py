"""
Training telemetry and metrics collection for Forge.
"""

import time
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from datetime import datetime

from forge.core.hardware import get_current_vram_usage, get_gpu_temperature


@dataclass
class TrainingMetrics:
    """Metrics from a single training step."""
    
    step: int
    epoch: float
    loss: float
    learning_rate: float
    vram_used_gb: float
    vram_total_gb: float
    gpu_temp: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass  
class TrainingSession:
    """Complete training session data."""
    
    start_time: str
    model_name: str
    dataset_size: int
    config_summary: dict
    metrics_history: list[TrainingMetrics] = field(default_factory=list)
    end_time: Optional[str] = None
    final_loss: Optional[float] = None
    status: str = "running"  # running, completed, failed, interrupted


class TelemetryCollector:
    """Collects and manages training telemetry."""
    
    def __init__(
        self,
        log_path: Optional[Path] = None,
        callback: Optional[Callable[[TrainingMetrics], None]] = None,
    ):
        """
        Initialize telemetry collector.
        
        Args:
            log_path: Path to save training log (default: training_log.txt)
            callback: Optional callback for each metrics update
        """
        self.log_path = log_path or Path("training_log.txt")
        self.callback = callback
        self.session: Optional[TrainingSession] = None
        self._start_time: float = 0
    
    def start_session(
        self,
        model_name: str,
        dataset_size: int,
        config_summary: dict,
    ) -> None:
        """Start a new training session."""
        self._start_time = time.time()
        
        self.session = TrainingSession(
            start_time=datetime.now().isoformat(),
            model_name=model_name,
            dataset_size=dataset_size,
            config_summary=config_summary,
        )
        
        # Write header to log
        self._write_log(f"=== Training Session Started ===")
        self._write_log(f"Model: {model_name}")
        self._write_log(f"Dataset Size: {dataset_size}")
        self._write_log(f"Time: {self.session.start_time}")
        self._write_log("-" * 40)
    
    def record_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        learning_rate: float,
    ) -> TrainingMetrics:
        """Record metrics for a training step."""
        vram_used, vram_total = get_current_vram_usage()
        gpu_temp = get_gpu_temperature()
        
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            gpu_temp=gpu_temp,
        )
        
        if self.session:
            self.session.metrics_history.append(metrics)
        
        # Log to file
        log_line = (
            f"Step {step:5d} | Epoch {epoch:.2f} | "
            f"Loss {loss:.4f} | LR {learning_rate:.2e} | "
            f"VRAM {vram_used:.1f}/{vram_total:.1f}GB | Temp {gpu_temp}°C"
        )
        self._write_log(log_line)
        
        # Callback if provided
        if self.callback:
            self.callback(metrics)
        
        return metrics
    
    def end_session(self, status: str = "completed", final_loss: Optional[float] = None) -> None:
        """End the current training session."""
        if not self.session:
            return
        
        self.session.end_time = datetime.now().isoformat()
        self.session.status = status
        self.session.final_loss = final_loss
        
        elapsed = time.time() - self._start_time
        
        self._write_log("-" * 40)
        self._write_log(f"=== Training Session {status.upper()} ===")
        self._write_log(f"Duration: {elapsed / 60:.1f} minutes")
        if final_loss:
            self._write_log(f"Final Loss: {final_loss:.4f}")
    
    def save_session(self, path: Optional[Path] = None) -> Path:
        """Save session data to JSON file."""
        if not self.session:
            raise ValueError("No active session to save")
        
        save_path = path or Path("training_session.json")
        
        # Convert metrics to dicts
        session_data = {
            "start_time": self.session.start_time,
            "end_time": self.session.end_time,
            "model_name": self.session.model_name,
            "dataset_size": self.session.dataset_size,
            "config_summary": self.session.config_summary,
            "status": self.session.status,
            "final_loss": self.session.final_loss,
            "metrics_count": len(self.session.metrics_history),
            "metrics_history": [asdict(m) for m in self.session.metrics_history[-100:]],
        }
        
        with open(save_path, "w") as f:
            json.dump(session_data, f, indent=2)
        
        return save_path
    
    def get_summary(self) -> dict:
        """Get summary of current training session."""
        if not self.session:
            return {}
        
        history = self.session.metrics_history
        if not history:
            return {"status": "no data"}
        
        losses = [m.loss for m in history]
        
        return {
            "total_steps": len(history),
            "current_loss": losses[-1] if losses else 0,
            "min_loss": min(losses) if losses else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
            "loss_trend": "decreasing" if len(losses) > 10 and losses[-1] < losses[-10] else "stable",
            "elapsed_minutes": (time.time() - self._start_time) / 60,
        }
    
    def _write_log(self, message: str) -> None:
        """Write a message to the log file."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")
