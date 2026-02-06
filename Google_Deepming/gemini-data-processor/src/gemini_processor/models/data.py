"""Data models for processing operations."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .enums import SessionStatus, ValidationStatus


@dataclass
class DataSnapshot:
    """Represents a sample of the input data for analysis."""
    rows: List[Dict[str, Any]]
    schema: Dict[str, str]  # column_name -> data_type
    file_format: str
    total_rows: int
    sample_size: int
    sanitized_fields: List[str] = field(default_factory=list)
    extraction_method: str = "random"  # 'random', 'first_n', 'stratified'


@dataclass
class DataAnalysis:
    """Results of Gemini AI analysis of the data snapshot."""
    data_quality_issues: List[str]
    suggested_operations: List[str]
    column_insights: Dict[str, str]
    processing_recommendations: List[str]
    estimated_complexity: str
    sensitive_data_detected: bool = False
    recommended_security_level: str = "normal"


@dataclass
class ProcessingScript:
    """Represents a generated script for data processing."""
    script_id: str
    content: str
    description: str
    required_packages: List[str]
    input_files: List[str]
    output_files: List[str]
    estimated_runtime: int = 60  # seconds
    validation_status: ValidationStatus = ValidationStatus.PENDING
    security_level: str = "safe"  # 'safe', 'caution', 'restricted'


@dataclass
class ResourceUsage:
    """Container resource usage statistics."""
    cpu_percent: float
    memory_mb: float
    disk_mb: float
    network_bytes: int
    execution_time: float


@dataclass
class ExecutionResult:
    """Results from script execution in Docker container."""
    success: bool
    output_data: Optional[str] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    output_files: List[str] = field(default_factory=list)
    logs: str = ""
    resource_usage: Optional[ResourceUsage] = None
    container_id: str = ""


@dataclass
class ProcessingContext:
    """Context for AI-driven processing decisions."""
    previous_analyses: List[DataAnalysis] = field(default_factory=list)
    executed_scripts: List[ProcessingScript] = field(default_factory=list)
    current_data_state: Optional[Dict[str, Any]] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Results from script syntax validation."""
    is_valid: bool
    syntax_errors: List[str] = field(default_factory=list)
    security_warnings: List[str] = field(default_factory=list)
    suggested_fixes: List[str] = field(default_factory=list)
    validation_time: float = 0.0


@dataclass
class ProcessingSession:
    """Represents a complete data processing session."""
    session_id: str  # Format: {timestamp}_{hash}
    input_file: str
    output_directory: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    data_snapshot: Optional[DataSnapshot] = None
    scripts_executed: List[ProcessingScript] = field(default_factory=list)
    current_context: ProcessingContext = field(default_factory=ProcessingContext)
    status: SessionStatus = SessionStatus.ACTIVE
    checkpoints: List[str] = field(default_factory=list)


@dataclass
class StorageStats:
    """Context storage usage statistics."""
    total_size_mb: float
    session_count: int
    oldest_session_date: Optional[datetime] = None
    available_space_mb: float = 100.0
