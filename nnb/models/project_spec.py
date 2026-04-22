"""Project specification models."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class ProjectSpec(BaseModel):
    """Project specification confirmed by user."""
    
    project_type: str = Field(..., description="Type of ML project")
    framework: str = Field(..., description="ML framework to use")
    task: str = Field(..., description="Specific task (classification, regression, etc.)")
    dataset_source: str = Field("custom", description="Dataset source: 'custom' or 'torchvision' or specific dataset name")
    num_classes: Optional[int] = Field(None, description="Number of classes for classification")
    input_shape: Optional[str] = Field(None, description="Expected input shape")
    output_shape: Optional[str] = Field(None, description="Expected output shape")
    pretrained: bool = Field(False, description="Use pretrained model")
    export_format: Optional[str] = Field(None, description="Export format (ONNX, TFLite, etc.)")
    compute_budget: str = Field("medium", description="Compute budget (low, medium, high)")
    latency_priority: str = Field("balanced", description="Latency vs accuracy (speed, balanced, accuracy)")
    additional_requirements: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("project_type")
    def validate_project_type(cls, v: str) -> str:
        """Validate project type."""
        allowed = ["vision", "nlp", "tabular", "time-series", "multimodal", "rl", "other"]
        if v.lower() not in allowed:
            raise ValueError(f"Invalid project_type: {v}. Must be one of {allowed}")
        return v.lower()
    
    @validator("framework")
    def validate_framework(cls, v: str) -> str:
        """Validate framework."""
        allowed = ["pytorch", "tensorflow", "jax", "scikit-learn"]
        if v.lower() not in allowed:
            raise ValueError(f"Invalid framework: {v}. Must be one of {allowed}")
        return v.lower()
    
    @validator("num_classes")
    def validate_num_classes(cls, v: Optional[int]) -> Optional[int]:
        """Validate number of classes."""
        if v is not None and (v < 2 or v > 10000):
            raise ValueError("num_classes must be between 2 and 10000")
        return v


class DataRequirements(BaseModel):
    """Data requirements for the project."""
    
    format: str = Field(..., description="Data format (images, csv, parquet, etc.)")
    min_samples: int = Field(..., description="Minimum number of samples")
    recommended_samples: int = Field(..., description="Recommended number of samples")
    structure: str = Field(..., description="Required folder/file structure")
    preprocessing_auto: list[str] = Field(default_factory=list, description="Auto preprocessing")
    preprocessing_manual: list[str] = Field(default_factory=list, description="Manual preprocessing")
    gotchas: list[str] = Field(default_factory=list, description="Common issues to watch for")


class ValidationIssue(BaseModel):
    """Data validation issue."""
    
    severity: str = Field(..., description="error or warning")
    message: str = Field(..., description="Issue description")
    fix: str = Field(..., description="How to fix the issue")
    
    @validator("severity")
    def validate_severity(cls, v: str) -> str:
        """Validate severity."""
        if v not in ["error", "warning"]:
            raise ValueError("severity must be 'error' or 'warning'")
        return v


class ValidationResult(BaseModel):
    """Result of data validation."""
    
    status: str = Field(..., description="pass, warn, or fail")
    issues: list[ValidationIssue] = Field(default_factory=list)
    class_distribution: Optional[Dict[str, int]] = None
    estimated_training_time: Optional[str] = None
    total_samples: int = 0
    
    @validator("status")
    def validate_status(cls, v: str) -> str:
        """Validate status."""
        if v not in ["pass", "warn", "fail"]:
            raise ValueError("status must be 'pass', 'warn', or 'fail'")
        return v


class MockRunResult(BaseModel):
    """Result of mock training run."""
    
    succeeded: bool
    forward_pass_succeeded: bool = False
    backward_pass_succeeded: bool = False
    optimizer_step_succeeded: bool = False
    loss_value: Optional[float] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    retry_count: int = 0
