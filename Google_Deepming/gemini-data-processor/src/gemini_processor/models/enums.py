"""Enumeration types for type safety."""

from enum import Enum


class SessionStatus(Enum):
    """Processing session status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ValidationStatus(Enum):
    """Script validation status."""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    PENDING = "pending"


class FileFormat(Enum):
    """Supported input file formats."""
    CSV = "csv"
    JSON = "json"
    TEXT = "text"
    UNKNOWN = "unknown"
