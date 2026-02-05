"""Data analysis and snapshot creation."""

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models.data import DataSnapshot
from ..models.enums import FileFormat


class DataAnalyzer:
    """Handles data structure analysis and snapshot creation."""
    
    SAMPLE_SIZE = 100
    
    def __init__(self, file_path: str):
        """
        Initialize the data analyzer.
        
        Args:
            file_path: Path to the input data file.
        """
        self.file_path = Path(file_path)
        self._file_format: Optional[FileFormat] = None
        self._total_rows: int = 0
    
    @property
    def file_format(self) -> FileFormat:
        """Detect and return the file format."""
        if self._file_format is None:
            suffix = self.file_path.suffix.lower()
            if suffix == ".csv":
                self._file_format = FileFormat.CSV
            elif suffix == ".json":
                self._file_format = FileFormat.JSON
            elif suffix in (".txt", ".text"):
                self._file_format = FileFormat.TEXT
            else:
                self._file_format = FileFormat.UNKNOWN
        return self._file_format
    
    def create_snapshot(self) -> DataSnapshot:
        """
        Create a data snapshot by sampling rows from the input file.
        
        Returns:
            DataSnapshot containing sample rows and schema information.
        """
        if self.file_format == FileFormat.CSV:
            return self._create_csv_snapshot()
        elif self.file_format == FileFormat.JSON:
            return self._create_json_snapshot()
        elif self.file_format == FileFormat.TEXT:
            return self._create_text_snapshot()
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
    
    def _create_csv_snapshot(self) -> DataSnapshot:
        """Create snapshot from CSV file."""
        rows: List[Dict[str, Any]] = []
        schema: Dict[str, str] = {}
        
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            # Count total rows first (efficiently)
            total_rows = sum(1 for _ in f) - 1  # Subtract header
            self._total_rows = max(0, total_rows)
        
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames:
                # Initialize schema with unknown types
                schema = {field: "unknown" for field in reader.fieldnames}
            
            all_rows = list(reader)
            
            # Sample rows
            if len(all_rows) <= self.SAMPLE_SIZE:
                sampled_rows = all_rows
            else:
                sampled_rows = random.sample(all_rows, self.SAMPLE_SIZE)
            
            rows = sampled_rows
            
            # Infer schema types from sampled data
            schema = self._infer_schema(rows, schema)
        
        return DataSnapshot(
            rows=rows,
            schema=schema,
            file_format=self.file_format.value,
            total_rows=self._total_rows,
            sample_size=len(rows),
            extraction_method="random" if len(rows) < self._total_rows else "all"
        )
    
    def _create_json_snapshot(self) -> DataSnapshot:
        """Create snapshot from JSON file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, list):
            all_rows = data
        elif isinstance(data, dict):
            # Look for common array keys
            for key in ['data', 'records', 'items', 'results', 'rows']:
                if key in data and isinstance(data[key], list):
                    all_rows = data[key]
                    break
            else:
                # Treat the dict as a single row
                all_rows = [data]
        else:
            all_rows = [{"value": data}]
        
        self._total_rows = len(all_rows)
        
        # Convert rows to dicts if needed
        rows = []
        for row in all_rows:
            if isinstance(row, dict):
                rows.append(row)
            else:
                rows.append({"value": row})
        
        # Sample rows
        if len(rows) > self.SAMPLE_SIZE:
            sampled_rows = random.sample(rows, self.SAMPLE_SIZE)
        else:
            sampled_rows = rows
        
        # Infer schema
        if sampled_rows:
            schema = {k: "unknown" for k in sampled_rows[0].keys()}
            schema = self._infer_schema(sampled_rows, schema)
        else:
            schema = {}
        
        return DataSnapshot(
            rows=sampled_rows,
            schema=schema,
            file_format=self.file_format.value,
            total_rows=self._total_rows,
            sample_size=len(sampled_rows),
            extraction_method="random" if len(sampled_rows) < self._total_rows else "all"
        )
    
    def _create_text_snapshot(self) -> DataSnapshot:
        """Create snapshot from text file."""
        lines: List[str] = []
        
        with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        
        self._total_rows = len(lines)
        
        # Sample lines
        if len(lines) > self.SAMPLE_SIZE:
            sampled_lines = random.sample(lines, self.SAMPLE_SIZE)
        else:
            sampled_lines = lines
        
        # Convert to row format
        rows = [{"line": i + 1, "content": line.rstrip()} for i, line in enumerate(sampled_lines)]
        
        return DataSnapshot(
            rows=rows,
            schema={"line": "integer", "content": "string"},
            file_format=self.file_format.value,
            total_rows=self._total_rows,
            sample_size=len(rows),
            extraction_method="random" if len(rows) < self._total_rows else "all"
        )
    
    def _infer_schema(
        self, 
        rows: List[Dict[str, Any]], 
        schema: Dict[str, str]
    ) -> Dict[str, str]:
        """Infer data types from sample rows."""
        type_counts: Dict[str, Dict[str, int]] = {col: {} for col in schema}
        
        for row in rows:
            for col, value in row.items():
                if col not in type_counts:
                    type_counts[col] = {}
                
                inferred_type = self._infer_value_type(value)
                type_counts[col][inferred_type] = type_counts[col].get(inferred_type, 0) + 1
        
        # Pick most common type for each column
        for col in schema:
            if col in type_counts and type_counts[col]:
                schema[col] = max(type_counts[col], key=type_counts[col].get)
        
        return schema
    
    def _infer_value_type(self, value: Any) -> str:
        """Infer the type of a single value."""
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return "null"
        
        if isinstance(value, bool):
            return "boolean"
        
        if isinstance(value, int):
            return "integer"
        
        if isinstance(value, float):
            return "float"
        
        if isinstance(value, str):
            # Try to parse as number
            try:
                if '.' in value:
                    float(value)
                    return "float"
                else:
                    int(value)
                    return "integer"
            except ValueError:
                pass
            
            # Check for boolean strings
            if value.lower() in ('true', 'false', 'yes', 'no', '1', '0'):
                return "boolean"
            
            # Check for date-like patterns
            if any(sep in value for sep in ['-', '/']) and len(value) >= 8:
                return "date"
            
            return "string"
        
        if isinstance(value, (list, tuple)):
            return "array"
        
        if isinstance(value, dict):
            return "object"
        
        return "unknown"
