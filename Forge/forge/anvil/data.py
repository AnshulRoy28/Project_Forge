"""
Dataset handling for Forge.
"""

import json
from pathlib import Path
from typing import Any

from forge.core.config import DataConfig


def load_dataset(config: DataConfig) -> Any:
    """Load a dataset from the configured path. Supports: CSV, JSON, JSONL, TXT."""
    from datasets import Dataset
    
    path = Path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    
    format_type = config.format.lower()
    if format_type == "auto":
        format_type = _detect_format(path)
    
    loaders = {"csv": _load_csv, "json": _load_json, "jsonl": _load_jsonl, "txt": _load_txt}
    if format_type not in loaders:
        raise ValueError(f"Unsupported format: {format_type}")
    
    dataset = loaders[format_type](path, config.text_column)
    
    if config.shuffle:
        dataset = dataset.shuffle(seed=config.seed)
    
    # Limit samples for quick validation
    if config.max_samples and len(dataset) > config.max_samples:
        dataset = dataset.select(range(config.max_samples))
    
    return dataset


def _detect_format(path: Path) -> str:
    """Detect dataset format from file extension."""
    suffix = path.suffix.lower()
    format_map = {".csv": "csv", ".json": "json", ".jsonl": "jsonl", ".txt": "txt"}
    if suffix in format_map:
        return format_map[suffix]
    
    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    
    if first_line.startswith("{"):
        return "jsonl"
    elif first_line.startswith("["):
        return "json"
    elif "," in first_line:
        return "csv"
    return "txt"


def _load_csv(path: Path, text_column: str) -> Any:
    """Load a CSV dataset."""
    from datasets import Dataset
    import csv
    
    texts = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_column) or " ".join(str(v) for v in row.values() if v)
            texts.append({"text": text})
    return Dataset.from_list(texts)


def _load_json(path: Path, text_column: str) -> Any:
    """Load a JSON dataset."""
    from datasets import Dataset
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        raise ValueError("JSON must contain a list of items")
    
    texts = [{"text": _extract_text(item, text_column)} for item in data]
    return Dataset.from_list(texts)


def _load_jsonl(path: Path, text_column: str) -> Any:
    """Load a JSONL dataset."""
    from datasets import Dataset
    
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                texts.append({"text": _extract_text(item, text_column)})
            except json.JSONDecodeError:
                continue
    return Dataset.from_list(texts)


def _load_txt(path: Path, text_column: str = None) -> Any:
    """Load a text file dataset."""
    from datasets import Dataset
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [content.strip()] if content.strip() else []
    
    return Dataset.from_list([{"text": p} for p in paragraphs])


def _extract_text(item: Any, text_column: str) -> str:
    """Extract text from a data item."""
    if isinstance(item, dict):
        if text_column in item:
            return item[text_column]
        elif "text" in item:
            return item["text"]
        elif "messages" in item:
            return _format_messages(item["messages"])
        else:
            return json.dumps(item)
    return str(item)


def _format_messages(messages: list) -> str:
    """Format a list of messages into training text (ChatML format)."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    
    return "\n".join(parts)
