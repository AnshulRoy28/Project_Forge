"""Generate unique project IDs."""

import uuid
from datetime import datetime


def generate_project_id() -> str:
    """Generate a unique project ID."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"nnb-{timestamp}-{short_uuid}"
