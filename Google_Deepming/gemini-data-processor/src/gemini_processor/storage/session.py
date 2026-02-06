"""Session lifecycle management."""

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..models.data import ProcessingSession
from ..models.enums import SessionStatus
from .storage import ContextStorage


class SessionManager:
    """Manages processing session lifecycle."""
    
    def __init__(self, storage: ContextStorage):
        """
        Initialize the session manager.
        
        Args:
            storage: The context storage instance.
        """
        self.storage = storage
        self._active_session: Optional[str] = None
    
    def create_session(
        self,
        input_file: str,
        output_directory: Optional[str] = None,
    ) -> ProcessingSession:
        """
        Create a new processing session.
        
        Args:
            input_file: Path to the input file.
            output_directory: Path to the output directory.
            
        Returns:
            The created ProcessingSession.
        """
        input_path = Path(input_file)
        
        # Default output directory
        if output_directory is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_directory = str(input_path.parent / f"output_{timestamp}")
        
        # Create session in storage
        session_id = self.storage.create_session(
            input_file=str(input_path.absolute()),
            output_directory=output_directory,
        )
        
        self._active_session = session_id
        
        # Ensure output directory exists
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        return self.storage.get_session(session_id)
    
    def get_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Get a session by ID."""
        return self.storage.get_session(session_id)
    
    def get_active_session(self) -> Optional[ProcessingSession]:
        """Get the currently active session."""
        if self._active_session:
            return self.storage.get_session(self._active_session)
        return None
    
    def complete_session(self, session_id: str) -> None:
        """Mark a session as completed."""
        self.storage.update_session_status(session_id, SessionStatus.COMPLETED)
        if self._active_session == session_id:
            self._active_session = None
    
    def fail_session(self, session_id: str) -> None:
        """Mark a session as failed."""
        self.storage.update_session_status(session_id, SessionStatus.FAILED)
        if self._active_session == session_id:
            self._active_session = None
    
    def abandon_session(self, session_id: str) -> None:
        """Mark a session as abandoned."""
        self.storage.update_session_status(session_id, SessionStatus.ABANDONED)
        if self._active_session == session_id:
            self._active_session = None
    
    def get_recent_sessions(self, limit: int = 10) -> List[ProcessingSession]:
        """
        Get recent sessions ordered by creation time.
        
        Args:
            limit: Maximum number of sessions to return.
            
        Returns:
            List of recent sessions.
        """
        conn = self.storage._get_connection()
        
        rows = conn.execute(
            "SELECT session_id FROM sessions ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        
        sessions = []
        for row in rows:
            session = self.storage.get_session(row["session_id"])
            if session:
                sessions.append(session)
        
        return sessions
    
    def cleanup_abandoned_sessions(self, max_age_hours: int = 24) -> int:
        """
        Clean up abandoned sessions older than specified hours.
        
        Returns:
            Number of sessions cleaned up.
        """
        # Convert hours to days for the cleanup function
        max_age_days = max(1, max_age_hours // 24)
        return self.storage.cleanup_old_sessions(max_age_days)
    
    def is_session_recoverable(self, session_id: str) -> bool:
        """
        Check if a session can be recovered.
        
        A session is recoverable if it's in ACTIVE or ABANDONED state
        and has a valid data snapshot.
        """
        session = self.storage.get_session(session_id)
        if not session:
            return False
        
        if session.status not in (SessionStatus.ACTIVE, SessionStatus.ABANDONED):
            return False
        
        return session.data_snapshot is not None
    
    def recover_session(self, session_id: str) -> Optional[ProcessingSession]:
        """
        Attempt to recover an abandoned session.
        
        Returns:
            The recovered session if successful, None otherwise.
        """
        if not self.is_session_recoverable(session_id):
            return None
        
        self.storage.update_session_status(session_id, SessionStatus.ACTIVE)
        self._active_session = session_id
        
        return self.storage.get_session(session_id)
