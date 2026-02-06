"""SQLite-based context storage."""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models.data import (
    DataAnalysis,
    DataSnapshot,
    ExecutionResult,
    ProcessingScript,
    ProcessingSession,
    StorageStats,
)
from ..models.enums import SessionStatus


class ContextStorage:
    """SQLite-based storage for processing context and history."""
    
    MAX_STORAGE_MB = 100
    DB_FILE = "context.db"
    
    def __init__(self, project_dir: str):
        """
        Initialize the context storage.
        
        Args:
            project_dir: Path to the project directory.
        """
        self.project_dir = Path(project_dir)
        self.storage_dir = self.project_dir / ".gemini-processor"
        self.db_path = self.storage_dir / self.DB_FILE
        
        self._connection: Optional[sqlite3.Connection] = None
        self._ensure_storage_dir()
        self._init_database()
    
    def _ensure_storage_dir(self) -> None:
        """Ensure the storage directory exists."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create a database connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._connection.row_factory = sqlite3.Row
            # Enable WAL mode for better concurrency
            self._connection.execute("PRAGMA journal_mode=WAL")
        return self._connection
    
    def _init_database(self) -> None:
        """Initialize the database schema."""
        conn = self._get_connection()
        
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                input_file TEXT NOT NULL,
                output_directory TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                status TEXT NOT NULL,
                data_snapshot TEXT,
                checkpoints TEXT
            );
            
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                analysis_data TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            
            CREATE TABLE IF NOT EXISTS scripts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                script_id TEXT NOT NULL,
                content TEXT NOT NULL,
                description TEXT,
                required_packages TEXT,
                execution_result TEXT,
                executed_at TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
            CREATE INDEX IF NOT EXISTS idx_analyses_session ON analyses(session_id);
            CREATE INDEX IF NOT EXISTS idx_scripts_session ON scripts(session_id);
        """)
        
        conn.commit()
    
    def create_session(
        self,
        input_file: str,
        output_directory: str,
    ) -> str:
        """
        Create a new processing session.
        
        Args:
            input_file: Path to the input file.
            output_directory: Path to the output directory.
            
        Returns:
            The session ID.
        """
        timestamp = datetime.now()
        file_hash = hash(input_file) % 10000
        session_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{file_hash:04d}"
        
        conn = self._get_connection()
        conn.execute(
            """
            INSERT INTO sessions 
            (session_id, input_file, output_directory, created_at, last_accessed, status, checkpoints)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                input_file,
                output_directory,
                timestamp.isoformat(),
                timestamp.isoformat(),
                SessionStatus.ACTIVE.value,
                "[]",
            )
        )
        conn.commit()
        
        # Check storage limits
        self._check_storage_limits()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[ProcessingSession]:
        """Get a session by ID."""
        conn = self._get_connection()
        
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        
        if not row:
            return None
        
        # Parse data snapshot if present
        data_snapshot = None
        if row["data_snapshot"]:
            snapshot_data = json.loads(row["data_snapshot"])
            data_snapshot = DataSnapshot(**snapshot_data)
        
        return ProcessingSession(
            session_id=row["session_id"],
            input_file=row["input_file"],
            output_directory=row["output_directory"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            status=SessionStatus(row["status"]),
            data_snapshot=data_snapshot,
            checkpoints=json.loads(row["checkpoints"]) if row["checkpoints"] else [],
        )
    
    def update_session_status(self, session_id: str, status: SessionStatus) -> None:
        """Update the status of a session."""
        conn = self._get_connection()
        conn.execute(
            "UPDATE sessions SET status = ?, last_accessed = ? WHERE session_id = ?",
            (status.value, datetime.now().isoformat(), session_id)
        )
        conn.commit()
    
    def store_snapshot(self, session_id: str, snapshot: DataSnapshot) -> None:
        """Store a data snapshot for a session."""
        conn = self._get_connection()
        
        snapshot_data = {
            "rows": snapshot.rows,
            "schema": snapshot.schema,
            "file_format": snapshot.file_format,
            "total_rows": snapshot.total_rows,
            "sample_size": snapshot.sample_size,
            "sanitized_fields": snapshot.sanitized_fields,
            "extraction_method": snapshot.extraction_method,
        }
        
        conn.execute(
            "UPDATE sessions SET data_snapshot = ?, last_accessed = ? WHERE session_id = ?",
            (json.dumps(snapshot_data), datetime.now().isoformat(), session_id)
        )
        conn.commit()
    
    def store_analysis(self, session_id: str, analysis: DataAnalysis) -> None:
        """Store a data analysis result."""
        conn = self._get_connection()
        
        analysis_data = {
            "data_quality_issues": analysis.data_quality_issues,
            "suggested_operations": analysis.suggested_operations,
            "column_insights": analysis.column_insights,
            "processing_recommendations": analysis.processing_recommendations,
            "estimated_complexity": analysis.estimated_complexity,
            "sensitive_data_detected": analysis.sensitive_data_detected,
            "recommended_security_level": analysis.recommended_security_level,
        }
        
        conn.execute(
            "INSERT INTO analyses (session_id, created_at, analysis_data) VALUES (?, ?, ?)",
            (session_id, datetime.now().isoformat(), json.dumps(analysis_data))
        )
        conn.commit()
    
    def store_script_result(
        self,
        session_id: str,
        script: ProcessingScript,
        result: ExecutionResult,
    ) -> None:
        """Store a script execution result."""
        conn = self._get_connection()
        
        result_data = {
            "success": result.success,
            "output_data": result.output_data,
            "error_message": result.error_message,
            "execution_time": result.execution_time,
            "output_files": result.output_files,
            "logs": result.logs,
        }
        
        conn.execute(
            """
            INSERT INTO scripts 
            (session_id, script_id, content, description, required_packages, execution_result, executed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                script.script_id,
                script.content,
                script.description,
                json.dumps(script.required_packages),
                json.dumps(result_data),
                datetime.now().isoformat(),
            )
        )
        conn.commit()
    
    def get_analyses(self, session_id: str) -> List[DataAnalysis]:
        """Get all analyses for a session."""
        conn = self._get_connection()
        
        rows = conn.execute(
            "SELECT analysis_data FROM analyses WHERE session_id = ? ORDER BY created_at",
            (session_id,)
        ).fetchall()
        
        analyses = []
        for row in rows:
            data = json.loads(row["analysis_data"])
            analyses.append(DataAnalysis(**data))
        
        return analyses
    
    def get_storage_stats(self) -> StorageStats:
        """Get storage usage statistics."""
        # Get database file size
        db_size_mb = 0
        if self.db_path.exists():
            db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
        
        # Get session count
        conn = self._get_connection()
        session_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        
        # Get oldest session
        oldest_row = conn.execute(
            "SELECT MIN(created_at) FROM sessions"
        ).fetchone()
        oldest_date = None
        if oldest_row[0]:
            oldest_date = datetime.fromisoformat(oldest_row[0])
        
        return StorageStats(
            total_size_mb=db_size_mb,
            session_count=session_count,
            oldest_session_date=oldest_date,
            available_space_mb=self.MAX_STORAGE_MB - db_size_mb,
        )
    
    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """
        Remove sessions older than specified days.
        
        Returns:
            Number of sessions removed.
        """
        conn = self._get_connection()
        cutoff = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        cutoff = cutoff.replace(day=cutoff.day - max_age_days)
        
        # Get sessions to delete
        old_sessions = conn.execute(
            "SELECT session_id FROM sessions WHERE created_at < ?",
            (cutoff.isoformat(),)
        ).fetchall()
        
        count = len(old_sessions)
        
        for row in old_sessions:
            session_id = row["session_id"]
            conn.execute("DELETE FROM analyses WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM scripts WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        conn.commit()
        return count
    
    def _check_storage_limits(self) -> None:
        """Check and enforce storage limits."""
        stats = self.get_storage_stats()
        
        if stats.total_size_mb >= self.MAX_STORAGE_MB:
            # Remove oldest sessions until under limit
            while stats.total_size_mb >= self.MAX_STORAGE_MB * 0.9:
                removed = self.cleanup_old_sessions(max_age_days=7)
                if removed == 0:
                    break
                stats = self.get_storage_stats()
    
    def close(self) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
