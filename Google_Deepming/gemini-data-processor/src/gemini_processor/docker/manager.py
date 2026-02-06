"""Docker manager for script execution."""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Set

from ..models.config import ResourceConfig
from ..models.data import ExecutionResult, ProcessingScript, ResourceUsage
from .container import ContainerConfig, ContainerOrchestrator


class DockerSession:
    """Represents a persistent Docker container session."""
    
    def __init__(
        self,
        container,
        orchestrator: ContainerOrchestrator,
        input_file: str,
        output_dir: str,
    ):
        self.container = container
        self.orchestrator = orchestrator
        self.input_file = input_file
        self.output_dir = output_dir
        self.installed_packages: Set[str] = set()
        self._active = True
    
    @property
    def is_active(self) -> bool:
        return self._active and self.container is not None
    
    def install_packages(self, packages: List[str]) -> tuple[bool, str]:
        """Install packages if not already installed."""
        # Filter out already installed packages
        new_packages = [p for p in packages if p not in self.installed_packages]
        
        if not new_packages:
            return True, "All packages already installed"
        
        success, output = self.orchestrator.install_packages(
            self.container,
            new_packages,
        )
        
        if success:
            self.installed_packages.update(new_packages)
        
        return success, output
    
    def execute_script(self, script_content: str, timeout: int) -> tuple[int, str]:
        """Execute a script in the persistent container."""
        # Write script to a temp file and copy to container
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as script_file:
            script_file.write(script_content)
            script_path = script_file.name
        
        try:
            # Copy script to container
            self.orchestrator.copy_to_container(
                self.container,
                script_path,
                '/app/current_script.py'
            )
            
            # Execute the script
            exit_code, output = self.orchestrator.execute_in_container(
                self.container,
                ['python', '/app/current_script.py'],
                timeout=timeout,
            )
            
            return exit_code, output
            
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass
    
    def get_stats(self) -> Optional[dict]:
        """Get container resource statistics."""
        return self.orchestrator.get_container_stats(self.container)
    
    def close(self) -> None:
        """Close and cleanup the session."""
        if self._active and self.container:
            self.orchestrator.cleanup_container(self.container)
            self._active = False
            self.container = None


class DockerManager:
    """Manages Docker containers for script execution."""
    
    def __init__(self, resource_config: Optional[ResourceConfig] = None):
        """
        Initialize the Docker manager.
        
        Args:
            resource_config: Resource limits configuration.
        """
        self.resource_config = resource_config or ResourceConfig()
        
        container_config = ContainerConfig(
            memory_limit=f"{self.resource_config.max_container_memory_gb}g",
            cpu_count=self.resource_config.max_container_cpu_cores,
        )
        
        self.orchestrator = ContainerOrchestrator(container_config)
        self._current_session: Optional[DockerSession] = None
    
    def is_available(self) -> tuple[bool, str]:
        """Check if Docker is available."""
        return self.orchestrator.is_available()
    
    def start_session(
        self,
        input_file: str,
        output_dir: str,
    ) -> Optional[DockerSession]:
        """
        Start a persistent container session.
        
        Args:
            input_file: Path to the input data file.
            output_dir: Directory for output files.
            
        Returns:
            DockerSession if successful, None otherwise.
        """
        # Close existing session if any
        self.end_session()
        
        # Check Docker availability
        available, msg = self.is_available()
        if not available:
            return None
        
        input_path = Path(input_file)
        output_path = Path(output_dir)
        
        # Prepare volume mappings
        volumes = {
            str(input_path.parent.absolute()): {
                'bind': '/data',
                'mode': 'ro'  # Read-only input
            },
            str(output_path.absolute()): {
                'bind': '/output',
                'mode': 'rw'
            },
        }
        
        # Environment variables
        environment = {
            'INPUT_FILE': f'/data/{input_path.name}',
            'OUTPUT_DIR': '/output',
            'PYTHONUNBUFFERED': '1',
        }
        
        # Create and start container
        container = self.orchestrator.create_container(
            volumes=volumes,
            environment=environment,
        )
        
        self.orchestrator.start_container(container)
        
        self._current_session = DockerSession(
            container=container,
            orchestrator=self.orchestrator,
            input_file=input_file,
            output_dir=output_dir,
        )
        
        return self._current_session
    
    def get_session(self) -> Optional[DockerSession]:
        """Get the current session."""
        return self._current_session
    
    def end_session(self) -> None:
        """End the current session and cleanup."""
        if self._current_session:
            self._current_session.close()
            self._current_session = None
    
    def execute_script(
        self,
        script: ProcessingScript,
        input_file: str,
        output_dir: str,
        timeout: Optional[int] = None,
    ) -> ExecutionResult:
        """
        Execute a processing script in a Docker container.
        
        Uses persistent session if available, otherwise creates a new one.
        
        Args:
            script: The script to execute.
            input_file: Path to the input data file.
            output_dir: Directory for output files.
            timeout: Execution timeout in seconds.
            
        Returns:
            ExecutionResult with execution outcome.
        """
        timeout = timeout or (self.resource_config.max_execution_time_minutes * 60)
        
        # Use existing session or create a new one
        session = self._current_session
        if session is None or not session.is_active:
            session = self.start_session(input_file, output_dir)
            if session is None:
                return ExecutionResult(
                    success=False,
                    error_message="Failed to create Docker session"
                )
        
        output_path = Path(output_dir)
        
        try:
            # Install required packages (only installs new ones)
            if script.required_packages:
                success, pkg_output = session.install_packages(script.required_packages)
                if not success:
                    return ExecutionResult(
                        success=False,
                        error_message=f"Failed to install packages: {pkg_output}",
                        container_id=session.container.id if session.container else None,
                    )
            
            # Execute the script
            exit_code, output = session.execute_script(script.content, timeout)
            
            # Get resource usage
            stats = session.get_stats()
            resource_usage = ResourceUsage(
                cpu_percent=stats.get('cpu_percent', 0),
                memory_mb=stats.get('memory_mb', 0),
                disk_mb=0,
                network_bytes=0,
                execution_time=0,
            ) if stats else None
            
            # Check for output files
            output_files = [
                str(f) for f in output_path.iterdir()
                if f.is_file()
            ]
            
            if exit_code == 0:
                return ExecutionResult(
                    success=True,
                    output_data=output,
                    output_files=output_files,
                    logs=output,
                    resource_usage=resource_usage,
                    container_id=session.container.id if session.container else None,
                )
            else:
                return ExecutionResult(
                    success=False,
                    error_message=f"Script exited with code {exit_code}",
                    output_data=output,
                    logs=output,
                    resource_usage=resource_usage,
                    container_id=session.container.id if session.container else None,
                )
                
        except Exception as e:
            return ExecutionResult(
                success=False,
                error_message=str(e),
                container_id=session.container.id if session.container else None,
            )
    
    def test_execution(self) -> ExecutionResult:
        """
        Test Docker execution with a simple script.
        
        Returns:
            ExecutionResult indicating if Docker execution works.
        """
        test_script = ProcessingScript(
            script_id="test",
            content='print("Docker execution test successful!")',
            description="Test script",
            required_packages=[],
            input_files=[],
            output_files=[],
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dummy input file
            input_file = Path(temp_dir) / "test_input.txt"
            input_file.write_text("test data")
            
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()
            
            result = self.execute_script(
                script=test_script,
                input_file=str(input_file),
                output_dir=str(output_dir),
                timeout=60,
            )
            
            # Clean up session after test
            self.end_session()
            
            return result
    
    def cleanup_all_containers(self) -> None:
        """Force cleanup of all managed containers."""
        self.end_session()
        self.orchestrator.cleanup_all()
