"""Container orchestration and lifecycle management."""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import docker
    from docker.models.containers import Container
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    Container = None


@dataclass
class ContainerConfig:
    """Configuration for a Docker container."""
    image: str = "python:3.11-slim"
    memory_limit: str = "4g"
    cpu_count: int = 2
    working_dir: str = "/app"
    network_mode: str = "bridge"  # Allow network for package installation
    user: str = ""  # Run as root in container (container is already isolated)


class ContainerOrchestrator:
    """Manages Docker container lifecycle."""
    
    BASE_IMAGE = "python:3.11-slim"
    
    def __init__(self, config: Optional[ContainerConfig] = None):
        """
        Initialize the container orchestrator.
        
        Args:
            config: Container configuration. Uses defaults if not provided.
        """
        self.config = config or ContainerConfig()
        self._client = None
        self._active_containers: Dict[str, Container] = {}
    
    @property
    def client(self):
        """Get the Docker client, initializing if needed."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError(
                "Docker SDK not installed. Install with: pip install docker"
            )
        
        if self._client is None:
            try:
                self._client = docker.from_env()
                # Verify connection
                self._client.ping()
            except docker.errors.DockerException as e:
                raise RuntimeError(
                    f"Cannot connect to Docker daemon: {e}\n"
                    "Make sure Docker is installed and running."
                )
        
        return self._client
    
    def is_available(self) -> Tuple[bool, str]:
        """
        Check if Docker is available and running.
        
        Returns:
            Tuple of (is_available, message)
        """
        if not DOCKER_AVAILABLE:
            return False, "Docker SDK not installed"
        
        try:
            client = docker.from_env()
            version = client.version()
            docker_version = version.get("Version", "unknown")
            return True, f"Docker {docker_version} available"
        except Exception as e:
            return False, f"Docker not available: {str(e)}"
    
    def ensure_image(self, image: Optional[str] = None) -> bool:
        """
        Ensure the required Docker image is available.
        
        Args:
            image: Image name to pull. Uses default if not provided.
            
        Returns:
            True if image is available, False otherwise.
        """
        image = image or self.config.image
        
        try:
            self.client.images.get(image)
            return True
        except docker.errors.ImageNotFound:
            try:
                self.client.images.pull(image)
                return True
            except Exception:
                return False
    
    def create_container(
        self,
        volumes: Optional[Dict[str, Dict]] = None,
        environment: Optional[Dict[str, str]] = None,
        command: Optional[str] = None,
    ) -> Container:
        """
        Create a new container with the specified configuration.
        
        Args:
            volumes: Volume mappings (host_path -> {bind: container_path, mode: ro/rw})
            environment: Environment variables
            command: Command to run
            
        Returns:
            The created container.
        """
        # Ensure image exists
        self.ensure_image()
        
        # Build container kwargs
        container_kwargs = {
            "image": self.config.image,
            "command": command or "sleep infinity",
            "volumes": volumes,
            "environment": environment,
            "mem_limit": self.config.memory_limit,
            "cpu_count": self.config.cpu_count,
            "working_dir": self.config.working_dir,
            "network_mode": self.config.network_mode,
            "detach": True,
        }
        
        # Only set user if specified
        if self.config.user:
            container_kwargs["user"] = self.config.user
        
        container = self.client.containers.create(**container_kwargs)
        
        self._active_containers[container.id] = container
        return container
    
    def start_container(self, container: Container) -> None:
        """Start a container."""
        container.start()
    
    def execute_in_container(
        self,
        container: Container,
        command: List[str],
        timeout: int = 300,
    ) -> Tuple[int, str]:
        """
        Execute a command in a running container.
        
        Args:
            container: The container to execute in.
            command: Command and arguments to execute.
            timeout: Execution timeout in seconds.
            
        Returns:
            Tuple of (exit_code, output)
        """
        exec_result = container.exec_run(
            command,
            workdir=self.config.working_dir,
            demux=True,
        )
        
        exit_code = exec_result.exit_code
        stdout = exec_result.output[0].decode() if exec_result.output[0] else ""
        stderr = exec_result.output[1].decode() if exec_result.output[1] else ""
        
        output = stdout
        if stderr:
            output += f"\n[STDERR]\n{stderr}"
        
        return exit_code, output
    
    def stop_container(self, container: Container, timeout: int = 10) -> None:
        """Stop a container."""
        try:
            container.stop(timeout=timeout)
        except Exception:
            pass
    
    def remove_container(self, container: Container, force: bool = False) -> None:
        """Remove a container."""
        container_id = container.id
        try:
            container.remove(force=force)
        except Exception:
            pass
        finally:
            self._active_containers.pop(container_id, None)
    
    def cleanup_container(self, container: Container) -> None:
        """Stop and remove a container."""
        self.stop_container(container)
        self.remove_container(container, force=True)
    
    def cleanup_all(self) -> None:
        """Clean up all active containers."""
        for container_id in list(self._active_containers.keys()):
            container = self._active_containers.get(container_id)
            if container:
                self.cleanup_container(container)
    
    def copy_to_container(
        self,
        container: Container,
        src_path: str,
        dest_path: str,
    ) -> bool:
        """
        Copy a file from the host to the container.
        
        Args:
            container: The container to copy to.
            src_path: Source path on host.
            dest_path: Destination path in container.
            
        Returns:
            True if successful, False otherwise.
        """
        import os
        import tarfile
        import io
        
        try:
            # Create a tar archive in memory
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                tar.add(src_path, arcname=os.path.basename(dest_path))
            tar_stream.seek(0)
            
            # Copy to container
            dest_dir = os.path.dirname(dest_path) or '/app'
            container.put_archive(dest_dir, tar_stream)
            return True
        except Exception as e:
            return False
    
    def install_packages(
        self,
        container: Container,
        packages: List[str],
    ) -> Tuple[bool, str]:
        """
        Install Python packages in a container.
        
        Args:
            container: The container to install packages in.
            packages: List of package names to install.
            
        Returns:
            Tuple of (success, output)
        """
        if not packages:
            return True, ""
        
        package_list = " ".join(packages)
        exit_code, output = self.execute_in_container(
            container,
            ["pip", "install", "--quiet", "--no-cache-dir"] + packages,
            timeout=300,
        )
        
        return exit_code == 0, output
    
    def get_container_stats(self, container: Container) -> Dict:
        """Get resource usage statistics for a container."""
        try:
            stats = container.stats(stream=False)
            return {
                "cpu_percent": self._calculate_cpu_percent(stats),
                "memory_mb": stats["memory_stats"].get("usage", 0) / (1024 * 1024),
                "memory_limit_mb": stats["memory_stats"].get("limit", 0) / (1024 * 1024),
            }
        except Exception:
            return {}
    
    def _calculate_cpu_percent(self, stats: Dict) -> float:
        """Calculate CPU percentage from container stats."""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"] -
                stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"] -
                stats["precpu_stats"]["system_cpu_usage"]
            )
            if system_delta > 0:
                cpu_count = len(stats["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))
                return (cpu_delta / system_delta) * cpu_count * 100
        except (KeyError, TypeError):
            pass
        return 0.0
