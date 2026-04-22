"""Docker container management."""

import docker
from typing import Optional
from rich.console import Console

from nnb.utils.logging import get_logger

console = Console()
logger = get_logger(__name__)


class Container:
    """Docker container wrapper."""
    
    def __init__(self, project_id: str, client: docker.DockerClient):
        self.project_id = project_id
        self.client = client
        self.image_tag = f"nnb-{project_id}:latest"
        self._container: Optional[docker.models.containers.Container] = None
    
    def start(self, command: Optional[str] = None, detach: bool = True, **kwargs) -> docker.models.containers.Container:
        """Start a new container."""
        try:
            # Check if image exists
            try:
                self.client.images.get(self.image_tag)
            except docker.errors.ImageNotFound:
                raise ValueError(
                    f"Docker image '{self.image_tag}' not found. "
                    "Run 'nnb env build' first."
                )
            
            # Default container configuration
            container_config = {
                "image": self.image_tag,
                "detach": detach,
                "name": f"nnb-{self.project_id}",
                "volumes": {
                    # Mount data directory (read-only for safety)
                    str(kwargs.get("data_dir", "/tmp")): {
                        "bind": "/data",
                        "mode": "ro"
                    },
                    # Mount workspace directory (read-write)
                    str(kwargs.get("workspace_dir", "/tmp")): {
                        "bind": "/workspace",
                        "mode": "rw"
                    }
                },
                "environment": kwargs.get("environment", {}),
                "remove": kwargs.get("remove", False),
            }
            
            if command:
                container_config["command"] = command
            
            # Remove existing container with same name
            try:
                old_container = self.client.containers.get(f"nnb-{self.project_id}")
                old_container.remove(force=True)
                logger.info(f"Removed old container: nnb-{self.project_id}")
            except docker.errors.NotFound:
                pass
            
            # Start new container
            self._container = self.client.containers.run(**container_config)
            logger.info(f"Started container: {self._container.id[:12]}")
            
            return self._container
            
        except docker.errors.DockerException as e:
            logger.error(f"Failed to start container: {e}")
            raise
    
    def stop(self) -> None:
        """Stop the container."""
        if self._container:
            try:
                self._container.stop()
                logger.info(f"Stopped container: {self._container.id[:12]}")
            except docker.errors.DockerException as e:
                logger.error(f"Failed to stop container: {e}")
                raise
    
    def remove(self, force: bool = False) -> None:
        """Remove the container."""
        if self._container:
            try:
                self._container.remove(force=force)
                logger.info(f"Removed container: {self._container.id[:12]}")
                self._container = None
            except docker.errors.DockerException as e:
                logger.error(f"Failed to remove container: {e}")
                raise
    
    def exec_run(self, command: str, **kwargs) -> tuple:
        """Execute a command in the container."""
        if not self._container:
            raise ValueError("Container not started")
        
        try:
            result = self._container.exec_run(command, **kwargs)
            return result
        except docker.errors.DockerException as e:
            logger.error(f"Failed to execute command: {e}")
            raise
    
    def open_shell(self, workspace_dir: Optional[str] = None, data_dir: Optional[str] = None) -> None:
        """Open an interactive shell in the container."""
        import subprocess
        
        try:
            # Check if container is running
            try:
                container = self.client.containers.get(f"nnb-{self.project_id}")
                if container.status != "running":
                    console.print("[yellow]Container not running. Starting...[/yellow]")
                    if workspace_dir and data_dir:
                        self.start(
                            command="/bin/bash",
                            detach=True,
                            remove=False,
                            workspace_dir=workspace_dir,
                            data_dir=data_dir
                        )
                    else:
                        container.start()
            except docker.errors.NotFound:
                if not workspace_dir or not data_dir:
                    console.print("[red]❌ Container not found and directories not provided[/red]")
                    console.print("💡 Run 'nnb env build' first")
                    raise ValueError("Container not found")
                
                console.print("[yellow]Container not found. Creating...[/yellow]")
                self.start(
                    command="/bin/bash",
                    detach=True,
                    remove=False,
                    workspace_dir=workspace_dir,
                    data_dir=data_dir
                )
            
            console.print(f"\n🐚 Opening shell in container: [bold]nnb-{self.project_id}[/bold]")
            console.print("[dim]Type 'exit' to close the shell[/dim]\n")
            
            # Open interactive shell using docker exec
            subprocess.run(
                ["docker", "exec", "-it", f"nnb-{self.project_id}", "/bin/bash"],
                check=True
            )
            
        except subprocess.CalledProcessError as e:
            console.print(f"[red]❌ Failed to open shell: {e}[/red]")
            raise
        except docker.errors.DockerException as e:
            console.print(f"[red]❌ Docker error: {e}[/red]")
            raise
    
    def get_logs(self, tail: int = 100, follow: bool = False) -> str:
        """Get container logs."""
        if not self._container:
            raise ValueError("Container not started")
        
        try:
            logs = self._container.logs(tail=tail, follow=follow)
            return logs.decode("utf-8")
        except docker.errors.DockerException as e:
            logger.error(f"Failed to get logs: {e}")
            raise


def get_container(project_id: str) -> Container:
    """Get container instance for project."""
    try:
        client = docker.from_env()
        return Container(project_id, client)
    except docker.errors.DockerException as e:
        console.print(f"[red]❌ Failed to connect to Docker: {e}[/red]")
        console.print("\n💡 Make sure Docker is installed and running:")
        console.print("  • Windows/Mac: Docker Desktop")
        console.print("  • Linux: Docker Engine")
        raise
