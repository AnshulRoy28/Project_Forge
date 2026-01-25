"""
Container Manager for Forge.

Manages persistent Docker containers with model caching for faster training resumption.
"""

import subprocess
import time
import json
import datetime
from typing import Optional, Dict, Any
from pathlib import Path
import datetime

from forge.ui.console import console, print_success, print_error, print_info, print_warning


class ContainerManager:
    """Manages persistent Docker containers for model caching."""
    
    def __init__(self, config):
        """Initialize container manager with config."""
        self.config = config
        self.container_name = f"forge-{config.model.name.replace('/', '-').replace('_', '-').lower()}"
        self.image_name = f"forge:{config.hardware.gpu_arch.value}"
    
    def check_container_exists(self, container_id: Optional[str] = None) -> bool:
        """Check if a container exists and is accessible."""
        if not container_id and not self.config.container.container_id:
            return False
        
        check_id = container_id or self.config.container.container_id
        
        try:
            result = subprocess.run(
                ["docker", "inspect", check_id],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def get_container_status(self, container_id: Optional[str] = None) -> Optional[str]:
        """Get container status (running, exited, etc.)."""
        if not container_id and not self.config.container.container_id:
            return None
        
        check_id = container_id or self.config.container.container_id
        
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Status}}", check_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        
        return None
    
    def start_persistent_container(self) -> Optional[str]:
        """Start a persistent container for model caching."""
        console.print("[bold]🐳 Starting persistent container for model caching...[/]")
        
        # Check if we have an existing container
        if self.config.container.container_id:
            status = self.get_container_status()
            
            if status == "running":
                print_info(f"Using existing running container: {self.config.container.container_id[:12]}")
                return self.config.container.container_id
            
            elif status == "exited":
                print_info(f"Restarting existing container: {self.config.container.container_id[:12]}")
                if self._restart_container():
                    return self.config.container.container_id
                else:
                    print_warning("Failed to restart container, creating new one...")
        
        # Create new persistent container
        return self._create_new_container()
    
    def _restart_container(self) -> bool:
        """Restart an existing container."""
        try:
            result = subprocess.run(
                ["docker", "start", self.config.container.container_id],
                capture_output=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # Wait for container to be ready
                time.sleep(2)
                return self.get_container_status() == "running"
            
        except Exception as e:
            print_error(f"Failed to restart container: {e}")
        
        return False
    
    def _create_new_container(self) -> Optional[str]:
        """Create a new persistent container."""
        try:
            # Check if container with this name already exists
            check_cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.ID}}"]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
            
            if check_result.returncode == 0 and check_result.stdout.strip():
                # Container exists, try to use it
                existing_id = check_result.stdout.strip()
                console.print(f"[dim]Found existing container: {existing_id}[/]")
                
                # Update config with existing container
                self.config.container.container_id = existing_id
                self.config.container.container_name = self.container_name
                
                # Try to start it if not running
                status = self.get_container_status()
                if status != "running":
                    if self._restart_container():
                        console.print("✓ Reused existing container")
                        return existing_id
                    else:
                        # Container can't be restarted - remove it and create new one
                        console.print("[dim]Removing stuck container and creating new one...[/]")
                        try:
                            subprocess.run(
                                ["docker", "rm", "-f", existing_id],
                                capture_output=True,
                                timeout=30
                            )
                            # Clear the container config
                            self.config.container.container_id = None
                            self.config.container.model_cached = False
                        except Exception as e:
                            print_warning(f"Failed to remove container: {e}")
                else:
                    console.print("✓ Reused running container")
                    return existing_id
            
            # Create persistent HuggingFace cache directory
            hf_cache_dir = Path.home() / ".cache" / "huggingface"
            hf_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Build docker run command for persistent container
            cmd = [
                "docker", "run",
                "--name", self.container_name,
                "--gpus", "all",
                "--detach",  # Run in background
                "--interactive",  # Keep stdin open
                "--tty",  # Allocate pseudo-TTY
                "-v", f"{Path('./data').resolve()}:/data",
                "-v", f"{Path('./output').resolve()}:/output",
                "-v", f"{Path('./checkpoints').resolve()}:/checkpoints",
                "-v", f"{hf_cache_dir}:/root/.cache/huggingface",  # Persistent HF cache
            ]
            
            # Mount forge source code for training
            forge_package_dir = Path(__file__).parent.parent.parent.resolve()
            cmd.extend(["-v", f"{forge_package_dir}:/app/forge_src"])
            cmd.extend(["-e", "PYTHONPATH=/app/forge_src"])
            cmd.extend(["-e", "HF_HOME=/root/.cache/huggingface"])
            
            # Pass session credentials
            from forge.core.security import get_api_key, get_hf_token
            
            session_api_key = get_api_key()
            if session_api_key:
                cmd.extend(["-e", f"GEMINI_API_KEY={session_api_key}"])
            
            session_hf_token = get_hf_token()
            if session_hf_token:
                cmd.extend(["-e", f"HF_TOKEN={session_hf_token}"])
            
            # Set working directory and keep container alive
            cmd.extend(["-w", "/app"])
            cmd.extend([self.image_name, "tail", "-f", "/dev/null"])  # Keep container alive
            
            console.print(f"[dim]Creating container: {self.container_name}[/]")
            console.print(f"[dim]HuggingFace cache: {hf_cache_dir}[/]")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                container_id = result.stdout.strip()
                
                # Wait a moment for container to fully start
                time.sleep(3)
                
                # Verify container is actually running
                status = self.get_container_status(container_id)
                if status != "running":
                    # Container failed to start - get logs for debugging
                    try:
                        logs_result = subprocess.run(
                            ["docker", "logs", container_id],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        print_error(f"Container failed to start. Status: {status}")
                        if logs_result.stdout:
                            console.print(f"[dim]Container logs:[/]")
                            console.print(f"[dim]{logs_result.stdout}[/]")
                        if logs_result.stderr:
                            console.print(f"[dim]Container errors:[/]")
                            console.print(f"[dim]{logs_result.stderr}[/]")
                    except Exception:
                        pass
                    
                    # Clean up failed container
                    try:
                        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=10)
                    except Exception:
                        pass
                    
                    return None
                
                # Update config with new container info
                self.config.container.container_id = container_id
                self.config.container.container_name = self.container_name
                self.config.container.gpu_arch = self.config.hardware.gpu_arch.value
                self.config.container.last_used = datetime.datetime.now().isoformat()
                self.config.container.model_cached = False  # Will be set to True after first model load
                
                print_success(f"Created persistent container with model server: {container_id[:12]}")
                
                # Wait a bit more for model server to be ready
                time.sleep(2)
                
                return container_id
            
            else:
                print_error(f"Failed to create container: {result.stderr}")
                return None
                
        except Exception as e:
            print_error(f"Error creating container: {e}")
            return None
        """Create a new persistent container."""
        try:
            # Check if container with this name already exists
            check_cmd = ["docker", "ps", "-a", "--filter", f"name={self.container_name}", "--format", "{{.ID}}"]
            check_result = subprocess.run(check_cmd, capture_output=True, text=True, timeout=10)
            
            if check_result.returncode == 0 and check_result.stdout.strip():
                # Container exists, try to use it
                existing_id = check_result.stdout.strip()
                console.print(f"[dim]Found existing container: {existing_id}[/]")
                
                # Update config with existing container
                self.config.container.container_id = existing_id
                self.config.container.container_name = self.container_name
                
                # Try to start it if not running
                status = self.get_container_status()
                if status != "running":
                    if self._restart_container():
                        console.print("✓ Reused existing container")
                        return existing_id
                    else:
                        # Container can't be restarted - remove it and create new one
                        console.print("[dim]Removing stuck container and creating new one...[/]")
                        try:
                            subprocess.run(
                                ["docker", "rm", "-f", existing_id],
                                capture_output=True,
                                timeout=30
                            )
                            # Clear the container config
                            self.config.container.container_id = None
                            self.config.container.model_cached = False
                        except Exception as e:
                            print_warning(f"Failed to remove container: {e}")
                else:
                    console.print("✓ Reused running container")
                    return existing_id
            
            # Build docker run command for persistent container
            cmd = [
                "docker", "run",
                "--name", self.container_name,
                "--gpus", "all",
                "--detach",  # Run in background
                "--interactive",  # Keep stdin open
                "--tty",  # Allocate pseudo-TTY
                "-v", f"{Path('./data').resolve()}:/data",
                "-v", f"{Path('./output').resolve()}:/output",
                "-v", f"{Path('./checkpoints').resolve()}:/checkpoints",
            ]
            
            # Mount forge source code for training
            forge_package_dir = Path(__file__).parent.parent.parent.resolve()
            cmd.extend(["-v", f"{forge_package_dir}:/app/forge_src"])
            cmd.extend(["-e", "PYTHONPATH=/app/forge_src"])
            
            # Pass session credentials
            from forge.core.security import get_api_key, get_hf_token
            
            session_api_key = get_api_key()
            if session_api_key:
                cmd.extend(["-e", f"GEMINI_API_KEY={session_api_key}"])
            
            session_hf_token = get_hf_token()
            if session_hf_token:
                cmd.extend(["-e", f"HF_TOKEN={session_hf_token}"])
            
            # Set working directory and keep container alive
            cmd.extend(["-w", "/app"])
            cmd.extend([self.image_name, "tail", "-f", "/dev/null"])  # Alternative to sleep infinity
            
            console.print(f"[dim]Creating container: {self.container_name}[/]")
            console.print(f"[dim]Command: docker run --name {self.container_name} --gpus all -d ... {self.image_name} tail -f /dev/null[/]")
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                container_id = result.stdout.strip()
                
                # Wait a moment for container to fully start
                time.sleep(2)
                
                # Verify container is actually running
                status = self.get_container_status(container_id)
                if status != "running":
                    # Container failed to start - get logs for debugging
                    try:
                        logs_result = subprocess.run(
                            ["docker", "logs", container_id],
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                        print_error(f"Container failed to start. Status: {status}")
                        if logs_result.stdout:
                            console.print(f"[dim]Container logs:[/]")
                            console.print(f"[dim]{logs_result.stdout}[/]")
                        if logs_result.stderr:
                            console.print(f"[dim]Container errors:[/]")
                            console.print(f"[dim]{logs_result.stderr}[/]")
                    except Exception:
                        pass
                    
                    # Clean up failed container
                    try:
                        subprocess.run(["docker", "rm", "-f", container_id], capture_output=True, timeout=10)
                    except Exception:
                        pass
                    
                    return None
                
                # Update config with new container info
                self.config.container.container_id = container_id
                self.config.container.container_name = self.container_name
                self.config.container.gpu_arch = self.config.hardware.gpu_arch.value
                self.config.container.last_used = datetime.datetime.now().isoformat()
                self.config.container.model_cached = False  # Will be set to True after first model load
                
                print_success(f"Created persistent container: {container_id[:12]}")
                return container_id
            
            else:
                print_error(f"Failed to create container: {result.stderr}")
                return None
                
        except Exception as e:
            print_error(f"Error creating container: {e}")
            return None
    
    def execute_in_container(self, command: list, timeout: int = 1800) -> subprocess.Popen:
        """Execute a command in the persistent container."""
        if not self.config.container.container_id:
            raise ValueError("No container available")
        
        # Ensure container is running and healthy
        status = self.get_container_status()
        if status != "running":
            print_warning(f"Container status: {status}, attempting to restart...")
            
            # Try to get container logs for debugging
            try:
                logs_result = subprocess.run(
                    ["docker", "logs", "--tail", "20", self.config.container.container_id],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if logs_result.stdout or logs_result.stderr:
                    console.print(f"[dim]Recent container logs:[/]")
                    if logs_result.stdout:
                        console.print(f"[dim]{logs_result.stdout}[/]")
                    if logs_result.stderr:
                        console.print(f"[dim]{logs_result.stderr}[/]")
            except Exception:
                pass
            
            if not self._restart_container():
                raise ValueError("Container is not running and cannot be restarted")
            # Wait a bit for container to be fully ready
            time.sleep(3)
        
        # Build docker exec command
        exec_cmd = [
            "docker", "exec",
            "-i",  # Interactive
            self.config.container.container_id
        ] + command
        
        console.print(f"[dim]$ docker exec {self.config.container.container_id[:12]} {' '.join(command)}[/]")
        
        # Start the process
        process = subprocess.Popen(
            exec_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        
        return process
    
    def cache_model_in_container(self) -> bool:
        """Pre-load and cache the model in the container."""
        if not self.config.container.container_id:
            return False
        
        if self.config.container.model_cached:
            print_info("Model already cached in container")
            return True
        
        console.print("[bold]📦 Model files will be cached via HuggingFace cache volume[/]")
        
        # Mark model as cached since we're using persistent HF cache
        self.config.container.model_cached = True
        self.config.container.cached_model_name = self.config.model.name
        self.config.container.last_used = datetime.datetime.now().isoformat()
        
        print_success("Model caching enabled via persistent HuggingFace cache!")
        return True
    
    def cleanup_container(self):
        """Clean up the persistent container."""
        if not self.config.container.container_id:
            return
        
        try:
            # Stop and remove container
            subprocess.run(
                ["docker", "stop", self.config.container.container_id],
                capture_output=True,
                timeout=30
            )
            
            subprocess.run(
                ["docker", "rm", self.config.container.container_id],
                capture_output=True,
                timeout=30
            )
            
            # Clear container info from config
            self.config.container.container_id = None
            self.config.container.container_name = None
            self.config.container.model_cached = False
            self.config.container.cached_model_name = None
            
            print_info("Container cleaned up")
            
        except Exception as e:
            print_warning(f"Error cleaning up container: {e}")
    
    def get_container_info(self) -> Dict[str, Any]:
        """Get information about the current container."""
        if not self.config.container.container_id:
            return {"status": "none"}
        
        try:
            # Get detailed container info
            result = subprocess.run(
                ["docker", "inspect", self.config.container.container_id],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                info = json.loads(result.stdout)[0]
                
                return {
                    "status": info["State"]["Status"],
                    "created": info["Created"],
                    "started": info["State"]["StartedAt"],
                    "image": info["Config"]["Image"],
                    "name": info["Name"].lstrip("/"),
                    "id": self.config.container.container_id[:12],
                    "model_cached": self.config.container.model_cached,
                }
            
        except Exception:
            pass
        
        return {"status": "unknown"}