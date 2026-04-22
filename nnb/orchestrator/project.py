"""Project management and orchestration."""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from nnb.orchestrator.state import State, InvalidStateTransitionError
from nnb.models.project_spec import (
    ProjectSpec,
    DataRequirements,
    ValidationResult,
    MockRunResult,
)
from nnb.utils.id_generator import generate_project_id
from nnb.utils.logging import get_logger

logger = get_logger(__name__)


class Project:
    """Manages a neural network project through its lifecycle."""
    
    def __init__(self, project_id: str, project_dir: Path):
        self.project_id = project_id
        self.project_dir = project_dir
        self.state_file = project_dir / "state.json"
        self.spec_file = project_dir / "steering-doc.yaml"
        self.data_requirements_file = project_dir / "data-requirements.md"
        self.data_manifest_file = project_dir / "data-manifest.json"
        
        self._state: State = State.INIT
        self._spec: Optional[ProjectSpec] = None
        self._load_state()
    
    @classmethod
    def create(cls, base_dir: Path = Path.cwd()) -> "Project":
        """Create a new project."""
        project_id = generate_project_id()
        project_dir = base_dir / ".nnb" / project_id
        
        # Create directory structure
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "workspace").mkdir(exist_ok=True)
        (project_dir / "logs").mkdir(exist_ok=True)
        
        project = cls(project_id, project_dir)
        project._save_state()
        
        logger.info(f"Created project {project_id} at {project_dir}")
        return project
    
    @classmethod
    def load(cls, project_id: str, base_dir: Path = Path.cwd()) -> "Project":
        """Load an existing project."""
        project_dir = base_dir / ".nnb" / project_id
        
        if not project_dir.exists():
            raise FileNotFoundError(f"Project {project_id} not found")
        
        project = cls(project_id, project_dir)
        logger.info(f"Loaded project {project_id}")
        return project
    
    @classmethod
    def load_from_current_dir(cls) -> "Project":
        """Load project from current directory."""
        nnb_dir = Path.cwd() / ".nnb"
        
        if not nnb_dir.exists():
            raise FileNotFoundError("No .nnb directory found")
        
        # Find most recent project
        projects = sorted(nnb_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not projects:
            raise FileNotFoundError("No projects found")
        
        project_id = projects[0].name
        return cls.load(project_id)
    
    @property
    def state(self) -> State:
        """Get current state."""
        return self._state
    
    def transition_to(self, next_state: State) -> None:
        """Transition to next state."""
        if not self._state.can_transition_to(next_state):
            raise InvalidStateTransitionError(self._state, next_state)
        
        logger.info(f"Transitioning from {self._state.value} to {next_state.value}")
        self._state = next_state
        self._save_state()
    
    def get_next_action(self) -> str:
        """Get next action for current state."""
        return self._state.get_next_action()
    
    def _load_state(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            with open(self.state_file) as f:
                data = json.load(f)
                self._state = State(data["state"])
                
                if "spec" in data and data["spec"]:
                    self._spec = ProjectSpec(**data["spec"])
    
    def _save_state(self) -> None:
        """Save state to file."""
        data = {
            "project_id": self.project_id,
            "state": self._state.value,
            "spec": self._spec.dict() if self._spec else None,
        }
        
        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def start_conversation(self) -> None:
        """Start Stage 1: User Conversation."""
        from nnb.stages.stage_01_conversation import start_conversation
        
        self.transition_to(State.SCOPING)
        start_conversation(self)
    
    def continue_from_state(self) -> None:
        """Continue project from current state."""
        if self._state == State.SCOPING:
            self.start_conversation()
        elif self._state == State.SPEC_CONFIRMED:
            from nnb.stages.stage_03_data_requirements import generate_data_requirements
            generate_data_requirements(self)
        elif self._state == State.DATA_REQUIRED:
            print("💡 Run: nnb data validate --path <data-path>")
        elif self._state == State.DATA_VALIDATED:
            print("💡 Run: nnb env build")
        elif self._state == State.ENV_READY:
            print("💡 Run: nnb mock-run")
        elif self._state == State.MOCK_PASSED:
            print("💡 Run: nnb train")
        elif self._state == State.TRAINING:
            print("💡 Run: nnb attach")
        else:
            print(f"💡 {self.get_next_action()}")
    
    def validate_data(self, data_path: Optional[Path] = None) -> ValidationResult:
        """Stage 4: Validate training data."""
        from nnb.stages.stage_04_data_validation import validate_data
        
        if self._state != State.DATA_REQUIRED:
            raise InvalidStateTransitionError(self._state, State.DATA_VALIDATED)
        
        # Check if using torchvision dataset
        dataset_source = self._spec.dataset_source
        if dataset_source in ["torchvision", "MNIST", "CIFAR10", "CIFAR100", "FashionMNIST", "ImageNet"]:
            # Skip validation for torchvision datasets
            result = ValidationResult(
                status="pass",
                issues=[],
                total_samples=0,
                estimated_training_time="Will be determined during training"
            )
            self.transition_to(State.DATA_VALIDATED)
            return result
        
        # For custom datasets, require data_path
        if data_path is None:
            raise ValueError("data_path is required for custom datasets")
        
        result = validate_data(self, data_path)
        
        if result.status != "fail":
            self.transition_to(State.DATA_VALIDATED)
        
        return result
    
    def get_data_requirements(self) -> str:
        """Get data requirements document."""
        if self.data_requirements_file.exists():
            return self.data_requirements_file.read_text()
        return "Data requirements not yet generated"
    
    def build_environment(self) -> None:
        """Stage 5: Build Docker environment."""
        from nnb.stages.stage_05_environment import build_environment
        
        if self._state != State.DATA_VALIDATED:
            raise InvalidStateTransitionError(self._state, State.ENV_BUILDING)
        
        self.transition_to(State.ENV_BUILDING)
        build_environment(self)
        self.transition_to(State.ENV_READY)
    
    def open_shell(self) -> None:
        """Open shell in Docker container."""
        from nnb.docker_runtime.container import get_container
        
        container = get_container(self.project_id)
        workspace_dir = str(self.project_dir / "workspace")
        data_dir = str(self.project_dir / "data")
        
        # Create data directory if it doesn't exist
        (self.project_dir / "data").mkdir(exist_ok=True)
        
        container.open_shell(workspace_dir=workspace_dir, data_dir=data_dir)
    
    def run_mock(self) -> MockRunResult:
        """Stage 6: Run mock training."""
        from nnb.stages.stage_06_code_generation import run_mock_training
        
        if self._state != State.ENV_READY and self._state != State.MOCK_RUNNING:
            raise InvalidStateTransitionError(self._state, State.MOCK_RUNNING)
        
        self.transition_to(State.MOCK_RUNNING)
        result = run_mock_training(self)
        
        if result.succeeded:
            self.transition_to(State.MOCK_PASSED)
        
        return result
    
    def start_training(self) -> None:
        """Stage 7: Start training."""
        from nnb.stages.stage_07_training import start_training
        
        if self._state != State.MOCK_PASSED:
            raise InvalidStateTransitionError(self._state, State.TRAINING)
        
        self.transition_to(State.TRAINING)
        start_training(self)
    
    def attach_to_training(self) -> None:
        """Attach to running training."""
        from nnb.stages.stage_07_training import attach_to_training
        
        if self._state != State.TRAINING:
            raise ValueError("No training in progress")
        
        attach_to_training(self)
