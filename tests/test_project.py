"""Test project management."""

import pytest
from pathlib import Path
import tempfile
import shutil

from nnb.orchestrator.project import Project
from nnb.orchestrator.state import State, InvalidStateTransitionError


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp = Path(tempfile.mkdtemp())
    yield temp
    shutil.rmtree(temp)


def test_create_project(temp_dir):
    """Test creating a new project."""
    project = Project.create(base_dir=temp_dir)
    
    assert project.project_id.startswith("nnb-")
    assert project.project_dir.exists()
    assert project.state == State.INIT
    assert (project.project_dir / "workspace").exists()
    assert (project.project_dir / "logs").exists()


def test_load_project(temp_dir):
    """Test loading an existing project."""
    # Create project
    project1 = Project.create(base_dir=temp_dir)
    project_id = project1.project_id
    
    # Load project
    project2 = Project.load(project_id, base_dir=temp_dir)
    
    assert project2.project_id == project_id
    assert project2.state == State.INIT


def test_state_transition(temp_dir):
    """Test state transitions."""
    project = Project.create(base_dir=temp_dir)
    
    # Valid transition
    project.transition_to(State.SCOPING)
    assert project.state == State.SCOPING
    
    # Invalid transition
    with pytest.raises(InvalidStateTransitionError):
        project.transition_to(State.DONE)


def test_state_persistence(temp_dir):
    """Test that state persists across loads."""
    # Create and transition
    project1 = Project.create(base_dir=temp_dir)
    project1.transition_to(State.SCOPING)
    project_id = project1.project_id
    
    # Load and check
    project2 = Project.load(project_id, base_dir=temp_dir)
    assert project2.state == State.SCOPING
