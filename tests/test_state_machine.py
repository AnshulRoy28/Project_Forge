"""Test state machine transitions."""

import pytest
from nnb.orchestrator.state import State, InvalidStateTransitionError


def test_valid_transitions():
    """Test valid state transitions."""
    assert State.INIT.can_transition_to(State.SCOPING)
    assert State.SCOPING.can_transition_to(State.SPEC_CONFIRMED)
    assert State.DATA_REQUIRED.can_transition_to(State.DATA_VALIDATED)
    assert State.ENV_READY.can_transition_to(State.CODE_GENERATED)
    assert State.CODE_GENERATED.can_transition_to(State.MOCK_RUNNING)


def test_invalid_transitions():
    """Test invalid state transitions."""
    assert not State.INIT.can_transition_to(State.TRAINING)
    assert not State.SCOPING.can_transition_to(State.DONE)
    assert not State.DONE.can_transition_to(State.INIT)
    # ENV_READY can transition to MOCK_RUNNING (recovery path when code exists)
    assert State.ENV_READY.can_transition_to(State.MOCK_RUNNING)


def test_code_generated_state():
    """Test CODE_GENERATED state transitions."""
    # Can go to MOCK_RUNNING
    assert State.CODE_GENERATED.can_transition_to(State.MOCK_RUNNING)
    # Cannot skip to TRAINING
    assert not State.CODE_GENERATED.can_transition_to(State.TRAINING)
    # Cannot go back
    assert not State.CODE_GENERATED.can_transition_to(State.ENV_READY)


def test_get_next_action():
    """Test getting next action for each state."""
    action = State.INIT.get_next_action()
    assert "nnb start" in action
    
    action = State.DATA_REQUIRED.get_next_action()
    assert "validate" in action.lower()
    
    action = State.ENV_READY.get_next_action()
    assert "generate" in action.lower()
    
    action = State.CODE_GENERATED.get_next_action()
    assert "mock-run" in action.lower()

