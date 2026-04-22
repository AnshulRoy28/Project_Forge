"""Test state machine transitions."""

import pytest
from nnb.orchestrator.state import State, InvalidStateTransitionError


def test_valid_transitions():
    """Test valid state transitions."""
    assert State.INIT.can_transition_to(State.SCOPING)
    assert State.SCOPING.can_transition_to(State.SPEC_CONFIRMED)
    assert State.DATA_REQUIRED.can_transition_to(State.DATA_VALIDATED)


def test_invalid_transitions():
    """Test invalid state transitions."""
    assert not State.INIT.can_transition_to(State.TRAINING)
    assert not State.SCOPING.can_transition_to(State.DONE)
    assert not State.DONE.can_transition_to(State.INIT)


def test_get_next_action():
    """Test getting next action for each state."""
    action = State.INIT.get_next_action()
    assert "nnb start" in action
    
    action = State.DATA_REQUIRED.get_next_action()
    assert "validate" in action.lower()
